import React, { useCallback, useEffect, useRef, useState } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import { ColorMode, ScalarMode } from '@kitware/vtk.js/Rendering/Core/Mapper/Constants';
import vtkOrientationMarkerWidget from '@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget';
import vtkAnnotatedCubeActor from '@kitware/vtk.js/Rendering/Core/AnnotatedCubeActor';
import { Box, Compass, Crosshair, Layers, Pause, Play } from 'lucide-react';
import { fetchVtpBuffer } from '../api';
import { TRANSLATIONS, type Lang } from '../i18n';
import type { F16FemSimResponse, F16FemViewState, FemFieldId } from '../types';

interface F16FemViewerProps {
  lang: Lang;
  result: F16FemSimResponse | null;
  view: F16FemViewState;
  onTogglePlay: () => void;
  loading: boolean;
}

type RGB = [number, number, number];

const FIELD_PALETTES: Record<FemFieldId, { stops: Array<[number, RGB]>; css: string }> = {
  // Inferno/Ember stress map: deep indigo -> plum -> fiery orange -> incandescent gold-white
  von_mises_mpa: {
    stops: [
      [0.0, [26, 20, 56]],
      [0.24, [106, 28, 122]],
      [0.52, [212, 66, 56]],
      [0.78, [249, 152, 24]],
      [1.0, [254, 248, 178]],
    ],
    css: 'linear-gradient(90deg, #1a1438 0%, #6a1c7a 24%, #d44238 52%, #f99818 78%, #fef8b2 100%)',
  },
  // Viridis/Teal-Gold strain map
  strain_micro: {
    stops: [
      [0.0, [24, 30, 74]],
      [0.28, [34, 110, 148]],
      [0.56, [48, 182, 134]],
      [0.82, [196, 224, 58]],
      [1.0, [254, 243, 152]],
    ],
    css: 'linear-gradient(90deg, #181e4a 0%, #226e94 28%, #30b686 56%, #c4e03a 82%, #fef398 100%)',
  },
  // Displacement cobalt -> cyan -> amber -> starlight gold
  disp_mm: {
    stops: [
      [0.0, [24, 44, 112]],
      [0.3, [46, 148, 232]],
      [0.62, [110, 224, 210]],
      [0.84, [245, 166, 35]],
      [1.0, [254, 240, 138]],
    ],
    css: 'linear-gradient(90deg, #182c70 0%, #2e94e8 30%, #6ee0d2 62%, #f5a623 84%, #fef08a 100%)',
  },
};

interface DeformableMesh {
  poly: ReturnType<typeof vtkPolyData.newInstance>;
  basePts: Float32Array;
  dispVec: Float32Array;
  curPts: Float32Array;
}

interface SceneHandle {
  genericRW: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
  femActor: ReturnType<typeof vtkActor.newInstance>;
  femMapper: ReturnType<typeof vtkMapper.newInstance>;
  ghostActor: ReturnType<typeof vtkActor.newInstance>;
  ghostMapper: ReturnType<typeof vtkMapper.newInstance>;
  sliceActor: ReturnType<typeof vtkActor.newInstance>;
  sliceMapper: ReturnType<typeof vtkMapper.newInstance>;
  lut: ReturnType<typeof vtkColorTransferFunction.newInstance>;
  orientWidget: ReturnType<typeof vtkOrientationMarkerWidget.newInstance> | null;
  mesh: DeformableMesh | null;
  dataLoaded: boolean;
}

const vtpCache = new Map<string, ArrayBuffer>();

async function getVtp(url: string): Promise<ArrayBuffer> {
  const hit = vtpCache.get(url);
  if (hit) return hit;
  const buf = await fetchVtpBuffer(url);
  vtpCache.set(url, buf);
  return buf;
}

function parseVtp(buf: ArrayBuffer) {
  const r = vtkXMLPolyDataReader.newInstance();
  r.parseAsArrayBuffer(buf);
  return r.getOutputData(0);
}

function applyStopsToLut(
  lut: ReturnType<typeof vtkColorTransferFunction.newInstance>,
  stops: Array<[number, RGB]>,
  minVal: number,
  maxVal: number
) {
  lut.removeAllPoints();
  const span = Math.max(maxVal - minVal, 1e-4);
  for (const [pos, [r, g, b]] of stops) {
    lut.addRGBPoint(minVal + pos * span, r / 255, g / 255, b / 255);
  }
  lut.setMappingRange(minVal, maxVal);
  lut.updateRange();
}

function applyDeformation(mesh: DeformableMesh, scale: number) {
  const { basePts, dispVec, curPts } = mesh;
  for (let i = 0; i < curPts.length; i++) {
    curPts[i] = basePts[i] + dispVec[i] * scale;
  }
  mesh.poly.getPoints().modified();
  mesh.poly.modified();
}

export const F16FemViewer: React.FC<F16FemViewerProps> = ({
  lang,
  result,
  view,
  onTogglePlay,
  loading,
}) => {
  const t = TRANSLATIONS[lang];
  const langRef = useRef<Lang>(lang);
  langRef.current = lang;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SceneHandle | null>(null);
  const viewRef = useRef<F16FemViewState>(view);
  viewRef.current = view;
  const animPhaseRef = useRef<number>(Math.PI);
  const hasInitCam = useRef(false);

  const [meshStatus, setMeshStatus] = useState<string>('');
  const [fieldMax, setFieldMax] = useState<number>(300);

  const getFieldRange = useCallback((res: F16FemSimResponse, field: FemFieldId): [number, number] => {
    const k = res.metrics.kpis;
    if (field === 'von_mises_mpa') return [0, Math.max(20, k.max_von_mises_mpa)];
    if (field === 'strain_micro') return [0, Math.max(100, k.max_strain_micro)];
    return [0, Math.max(5, k.max_disp_mm)];
  }, []);

  const applyView = useCallback(
    (s: SceneHandle, res: F16FemSimResponse, v: F16FemViewState) => {
      if (!s.dataLoaded) return;
      const [lo, hi] = getFieldRange(res, v.field);
      setFieldMax(hi);

      applyStopsToLut(s.lut, FIELD_PALETTES[v.field].stops, lo, hi);

      s.femMapper.setColorByArrayName(v.field);
      s.femMapper.setScalarRange(lo, hi);
      s.femActor.getProperty().setEdgeVisibility(v.showMeshEdges);
      s.femActor.getProperty().setOpacity(v.showInternalSlice ? 0.34 : 1.0);

      s.ghostActor.setVisibility(v.showGhost);

      s.sliceMapper.setColorByArrayName(v.field);
      s.sliceMapper.setScalarRange(lo, hi);
      s.sliceActor.setVisibility(v.showInternalSlice);

      if (s.mesh && !v.playing) {
        applyDeformation(s.mesh, v.warpScale);
      }

      s.genericRW.getRenderWindow().render();
    },
    [getFieldRange]
  );

  const setCameraPreset = useCallback((preset: 'iso' | 'front' | 'top' | 'root') => {
    const s = sceneRef.current;
    if (!s) return;
    const ren = s.genericRW.getRenderer();
    const cam = ren.getActiveCamera();
    if (preset === 'front') {
      cam.setFocalPoint(1.5, 0.0, 0.4);
      cam.setPosition(-19.0, 0.0, 2.2);
      cam.setViewUp(0, 0, 1);
    } else if (preset === 'top') {
      cam.setFocalPoint(0.2, 0.0, 0.0);
      cam.setPosition(0.2, 0.0, 24.0);
      cam.setViewUp(0, 1, 0);
    } else if (preset === 'root') {
      cam.setFocalPoint(1.5, 1.4, 0.1);
      cam.setPosition(-5.0, 9.0, 5.2);
      cam.setViewUp(0, 0, 1);
    } else {
      cam.setFocalPoint(0.5, 0.0, 0.5);
      cam.setPosition(-14.5, -14.5, 7.8);
      cam.setViewUp(0, 0, 1);
    }
    ren.resetCameraClippingRange();
    s.genericRW.getRenderWindow().render();
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let disposed = false;

    const genericRW = vtkGenericRenderWindow.newInstance({
      background: [0.105, 0.082, 0.066],
    });
    genericRW.setContainer(el);
    genericRW.resize();

    const renderer = genericRW.getRenderer();
    const interactor = genericRW.getInteractor();

    const lut = vtkColorTransferFunction.newInstance();
    applyStopsToLut(lut, FIELD_PALETTES.von_mises_mpa.stops, 0, 350);

    // Actor 1: Deformable F-16 3D FEM Surface
    const femMapper = vtkMapper.newInstance();
    femMapper.setLookupTable(lut);
    femMapper.setColorMode(ColorMode.MAP_SCALARS);
    femMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    femMapper.setColorByArrayName('von_mises_mpa');
    femMapper.setScalarVisibility(true);
    const femActor = vtkActor.newInstance();
    femActor.setMapper(femMapper);
    femActor.setVisibility(false);
    femActor.getProperty().setAmbient(0.32);
    femActor.getProperty().setDiffuse(0.7);
    femActor.getProperty().setSpecular(0.32);
    femActor.getProperty().setSpecularPower(26);
    femActor.getProperty().setEdgeColor(0.98, 0.75, 0.35);
    renderer.addActor(femActor);

    // Actor 2: Undeformed Reference Ghost Wireframe
    const ghostMapper = vtkMapper.newInstance();
    ghostMapper.setScalarVisibility(false);
    const ghostActor = vtkActor.newInstance();
    ghostActor.setMapper(ghostMapper);
    ghostActor.setVisibility(false);
    ghostActor.getProperty().setRepresentation(1);
    ghostActor.getProperty().setColor(0.98, 0.78, 0.42);
    ghostActor.getProperty().setOpacity(0.24);
    ghostActor.getProperty().setLighting(false);
    renderer.addActor(ghostActor);

    // Actor 3: Internal 3D Tetrahedral Wing-Spar Stress Slice
    const sliceMapper = vtkMapper.newInstance();
    sliceMapper.setLookupTable(lut);
    sliceMapper.setColorMode(ColorMode.MAP_SCALARS);
    sliceMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    sliceMapper.setColorByArrayName('von_mises_mpa');
    sliceMapper.setScalarVisibility(true);
    const sliceActor = vtkActor.newInstance();
    sliceActor.setMapper(sliceMapper);
    sliceActor.setVisibility(false);
    sliceActor.getProperty().setEdgeVisibility(true);
    sliceActor.getProperty().setEdgeColor(0.15, 0.1, 0.08);
    sliceActor.getProperty().setAmbient(0.5);
    sliceActor.getProperty().setDiffuse(0.5);
    renderer.addActor(sliceActor);

    renderer.resetCamera();
    genericRW.getRenderWindow().render();

    let orientWidget: ReturnType<typeof vtkOrientationMarkerWidget.newInstance> | null = null;
    try {
      const cube = vtkAnnotatedCubeActor.newInstance();
      cube.setDefaultStyle({
        text: '+X',
        fontStyle: 'bold',
        fontFamily: 'Inter, system-ui, sans-serif',
        fontColor: '#fdf6ee',
        fontSizeScale: (res: number) => res / 2.2,
        faceColor: '#2a2019',
        edgeThickness: 0.12,
        edgeColor: '#f59e0b',
        resolution: 256,
      });
      cube.setXPlusFaceProperty({ text: 'TAIL' });
      cube.setXMinusFaceProperty({ text: 'NOSE' });
      cube.setYPlusFaceProperty({ text: 'L' });
      cube.setYMinusFaceProperty({ text: 'R' });
      cube.setZPlusFaceProperty({ text: 'TOP' });
      cube.setZMinusFaceProperty({ text: 'BOT' });

      orientWidget = vtkOrientationMarkerWidget.newInstance({
        actor: cube,
        interactor,
        parentRenderer: renderer,
        viewportCorner: vtkOrientationMarkerWidget.Corners.BOTTOM_RIGHT,
        viewportSize: 0.15,
        minPixelSize: 76,
        maxPixelSize: 118,
      });
      orientWidget.setEnabled(true);
    } catch {
      orientWidget = null;
    }

    sceneRef.current = {
      genericRW,
      femActor,
      femMapper,
      ghostActor,
      ghostMapper,
      sliceActor,
      sliceMapper,
      lut,
      orientWidget,
      mesh: null,
      dataLoaded: false,
    };

    const ro = new ResizeObserver(() => {
      if (!disposed) genericRW.resize();
    });
    ro.observe(el);

    // 60 FPS Elastic Wing-Bending Pulse Loop
    let rafId = 0;
    let lastTs = performance.now();
    const tick = (ts: number) => {
      if (disposed) return;
      const dt = Math.min((ts - lastTs) / 1000, 0.05);
      lastTs = ts;

      const s = sceneRef.current;
      const v = viewRef.current;
      if (s && s.dataLoaded && s.mesh && v.playing) {
        animPhaseRef.current += dt * 2.1;
        const factor = 0.5 - 0.5 * Math.cos(animPhaseRef.current);
        applyDeformation(s.mesh, v.warpScale * factor);
        s.genericRW.getRenderWindow().render();
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      if (orientWidget) {
        try {
          orientWidget.setEnabled(false);
          orientWidget.delete();
        } catch {
          // ignore
        }
      }
      try {
        genericRW.delete();
      } catch {
        // ignore
      }
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!result || !sceneRef.current) return;
    let cancelled = false;

    const loadAll = async () => {
      setMeshStatus(TRANSLATIONS[langRef.current].loadingVtp);
      try {
        const [femBuf, sliceBuf] = await Promise.all([
          getVtp(result.urls.fem),
          getVtp(result.urls.slice),
        ]);
        if (cancelled || !sceneRef.current) return;

        const s = sceneRef.current;
        const femPoly = parseVtp(femBuf);
        const ghostPoly = parseVtp(femBuf);
        const slicePoly = parseVtp(sliceBuf);

        const rawPts = femPoly.getPoints().getData() as Float32Array;
        const basePts = new Float32Array(rawPts);
        const dispArr = femPoly.getPointData().getArrayByName('disp_vec');
        const dispVec = dispArr
          ? new Float32Array(dispArr.getData() as Float32Array)
          : new Float32Array(basePts.length);

        const mesh: DeformableMesh = {
          poly: femPoly,
          basePts,
          dispVec,
          curPts: rawPts,
        };
        s.mesh = mesh;
        applyDeformation(mesh, viewRef.current.warpScale);

        s.femMapper.setInputData(femPoly);
        s.femActor.setVisibility(true);
        s.ghostMapper.setInputData(ghostPoly);
        s.sliceMapper.setInputData(slicePoly);
        s.dataLoaded = true;

        applyView(s, result, viewRef.current);
        if (!hasInitCam.current) {
          setCameraPreset('iso');
          hasInitCam.current = true;
        }
        setMeshStatus('');
      } catch (err) {
        if (!cancelled) setMeshStatus((err as Error).message);
      }
    };

    loadAll();
    return () => {
      cancelled = true;
    };
  }, [result, applyView, setCameraPreset]);

  useEffect(() => {
    if (sceneRef.current?.dataLoaded && result) {
      applyView(sceneRef.current, result, view);
    }
  }, [result, view, applyView]);

  const pal = FIELD_PALETTES[view.field];

  return (
    <div className="viewer-shell">
      <div ref={containerRef} className="vtk-canvas" />

      <div className="viewer-toolbar">
        <button type="button" className={view.playing ? 'active' : ''} onClick={onTogglePlay}>
          {view.playing ? <Pause size={14} /> : <Play size={14} />}
          {view.playing ? t.pauseBend : t.playBend}
        </button>
        <button type="button" onClick={() => setCameraPreset('iso')}>
          <Box size={14} /> {t.camIso}
        </button>
        <button type="button" onClick={() => setCameraPreset('front')}>
          <Layers size={14} /> {t.camFront}
        </button>
        <button type="button" onClick={() => setCameraPreset('top')}>
          <Compass size={14} /> {t.camTop}
        </button>
        <button type="button" onClick={() => setCameraPreset('root')}>
          <Crosshair size={14} /> {t.camRoot}
        </button>
      </div>

      {result && (
        <div className="colorbars-stack">
          <div className="colorbar-card">
            <div className="colorbar-header">
              <span>{t.fields[view.field]}</span>
              <span className="colorbar-unit">{t.colorbarUnit[view.field]}</span>
            </div>
            <div className="colorbar-strip" style={{ background: pal.css }} />
            <div className="colorbar-ticks">
              <span>0</span>
              <span>{(fieldMax * 0.5).toFixed(view.field === 'strain_micro' ? 0 : 1)}</span>
              <span>{fieldMax.toFixed(view.field === 'strain_micro' ? 0 : 1)}</span>
            </div>
          </div>
        </div>
      )}

      <div className="viewer-hint">{t.hint}</div>

      {(loading || meshStatus || !result) && (
        <div className="viewer-overlay">
          {loading ? (
            <div className="spinner-card">
              <div className="spinner" />
              <h3>{t.loadingCloud}</h3>
              <p>{t.loadingCloudDesc}</p>
            </div>
          ) : meshStatus ? (
            <div className="spinner-card">
              <div className="spinner" />
              <h3>{meshStatus}</h3>
            </div>
          ) : (
            <div className="spinner-card">
              <h3>{t.emptyTitle}</h3>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
