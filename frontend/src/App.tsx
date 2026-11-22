import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowUpDown,
  Database,
  Eye,
  Flame,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  ShieldCheck,
  Timer,
  Wrench,
  Zap,
} from 'lucide-react';
import { fetchCachedCases, runF16FemSimulation } from './api';
import { F16FemViewer } from './components/F16FemViewer';
import { detectInitialLang, TRANSLATIONS, type Lang } from './i18n';
import type {
  F16FemMetrics,
  F16FemParams,
  F16FemSimResponse,
  F16FemViewState,
  FemFieldId,
  MaterialId,
} from './types';

const DEFAULT_PARAMS: F16FemParams = {
  g_load: 9,
  speed_ms: 220,
  material: 'al7075',
};

export const App: React.FC = () => {
  const [lang, setLang] = useState<Lang>(detectInitialLang);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [params, setParams] = useState<F16FemParams>(DEFAULT_PARAMS);
  const [view, setView] = useState<F16FemViewState>({
    field: 'von_mises_mpa',
    warpScale: 8.0,
    playing: true,
    showGhost: true,
    showMeshEdges: false,
    showInternalSlice: false,
  });

  const [result, setResult] = useState<F16FemSimResponse | null>(null);
  const [cachedList, setCachedList] = useState<F16FemMetrics[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const t = TRANSLATIONS[lang];

  useEffect(() => {
    document.title = t.pageTitle;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    try {
      localStorage.setItem('f16fem_lang', lang);
    } catch {
      // ignore
    }
  }, [lang, t.pageTitle]);

  useEffect(() => {
    (async () => {
      try {
        const list = await fetchCachedCases();
        setCachedList(list);
        if (list.length > 0) {
          const first = list.find((c) => c.params.g_load >= 8) || list[0];
          setParams(first.params);
          setResult({
            status: 'ok',
            cached: true,
            backend: 'cloudflare-kv-cache',
            case_key: first.case_key,
            metrics: first,
            urls: {
              fem: `/api/results/${first.case_key}/f16_fem.vtp`,
              slice: `/api/results/${first.case_key}/f16_slice.vtp`,
              metrics: `/api/results/${first.case_key}/metrics.json`,
            },
          });
        }
      } catch {
        // ignore on initial load
      }
    })();
  }, []);

  const handleRun = async (override?: F16FemParams) => {
    const target = override ?? params;
    setLoading(true);
    setError('');
    try {
      const resp = await runF16FemSimulation(target);
      setResult(resp);
      setParams(resp.metrics.params);
      const list = await fetchCachedCases();
      setCachedList(list);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const kpis = result?.metrics.kpis;
  const timing = result?.metrics.timing_sec;
  const sf = kpis?.safety_factor ?? 0;
  const sfClass = sf >= 1.5 ? 'sage' : sf >= 1.05 ? 'gold' : 'coral';
  const sfTag = sf >= 1.5 ? t.safeOk : sf >= 1.05 ? t.safeWarn : t.safeDanger;

  return (
    <div className={`app-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-top">
            <div className="brand-badge">{t.badge}</div>
            <div className="lang-switch" role="group" aria-label="Language">
              <button type="button" className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>
                中文
              </button>
              <button type="button" className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>
                EN
              </button>
            </div>
          </div>
          <h1>{t.title}</h1>
          <p>{t.subtitle}</p>
        </div>

        <section className="panel-section">
          <div className="section-title">
            <Wrench size={14} /> {t.sectionLoad}
          </div>

          <div className="preset-list">
            {[
              { p: { g_load: 9, speed_ms: 220, material: 'al7075' as MaterialId }, label: t.presets.pull9g },
              { p: { g_load: 5, speed_ms: 260, material: 'ti6al4v' as MaterialId }, label: t.presets.turn5g },
              { p: { g_load: -3, speed_ms: 200, material: 'cfrp' as MaterialId }, label: t.presets.neg3g },
            ].map((item) => {
              const active =
                params.g_load === item.p.g_load &&
                params.speed_ms === item.p.speed_ms &&
                params.material === item.p.material;
              return (
                <button
                  key={item.label}
                  type="button"
                  className={`preset-btn ${active ? 'active' : ''}`}
                  onClick={() => setParams(item.p)}
                >
                  {item.label}
                </button>
              );
            })}
          </div>

          <label className="slider-row">
            <div className="slider-header">
              <span>{t.gLoad}</span>
              <strong>{params.g_load > 0 ? `+${params.g_load}G` : `${params.g_load}G`}</strong>
            </div>
            <input
              type="range"
              min={-3}
              max={9}
              step={1}
              value={params.g_load}
              onChange={(e) => {
                const v = Number(e.target.value);
                setParams({ ...params, g_load: v === 0 ? 1 : v });
              }}
            />
          </label>

          <label className="slider-row">
            <div className="slider-header">
              <span>{t.speed}</span>
              <strong>
                {params.speed_ms} m/s ({Math.round(params.speed_ms * 3.6)} km/h)
              </strong>
            </div>
            <input
              type="range"
              min={100}
              max={280}
              step={20}
              value={params.speed_ms}
              onChange={(e) => setParams({ ...params, speed_ms: Number(e.target.value) })}
            />
          </label>

          <label className="select-row">
            <span>{t.material}</span>
            <select
              value={params.material}
              onChange={(e) => setParams({ ...params, material: e.target.value as MaterialId })}
            >
              <option value="al7075">{t.matOptions.al7075}</option>
              <option value="ti6al4v">{t.matOptions.ti6al4v}</option>
              <option value="cfrp">{t.matOptions.cfrp}</option>
            </select>
          </label>

          <button type="button" className="run-btn" disabled={loading} onClick={() => handleRun()}>
            <Play size={15} />
            {loading ? t.runningBtn : t.runBtn}
          </button>
        </section>

        <section className="panel-section">
          <div className="section-title">
            <Eye size={14} /> {t.sectionVisual}
          </div>

          <label className="select-row">
            <span>{t.fieldLabel}</span>
            <select
              value={view.field}
              onChange={(e) => setView({ ...view, field: e.target.value as FemFieldId })}
            >
              <option value="von_mises_mpa">{t.fields.von_mises_mpa}</option>
              <option value="strain_micro">{t.fields.strain_micro}</option>
              <option value="disp_mm">{t.fields.disp_mm}</option>
            </select>
          </label>

          <label className="slider-row">
            <div className="slider-header">
              <span>{t.warpScale}</span>
              <strong>{view.warpScale.toFixed(0)}×</strong>
            </div>
            <input
              type="range"
              min={0}
              max={25}
              step={1}
              value={view.warpScale}
              onChange={(e) => setView({ ...view, warpScale: Number(e.target.value) })}
            />
          </label>

          <div className="toggle-list">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={view.showGhost}
                onChange={(e) => setView({ ...view, showGhost: e.target.checked })}
              />
              <span>{t.showGhost}</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={view.showMeshEdges}
                onChange={(e) => setView({ ...view, showMeshEdges: e.target.checked })}
              />
              <span>{t.showMeshEdges}</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={view.showInternalSlice}
                onChange={(e) => setView({ ...view, showInternalSlice: e.target.checked })}
              />
              <span>{t.showInternalSlice}</span>
            </label>
          </div>
        </section>

        <div className="sidebar-footer">
          <Flame size={13} />
          <span>{t.footerNote}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="kpi-bar">
          <button
            type="button"
            className="collapse-btn"
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? t.hideSidebar : t.showSidebar}
          >
            {sidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </button>

          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">
                <Zap size={14} /> {t.kpiStress}
              </div>
              <div className="kpi-value gold">
                {kpis ? `${kpis.max_von_mises_mpa.toFixed(1)} MPa` : '--'}
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">
                <ShieldCheck size={14} /> {t.kpiSafety}
              </div>
              <div className={`kpi-value ${kpis ? sfClass : ''}`}>
                {kpis ? `${kpis.safety_factor.toFixed(2)} (${sfTag})` : '--'}
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">
                <Activity size={14} /> {t.kpiStrain}
              </div>
              <div className="kpi-value">
                {kpis ? `${kpis.max_strain_micro.toFixed(0)} με (${(kpis.max_strain_micro / 10000).toFixed(2)}%)` : '--'}
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">
                <ArrowUpDown size={14} /> {t.kpiTip}
              </div>
              <div className="kpi-value coral">
                {kpis ? `${kpis.wingtip_dz_mm > 0 ? '+' : ''}${kpis.wingtip_dz_mm.toFixed(1)} mm` : '--'}
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">
                {result?.cached ? <Database size={14} /> : <Timer size={14} />}
                {t.kpiSource}
              </div>
              <div className={`kpi-value ${result?.cached ? 'sage' : 'gold'}`}>
                {kpis && timing
                  ? result?.cached
                    ? t.hitSaved(kpis.n_tets, timing.total.toFixed(1))
                    : t.freshTime(kpis.n_tets, timing.total.toFixed(1))
                  : '--'}
              </div>
            </div>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <AlertCircle size={15} />
            <span>{error}</span>
          </div>
        )}

        <div className="viewer-wrapper">
          <F16FemViewer
            lang={lang}
            result={result}
            view={view}
            onTogglePlay={() => setView((prev) => ({ ...prev, playing: !prev.playing }))}
            loading={loading}
          />
        </div>

        {cachedList.length > 0 && (
          <footer className="presets-strip">
            <span className="strip-label">
              <Database size={13} /> {t.cachedStrip}
            </span>
            <div className="strip-items">
              {cachedList.map((c) => {
                const active = c.case_key === result?.case_key;
                const gStr = c.params.g_load > 0 ? `+${c.params.g_load}G` : `${c.params.g_load}G`;
                return (
                  <button
                    key={c.case_key}
                    type="button"
                    className={`case-chip ${active ? 'active' : ''}`}
                    onClick={() => handleRun(c.params)}
                  >
                    {gStr} · {c.params.material.toUpperCase()} · σ={c.kpis.max_von_mises_mpa.toFixed(0)}MPa · Δz={c.kpis.wingtip_dz_mm.toFixed(0)}mm
                  </button>
                );
              })}
            </div>
          </footer>
        )}
      </main>
    </div>
  );
};
