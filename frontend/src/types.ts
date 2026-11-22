export type MaterialId = 'al7075' | 'ti6al4v' | 'cfrp';
export type FemFieldId = 'von_mises_mpa' | 'strain_micro' | 'disp_mm';

export interface F16FemParams {
  g_load: number;
  speed_ms: number;
  material: MaterialId;
}

export interface F16FemMetrics {
  case_key: string;
  aircraft: string;
  fsi_source: string;
  params: F16FemParams;
  kpis: {
    max_von_mises_mpa: number;
    peak_singularity_mpa: number;
    yield_strength_mpa: number;
    safety_factor: number;
    max_strain_micro: number;
    max_disp_mm: number;
    wingtip_dz_mm: number;
    total_lift_kn: number;
    total_lift_tonnes: number;
    n_nodes: number;
    n_tets: number;
    n_dofs: number;
    n_fixed_nodes: number;
    material_en: string;
    material_zh: string;
    E_gpa: number;
    nu: number;
  };
  timing_sec: {
    mesh: number;
    solve: number;
    postprocess: number;
    total: number;
  };
  file_sizes_kb: Record<string, number>;
  computed_at: string;
}

export interface F16FemSimResponse {
  status: string;
  cached: boolean;
  backend: string;
  case_key: string;
  remaining_quota?: number;
  metrics: F16FemMetrics;
  urls: {
    fem: string;
    slice: string;
    metrics: string;
  };
}

export interface F16FemViewState {
  field: FemFieldId;
  warpScale: number;
  playing: boolean;
  showGhost: boolean;
  showMeshEdges: boolean;
  showInternalSlice: boolean;
}
