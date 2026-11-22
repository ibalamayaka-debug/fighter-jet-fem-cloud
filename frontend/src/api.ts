import type { F16FemMetrics, F16FemParams, F16FemSimResponse } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

export async function fetchCachedCases(): Promise<F16FemMetrics[]> {
  const res = await fetch(resolveApiUrl('/api/cases'));
  if (!res.ok) throw new Error(`Failed to fetch cached cases (${res.status})`);
  const data = (await res.json()) as { cases?: F16FemMetrics[] };
  return data.cases ?? [];
}

export async function runF16FemSimulation(params: F16FemParams): Promise<F16FemSimResponse> {
  const res = await fetch(resolveApiUrl('/api/simulate'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = (await res.json()) as F16FemSimResponse & { error?: string };
  if (!res.ok) {
    throw new Error(data.error || `Simulation failed (${res.status})`);
  }
  return data;
}

export async function fetchVtpBuffer(urlPath: string): Promise<ArrayBuffer> {
  const res = await fetch(resolveApiUrl(urlPath));
  if (!res.ok) throw new Error(`Failed to fetch VTP (${res.status})`);
  return res.arrayBuffer();
}
