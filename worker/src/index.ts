export interface Env {
  F16_FEM_KV: KVNamespace;
  RATE_LIMIT_KV: KVNamespace;
  COMPUTE_ENDPOINT_URL?: string;
  COMPUTE_AUTH_TOKEN?: string;
  ALLOWED_ORIGIN: string;
  MAX_RUNS_PER_IP_PER_HOUR: string;
}

interface FemParams {
  g_load: number;
  speed_ms: number;
  material: 'al7075' | 'ti6al4v' | 'cfrp';
}

const VALID_FILES = new Set(['f16_fem.vtp', 'f16_slice.vtp', 'metrics.json']);

function corsHeaders(env: Env): Record<string, string> {
  return {
    'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function jsonResp(data: unknown, status: number, env: Env): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...corsHeaders(env),
    },
  });
}

function normalizeParams(raw: Partial<FemParams>): FemParams {
  let g = Math.round(Math.min(Math.max(Number(raw.g_load ?? 9), -3), 9));
  if (g === 0) g = 1;
  const spd = Math.round(Math.min(Math.max(Number(raw.speed_ms ?? 220), 100), 280) / 20) * 20;
  const m = String(raw.material ?? 'al7075').toLowerCase().trim();
  const material: FemParams['material'] =
    m === 'ti6al4v' || m === 'cfrp' ? m : 'al7075';
  return { g_load: g, speed_ms: spd, material };
}

async function computeCaseKey(p: FemParams): Promise<string> {
  const canonical = `f16fem2|g=${p.g_load.toFixed(0)}|v=${p.speed_ms.toFixed(0)}|mat=${p.material}`;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 16);
}

function buildUrls(caseKey: string) {
  return {
    fem: `/api/results/${caseKey}/f16_fem.vtp`,
    slice: `/api/results/${caseKey}/f16_slice.vtp`,
    metrics: `/api/results/${caseKey}/metrics.json`,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    if (path === '/api/health') {
      return jsonResp(
        {
          status: 'ok',
          aircraft: 'F-16',
          solver: '3d-tet-fem',
          compute_configured: Boolean(env.COMPUTE_ENDPOINT_URL),
        },
        200,
        env
      );
    }

    if (path === '/api/cases' && request.method === 'GET') {
      const idxRaw = await env.F16_FEM_KV.get('index:cases');
      const cases = idxRaw ? JSON.parse(idxRaw) : [];
      return jsonResp({ cases }, 200, env);
    }

    const resMatch = path.match(/^\/api\/results\/([a-f0-9]{16})\/([a-z0-9_.-]+)$/);
    if (resMatch && (request.method === 'GET' || request.method === 'HEAD')) {
      const [, caseKey, fname] = resMatch;
      if (!VALID_FILES.has(fname)) {
        return jsonResp({ error: 'Invalid result file' }, 400, env);
      }
      const val = await env.F16_FEM_KV.get(`case:${caseKey}:${fname}`);
      if (!val) {
        return jsonResp({ error: 'Result not found in KV cache' }, 404, env);
      }
      const contentType = fname.endsWith('.json')
        ? 'application/json; charset=utf-8'
        : 'application/xml; charset=utf-8';
      return new Response(request.method === 'HEAD' ? null : val, {
        status: 200,
        headers: {
          'Content-Type': contentType,
          'Cache-Control': 'public, max-age=31536000, immutable',
          ...corsHeaders(env),
        },
      });
    }

    if (path === '/api/simulate' && request.method === 'POST') {
      let body: Partial<FemParams> = {};
      try {
        body = (await request.json()) as Partial<FemParams>;
      } catch {
        return jsonResp({ error: 'Invalid JSON body' }, 400, env);
      }

      const params = normalizeParams(body);
      const caseKey = await computeCaseKey(params);

      // 1. Check Cloudflare KV Edge Cache first (0ms hit, $0 compute)
      const cachedMetricsRaw = await env.F16_FEM_KV.get(`case:${caseKey}:metrics.json`);
      if (cachedMetricsRaw) {
        return jsonResp(
          {
            status: 'ok',
            cached: true,
            backend: 'cloudflare-kv-cache',
            case_key: caseKey,
            metrics: JSON.parse(cachedMetricsRaw),
            urls: buildUrls(caseKey),
          },
          200,
          env
        );
      }

      if (!env.COMPUTE_ENDPOINT_URL) {
        return jsonResp({ error: 'COMPUTE_ENDPOINT_URL not configured' }, 503, env);
      }

      // 2. Per-IP Hourly Rate Limiting
      const ip = request.headers.get('CF-Connecting-IP') || 'anon';
      const hourBucket = Math.floor(Date.now() / 3600000);
      const rlKey = `rl:f16fem:${ip}:${hourBucket}`;
      const limit = Math.max(1, parseInt(env.MAX_RUNS_PER_IP_PER_HOUR || '15', 10));
      const used = parseInt((await env.RATE_LIMIT_KV.get(rlKey)) || '0', 10);
      if (used >= limit) {
        return jsonResp(
          { error: `Hourly cloud FEM quota reached (${limit}/hr). Cached cases remain instant.` },
          429,
          env
        );
      }

      // 3. Call Modal Cloud 3D FEM Solver
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (env.COMPUTE_AUTH_TOKEN) {
        headers['Authorization'] = `Bearer ${env.COMPUTE_AUTH_TOKEN}`;
      }

      const modalRes = await fetch(env.COMPUTE_ENDPOINT_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify(params),
      });
      if (!modalRes.ok) {
        const errText = await modalRes.text();
        return jsonResp(
          { error: `Cloud FEM failed (${modalRes.status}): ${errText.slice(0, 600)}` },
          502,
          env
        );
      }

      const payload = (await modalRes.json()) as {
        metrics: Record<string, unknown>;
        files_content: Record<string, string>;
      };

      // 4. Persist results to Cloudflare KV & update case index
      const writes: Promise<void>[] = [
        env.F16_FEM_KV.put(`case:${caseKey}:metrics.json`, JSON.stringify(payload.metrics)),
      ];
      for (const [fname, content] of Object.entries(payload.files_content || {})) {
        if (VALID_FILES.has(fname)) {
          writes.push(env.F16_FEM_KV.put(`case:${caseKey}:${fname}`, content));
        }
      }
      await Promise.all(writes);

      const idxRaw = await env.F16_FEM_KV.get('index:cases');
      const idxList: Array<{ case_key: string }> = idxRaw ? JSON.parse(idxRaw) : [];
      const filtered = idxList.filter((item) => item.case_key !== caseKey);
      filtered.unshift(payload.metrics as { case_key: string });
      await env.F16_FEM_KV.put('index:cases', JSON.stringify(filtered.slice(0, 30)));
      await env.RATE_LIMIT_KV.put(rlKey, String(used + 1), { expirationTtl: 3900 });

      return jsonResp(
        {
          status: 'ok',
          cached: false,
          backend: 'modal-gmsh-3d-fem',
          case_key: caseKey,
          remaining_quota: Math.max(0, limit - used - 1),
          metrics: payload.metrics,
          urls: buildUrls(caseKey),
        },
        200,
        env
      );
    }

    return jsonResp({ error: 'Not found' }, 404, env);
  },
};
