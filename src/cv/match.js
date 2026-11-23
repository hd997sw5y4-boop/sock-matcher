import { histogramIntersection, labDist } from './color.js';
import { paletteDistance } from './describe.js';

export function descriptorSimilarity(a, b) {
  const h = histogramIntersection(a.histH, b.histH);
  const s = histogramIntersection(a.histS, b.histS);
  const l = histogramIntersection(a.histL, b.histL);
  const ab = histogramIntersection(a.histAB, b.histAB);
  const meanD = labDist(a.mean[0], a.mean[1], a.mean[2], b.mean[0], b.mean[1], b.mean[2]);
  const meanScore = Math.max(0, 1 - meanD / 55);
  const pal = Math.max(0, 1 - paletteDistance(a.palette, b.palette) / 45);
  const edge = 1 - Math.min(1, Math.abs(a.edgeDensity - b.edgeDensity) / 0.28);
  const gvar = 1 - Math.min(1, Math.abs(Math.sqrt(a.gradVar) - Math.sqrt(b.gradVar)) / 18);
  const orient = histogramIntersection(a.orient, b.orient);
  const aspect = 1 - Math.min(1, Math.abs(a.aspect - b.aspect) / 2.4);
  const area = 1 - Math.min(1, Math.abs(a.area - b.area) / Math.max(a.area, b.area, 1));

  // Color dominates; texture catches stripes vs solids vs dots; shape is mild.
  const color = 0.22 * h + 0.10 * s + 0.10 * l + 0.22 * ab + 0.10 * meanScore + 0.12 * pal;
  const texture = 0.06 * edge + 0.04 * gvar + 0.04 * orient;
  const shape = 0.03 * aspect + 0.02 * area;
  return Math.max(0, Math.min(1, (color + texture + shape) / 1.05));
}

export function similarityMatrix(descs) {
  const n = descs.length;
  const m = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    m[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const s = descriptorSimilarity(descs[i], descs[j]);
      m[i][j] = s;
      m[j][i] = s;
    }
  }
  return m;
}

/**
 * Pair socks. `strictness` is 0..1, mapped to a similarity threshold.
 * Returns { pairs: [{a,b,score}], unmatched: number[] }
 */
export function pairSocks(descs, strictness = 0.55) {
  const n = descs.length;
  if (n < 2) return { pairs: [], unmatched: n === 1 ? [0] : [] };
  const sim = similarityMatrix(descs);
  const threshold = 0.42 + strictness * 0.36; // 0.42 .. 0.78
  const pairs = hungarianPairs(sim, threshold);
  const used = new Set();
  for (const p of pairs) { used.add(p.a); used.add(p.b); }
  const unmatched = [];
  for (let i = 0; i < n; i++) if (!used.has(i)) unmatched.push(i);
  return { pairs, unmatched, sim, threshold };
}

/** Min-cost assignment with an optional dummy for an odd leftover. */
function hungarianPairs(sim, threshold) {
  const n = sim.length;
  const useDummy = n % 2 === 1;
  const dim = n + (useDummy ? 1 : 0);
  const INF = 1e6;
  const cost = Array.from({ length: dim }, () => new Float64Array(dim));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      if (i === j) { cost[i][j] = INF; continue; }
      if (useDummy && (i === n || j === n)) {
        cost[i][j] = 1 - (threshold - 0.02);
        continue;
      }
      const s = sim[i][j];
      cost[i][j] = s >= threshold ? 1 - s : INF;
    }
  }
  const match = hungarian(cost);
  const pairs = [];
  const seen = new Set();
  for (let i = 0; i < n; i++) {
    const j = match[i];
    if (j < 0 || j >= n || seen.has(i) || seen.has(j)) continue;
    const s = sim[i][j];
    if (s < threshold) continue;
    pairs.push({ a: Math.min(i, j), b: Math.max(i, j), score: s });
    seen.add(i); seen.add(j);
  }
  pairs.sort((p, q) => q.score - p.score);
  return pairs;
}

/**
 * Hungarian / Kuhn-Munkres for square cost matrices. Returns col-for-row assignment.
 * Compact O(n^3) implementation.
 */
export function hungarian(cost) {
  const n = cost.length;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  const minv = new Float64Array(n + 1);
  const used = new Uint8Array(n + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    minv.fill(1e15);
    used.fill(0);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = 1e15;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const assignment = new Int32Array(n);
  assignment.fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j]) assignment[p[j] - 1] = j - 1;
  }
  return assignment;
}
