/**
 * Copy quantized models and a same-origin Transformers.js bundle into the site
 * so the app can load them on iPhone Safari. Safe to re-run.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cache = path.join(root, '.cache/hf');
const modelsDir = path.join(root, 'models/Xenova/dinov2-small');
const vendorTf = path.join(root, 'vendor');
const vendorOrt = path.join(root, 'vendor/onnx');

function ensureDir(d) { fs.mkdirSync(d, { recursive: true }); }

function copy(src, dest) {
  ensureDir(path.dirname(dest));
  if (fs.existsSync(dest) && fs.statSync(dest).size === fs.statSync(src).size) return;
  fs.copyFileSync(src, dest);
  console.log('copied', path.relative(root, dest), '(' + (fs.statSync(dest).size / 1e6).toFixed(1) + ' MB)');
}

async function download(url, dest) {
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1000) {
    console.log('have', path.relative(root, dest));
    return;
  }
  console.log('download', url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status + ' ' + url);
  ensureDir(path.dirname(dest));
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(dest, buf);
  console.log('wrote', path.relative(root, dest), '(' + (buf.length / 1e6).toFixed(1) + ' MB)');
}

const HF = 'https://huggingface.co/Xenova/dinov2-small/resolve/main';
ensureDir(path.join(modelsDir, 'onnx'));
await download(HF + '/config.json', path.join(modelsDir, 'config.json'));
await download(HF + '/preprocessor_config.json', path.join(modelsDir, 'preprocessor_config.json'));
const q4Cache = path.join(cache, 'Xenova/dinov2-small/onnx/model_q4.onnx');
if (fs.existsSync(q4Cache)) copy(q4Cache, path.join(modelsDir, 'onnx/model_q4.onnx'));
else await download(HF + '/onnx/model_q4.onnx', path.join(modelsDir, 'onnx/model_q4.onnx'));

const tfSrc = path.join(root, 'node_modules/@huggingface/transformers/dist/transformers.web.js');
const ortDist = path.join(root, 'node_modules/onnxruntime-web/dist');
const ortWasmJs = path.join(ortDist, 'ort.wasm.min.mjs');
if (!fs.existsSync(tfSrc) || !fs.existsSync(ortWasmJs)) {
  throw new Error('Run npm install first so transformers and onnxruntime-web are available.');
}

// Bundle Transformers.js with the WASM-only ONNX Runtime so the worker can
// load same-origin (bare "onnxruntime-web/webgpu" imports fail in Safari).
await build({
  entryPoints: [tfSrc],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  outfile: path.join(vendorTf, 'transformers.web.js'),
  alias: {
    'onnxruntime-web/webgpu': ortWasmJs,
    'onnxruntime-web': ortWasmJs,
  },
  external: ['sharp'],
  legalComments: 'none',
  logLevel: 'info',
});

// WASM only (no WebGPU/jsep). Keep both the classic and asyncify builds so
// older iPhone Safari and newer Chromium can each pick one file, not both.
const ortFiles = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
];
for (const f of ortFiles) {
  const src = path.join(ortDist, f);
  if (fs.existsSync(src)) copy(src, path.join(vendorOrt, f));
  else console.warn('missing', f);
}
console.log('models and vendor assets are ready');
