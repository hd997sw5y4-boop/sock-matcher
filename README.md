# Sock Matcher

A fast, mobile-first web app that finds matching socks from your phone camera. It runs entirely on the device — no account, no server. Classic color matching works immediately. A small visual-pattern model (quantized DINOv2-small, about 15 MB) can load in the background after you pick a sock, is cached on the phone, and never sends your photos anywhere.

## On your iPhone

1. Open the live site in **Safari** (HTTPS is required for the camera).
2. Tap **Add to Home Screen** if you want it to feel like a regular app.
3. Tap **Enable Camera** and point at socks on a bed, floor, or table. You can scan right away. If smart matching is downloading, the status reads **Loading smart matching… x%**. If it can’t load, it reads **Smart matching unavailable** and color matching still works.
4. Or tap **Try Demo Pile** first to see the flow without laundry.

**Battery saver** is on by default (you can turn it off in Settings) so the phone stays cooler.

### Modes

- **Scan a Sock** (default) — tap a sock in the live view, or put one in the center box and tap **Scan**. The app remembers that sock’s appearance (color, stripes/texture, and the pattern embedding if it’s loaded) until you tap ✕ or scan again, including if you switch away from the tab. The camera then outlines the best 1–3 appearance matches — any size or shape, including bunched or partial socks. Size and outline are not part of the score.
- **Auto Pairs** — finds the pairs it is most confident about and shows at most 3 at a time. Unmatched socks get no overlay. **Freeze** holds the outlines still while you grab socks.
- **Color Sort** — pick a color chip (pink, white, grey, black, blue, green, purple, red, yellow, multi/striped, or tap a sock to pick a color). The camera outlines every sock of that color, with a count, so you can pull a huge pile into smaller ones first. This uses a lightweight color check on the same low-resolution, battery-saver camera path — no 1–3 cap, and the pattern model is not used.
- **Photo** / **Demo** — run the same matching on a still picture.
- **Settings** — detection sensitivity, match strictness, **Battery saver**, and **Set Surface** for a tricky bedspread.

## How it works

- The camera stream is requested at 30 fps and a modest resolution. Live tracking runs in a Web Worker on a ~320-pixel-wide copy of the visible area, about 8 times a second. Nearly unchanged frames are skipped, and processing pauses when the tab is hidden.
- Scanning and tapping always use the classic color path immediately. The pattern model is enhancement-only: it embeds the scanned sock and a few candidate crops on demand (about 1–2 times a second), never the full frame continuously.
- **Matching:** a DINOv2-small embedding of that crop is compared with candidate socks in the pile (cosine similarity), then combined with the existing color / stripe-band descriptor (heel and toe color, stripe count, cuff). Only strong top 1–3 matches are shown.
- Highlights use only light hysteresis, so a brief miss doesn’t flicker but a wrong glow does not stick.
- WebGPU is not used, so the app never runs WebGPU and WebAssembly at the same time.

## Models and licenses

- [DINOv2-small](https://github.com/facebookresearch/dinov2) (Meta), Apache-2.0, via the `Xenova/dinov2-small` ONNX build (~15 MB, 4-bit quantized), self-hosted at `/models`.
- Loaded in the browser with [Transformers.js](https://huggingface.co/docs/transformers.js) over same-origin files in `/vendor`, using WebAssembly only.
- Before a production deploy or a local model test, run `npm run fetch-models` so those files are on disk (they are not committed to git).

## Develop

```bash
npm start          # http://127.0.0.1:4173
npm run fetch-models
npx playwright install chromium webkit
npm test
npm run test:ios
npm run test:colors
npm run test:rank
npm run test:lock
```
