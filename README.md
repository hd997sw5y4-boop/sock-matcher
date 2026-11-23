# Sock Matcher

A fast, mobile-first web app that finds matching socks from your phone camera. It runs entirely on the device — no account, no server. A small visual-pattern model (DINOv2-small) and a tap-to-cut model (SlimSAM-77) download once, are cached on the phone, and never send your photos anywhere.

## On your iPhone

1. Open the live site in **Safari** (HTTPS is required for the camera).
2. Tap **Add to Home Screen** if you want it to feel like a regular app.
3. Tap **Enable Camera** and point at socks on a bed, floor, or table. The first visit shows a short “Loading pattern model…” bar (about 40 MB, then it stays cached).
4. Or tap **Try Demo Pile** first to see the flow without laundry.

### Modes

- **Scan a Sock** (default) — tap a sock in the live view, or put one in the center box and tap **Scan**. A thumbnail sits in the corner, and the camera outlines only its best 1–3 matches, with a confidence %. If the model isn’t sure, nothing glows.
- **Auto Pairs** — finds the pairs it is most confident about and shows at most 3 at a time. Unmatched socks get no overlay. **Freeze** holds the outlines still while you grab socks.
- **Photo** / **Demo** — run the same matching on a still picture.
- **Settings** — detection sensitivity, match strictness, and **Set Surface** for a tricky bedspread.

## How it works

- The camera preview is the native `<video>`; tracking still runs in a Web Worker on a ~90k-pixel copy of the visible area, about 15 times a second. Outlines are drawn every frame and follow each sock’s velocity, so the preview never waits on matching.
- **Tap-to-pick:** SlimSAM (with a flood-fill fallback) cuts out the sock you tapped, including cuff, heel, and print, instead of a random color scrap.
- **Matching:** a DINOv2-small embedding of that crop is compared with candidate socks in the pile (cosine similarity), then combined with the existing color / stripe-band descriptor (heel and toe color, stripe count, cuff). Only strong top 1–3 matches are shown.
- Pattern scoring runs in its own worker a few times a second, and embeddings are cached per tracked sock so the camera stays smooth.
- Highlights use only light hysteresis, so a brief miss doesn’t flicker but a wrong glow does not stick.

## Models and licenses

- [DINOv2-small](https://github.com/facebookresearch/dinov2) (Meta), Apache-2.0, via the `Xenova/dinov2-small` ONNX build (~24 MB quantized).
- [SlimSAM-77-uniform](https://github.com/czg1225/SlimSAM), Apache-2.0, via the `Xenova/slimsam-77-uniform` ONNX build (~14 MB quantized).
- Loaded in the browser with [Transformers.js](https://huggingface.co/docs/transformers.js). WebGPU is used when the phone supports it, otherwise WebAssembly.

## Develop

```bash
npm start          # http://127.0.0.1:4173
npx playwright install chromium
npm test
npm run test:rank
npm run test:lock
```
