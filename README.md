# Sock Matcher

A fast, mobile-first web app that finds matching socks from your phone camera. It runs entirely on the device with classic computer vision — no account, no server, no AI.

## On your iPhone

1. Open the live site in **Safari** (HTTPS is required for the camera).
2. Tap **Add to Home Screen** if you want it to feel like a regular app.
3. Tap **Enable Camera** and point at socks spread on a bed, floor, or table.
4. Or tap **Try Demo Pile** first to see it work without laundry.

### Modes

- **Scan a Sock** (default) — put one sock in the center box (or hold it up) and tap **Scan**, or just tap a sock on screen. A thumbnail of it sits in the corner, and the live camera outlines only its best 1–3 matches with a match %.
- **Auto Pairs** — finds the pairs it is most confident about and shows at most 3 at a time. Unmatched socks get no overlay. **Freeze** holds the outlines still while you grab socks.
- **Photo** / **Demo** — run the same matching on a still picture.
- **Settings** — detection sensitivity, match strictness, and **Set Surface** for a tricky bedspread.

## How it works (no AI)

- The camera preview is the native `<video>`; vision runs in a Web Worker on a ~90k-pixel copy of the visible area, about 15 times a second. Outlines are drawn every frame on their own canvas and follow each sock using its tracked velocity, so the preview never waits on the matching.
- Surface model: dominant chroma plus per-cell lightness of the bed or floor, so wrinkles and soft shadows aren't socks. Touching socks are split by color-aware region growing.
- Descriptor per sock (cached per tracked sock): gray-world white balance, soft a*b* and lightness histograms, 3 dominant colors, 5 color bands along the sock's long axis (cuff → toe, either direction), gradient orientation relative to the axis, edge density, and stripe frequency.
- Matches must persist over several frames (hysteresis) before they light up, which keeps outlines steady.

## Develop

```bash
npm start          # http://127.0.0.1:4173
npx playwright install chromium
npm test
```
