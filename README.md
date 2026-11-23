# Sock Matcher

A fast, mobile-first web app that finds matching socks from your phone camera. It runs entirely on the device with classic computer vision — no account, no server, no AI.

## On your iPhone

1. Open the live site in **Safari** (HTTPS is required for the camera).
2. Tap **Add to Home Screen** if you want it to feel like a regular app.
3. Tap **Enable Camera** and point at socks spread on a bed, floor, or table.
4. Or tap **Try Demo Pile** first to see it work without laundry.

### Modes

- **Auto Pairs** — every sock is outlined. Matching pairs share a color and a number (`1`, `2`, `3`…). Leftovers get a small “solo” badge.
- **Tap to Match** — tap one sock. Everything else dims, and its best match pulses with a match percentage.
- **Freeze** — on live camera, freeze the frame so the boxes stay put while you pick socks up.
- **Photo** — snap a picture or pick one from your Camera Roll if you’d rather not use the live camera.
- **Set Surface** (in Settings) — tap the sheet or floor if the background is busy. Drag **Detection sensitivity** and **Match strictness** if the pile is tricky.

## Honest limits

- Plain identical black or white socks look almost the same to a camera, so the app can’t reliably tell one pair from another.
- Heavily overlapping piles may be read as one blob. Spread socks out with a little space between them.
- A very patterned bedspread can confuse detection — use **Set Surface** or a plainer background.

## Develop

```bash
npm start          # http://127.0.0.1:4173
npx playwright install chromium
npm test
```
