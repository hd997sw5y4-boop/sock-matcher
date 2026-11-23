export function createCamera(video) {
  let stream = null;
  let facing = 'environment';

  async function start(preferred = facing) {
    stop();
    facing = preferred;
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: preferred },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('muted', '');
    await video.play();
    await ready();
    facing = detectFacing() || preferred;
    return stream;
  }

  function detectFacing() {
    const track = stream?.getVideoTracks?.()[0];
    const mode = track?.getSettings?.().facingMode;
    return mode || null;
  }

  function ready() {
    if (video.videoWidth) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => resolve();
      video.addEventListener('loadedmetadata', done, { once: true });
      video.addEventListener('playing', done, { once: true });
    });
  }

  function stop() {
    if (stream) {
      for (const t of stream.getTracks()) t.stop();
      stream = null;
    }
    video.srcObject = null;
  }

  function torchSupported() {
    const track = stream?.getVideoTracks?.()[0];
    const caps = track?.getCapabilities?.();
    return !!(caps && 'torch' in caps);
  }

  async function setTorch(on) {
    const track = stream?.getVideoTracks?.()[0];
    if (!track || !torchSupported()) return false;
    try {
      await track.applyConstraints({ advanced: [{ torch: !!on }] });
      return true;
    } catch {
      return false;
    }
  }

  async function flip() {
    const next = facing === 'environment' ? 'user' : 'environment';
    await start(next);
  }

  function isFront() {
    return facing === 'user';
  }

  function getStream() { return stream; }

  return { start, stop, flip, setTorch, torchSupported, ready, isFront, getStream };
}
