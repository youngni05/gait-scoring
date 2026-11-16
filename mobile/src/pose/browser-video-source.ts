import type { DecodedFrame, VideoInfo, VideoSession, VideoSource } from './ports';
import { FrameAudit } from './frame-audit';
const cancelled = () => new DOMException('작업을 취소했습니다.', 'AbortError');

function waitEvent(video: HTMLVideoElement, name: string, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (error?: unknown) => { clearTimeout(timer); video.removeEventListener(name, done); video.removeEventListener('error', failed); signal.removeEventListener('abort', abort); error ? reject(error) : resolve(); };
    const done = () => finish();
    const failed = () => finish(new Error('영상을 읽을 수 없습니다. 손상되었거나 지원하지 않는 코덱일 수 있습니다.'));
    const abort = () => finish(cancelled());
    const timer = setTimeout(() => finish(new Error('영상 응답 시간 초과 (15초). 다시 선택해 주세요.')), 15000);
    video.addEventListener(name, done, { once: true }); video.addEventListener('error', failed, { once: true }); signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}

export class BrowserVideoSession implements VideoSession {
  audit = new FrameAudit();
  private disposed = false;
  private running = false;
  private lifetime = new AbortController();
  private canvas = document.createElement('canvas');
  private qualityStart: { dropped: number; total: number } | null = null;
  constructor(readonly element: HTMLVideoElement, readonly info: VideoInfo, private url: string) {}
  playbackQuality(): { dropped: number; total: number } | null {
    if (!this.element.getVideoPlaybackQuality || !this.qualityStart) return null;
    const q = this.element.getVideoPlaybackQuality();
    return { dropped: Math.max(0, q.droppedVideoFrames - this.qualityStart.dropped), total: Math.max(0, q.totalVideoFrames - this.qualityStart.total) };
  }
  async *frames(signal: AbortSignal): AsyncIterable<DecodedFrame> {
    if (this.disposed || signal.aborted) throw cancelled();
    if (this.running) throw new Error('이미 프레임을 읽고 있습니다.');
    if (!this.element.requestVideoFrameCallback) throw new Error('이 브라우저는 실제 프레임 시각 API를 지원하지 않습니다. WebView/브라우저 업데이트가 필요합니다.');
    this.running = true;
    this.audit = new FrameAudit();
    this.qualityStart = null;
    const video = this.element;
    const controller = new AbortController();
    const stop = () => controller.abort();
    signal.addEventListener('abort', stop, { once: true }); this.lifetime.signal.addEventListener('abort', stop, { once: true });
    const active = controller.signal;
    video.pause(); video.controls = false;
    this.canvas.width = this.info.width; this.canvas.height = this.info.height;
    const ctx = this.canvas.getContext('2d');
    try {
      if (!ctx) throw new Error('Canvas 2D 사용 불가');
      if (video.currentTime !== 0) { const seek = waitEvent(video, 'seeked', active); video.currentTime = 0; await seek; }
      const quality = video.getVideoPlaybackQuality?.();
      this.qualityStart = quality ? { dropped: quality.droppedVideoFrames, total: quality.totalVideoFrames } : null;
      let started = false;
      while (!video.ended) {
        if (active.aborted) throw cancelled();
        const next = new Promise<DecodedFrame | null>((resolve, reject) => {
          let callback = 0;
          const finish = (frame: DecodedFrame | null, error?: unknown) => {
            clearTimeout(timer); video.cancelVideoFrameCallback(callback);
            video.removeEventListener('ended', ended); video.removeEventListener('error', failed); active.removeEventListener('abort', abort);
            error ? reject(error) : resolve(frame);
          };
          const ended = () => finish(null);
          const failed = () => finish(null, new Error('재생 중 디코딩 오류'));
          const abort = () => finish(null, cancelled());
          const timer = setTimeout(() => finish(null, new Error('15초 동안 새 프레임이 없습니다. 화면 전환/코덱/재생 상태를 확인하세요.')), 15000);
          video.addEventListener('ended', ended, { once: true }); video.addEventListener('error', failed, { once: true }); active.addEventListener('abort', abort, { once: true });
          callback = video.requestVideoFrameCallback((_now, meta) => {
            try {
              if (video.videoWidth !== this.info.width || video.videoHeight !== this.info.height) throw new Error('영상 중간에 크기가 변경되었습니다.');
              this.audit.add(meta.mediaTime, meta.presentedFrames);
              ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
              finish({ t: meta.mediaTime, pixels: this.canvas, release() { /* Single reusable buffer; valid until next iteration. */ } });
            } catch (error) { finish(null, error); }
          });
          if (!started) { started = true; void video.play().catch(error => finish(null, error)); }
        });
        const frame = await next;
        if (!frame) break;
        yield frame;
      }
    } finally {
      controller.abort(); video.pause(); video.controls = !this.disposed; this.running = false;
      signal.removeEventListener('abort', stop); this.lifetime.signal.removeEventListener('abort', stop);
      this.canvas.width = 0; this.canvas.height = 0;
    }
  }
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true; this.lifetime.abort();
    this.element.pause(); this.element.removeAttribute('src'); this.element.load(); this.element.remove();
    URL.revokeObjectURL(this.url); this.canvas.width = 0; this.canvas.height = 0;
  }
}

/** Playback-observed frames, not a complete offline decoder. No network or persistent files. */
export class BrowserVideoSource implements VideoSource {
  constructor(private mount: HTMLElement) {}
  async open(selection: unknown, signal: AbortSignal): Promise<BrowserVideoSession> {
    if (!(selection instanceof File) || selection.size === 0) throw new Error('비어 있지 않은 영상 파일을 선택하세요.');
    if (signal.aborted) throw cancelled();
    const video = document.createElement('video');
    video.controls = true; video.muted = true; video.playsInline = true; video.preload = 'metadata'; video.setAttribute('aria-label', '선택한 영상 미리보기');
    const url = URL.createObjectURL(selection);
    this.mount.replaceChildren(video);
    try {
      const ready = waitEvent(video, 'loadedmetadata', signal); video.src = url; await ready;
      if (!Number.isFinite(video.duration) || video.duration <= 0 || video.videoWidth <= 0 || video.videoHeight <= 0) throw new Error('유효한 길이/프레임 크기를 읽지 못했습니다.');
      return new BrowserVideoSession(video, { width: video.videoWidth, height: video.videoHeight, durationSeconds: video.duration,
        orientation: video.videoWidth >= video.videoHeight ? 'landscape' : 'portrait', rotationAppliedDegrees: null }, url);
    } catch (error) {
      video.pause(); video.removeAttribute('src'); video.load(); video.remove(); URL.revokeObjectURL(url); throw error;
    }
  }
}
