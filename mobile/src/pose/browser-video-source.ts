import { koreanMetadataTime } from './recorded-at';
import type { DecodedFrame, VideoInfo, VideoSession, VideoSource } from './ports';
import { FrameAudit } from './frame-audit';
import { CompleteFrameAudit } from './complete-frame-audit';
const cancelled = () => new DOMException('작업을 취소했습니다.', 'AbortError');
const inputFor = async (file: File) => { const { Input, BlobSource, MP4 } = await import('mediabunny'); return new Input({ source: new BlobSource(file), formats: [MP4] }); };


async function supportedTrack(input: Awaited<ReturnType<typeof inputFor>>) {
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('파일에 비디오 트랙이 없습니다.');
  const codec = await track.getCodec();
  if (codec !== 'avc' && codec !== 'hevc') throw new Error(`지원하지 않는 영상 코덱: ${codec ?? '알 수 없음'}. MP4/H.264 또는 HEVC(H.265)가 필요합니다.`);
  if (!await track.canDecode()) throw new Error(`이 브라우저/기기는 ${codec === 'hevc' ? 'HEVC(H.265)' : 'H.264'} 디코딩을 지원하지 않습니다. 파일 형식은 지원하지만 현재 실행 환경의 디코더를 사용할 수 없습니다.`);
  return track;
}

export class BrowserVideoSession implements VideoSession {
  audit = new FrameAudit();
  verification: CompleteFrameAudit | null = null;
  private disposed = false;
  private running = false;
  private lifetime = new AbortController();
  private canvas = document.createElement('canvas');
  constructor(readonly element: HTMLVideoElement, readonly info: VideoInfo, private url: string, private file: File) {}
  async *frames(signal: AbortSignal): AsyncIterable<DecodedFrame> {
    if (this.disposed || signal.aborted) throw cancelled();
    if (this.running) throw new Error('이미 프레임을 읽고 있습니다.');
    this.running = true; this.audit = new FrameAudit(); this.verification = null;
    let input: Awaited<ReturnType<typeof inputFor>>;
    try { input = await inputFor(this.file); } catch (error) { this.running = false; throw error; }
    if (signal.aborted || this.disposed) { input.dispose(); this.running = false; throw cancelled(); }
    const stop = () => input.dispose();
    const check = () => { if (signal.aborted || this.disposed) throw cancelled(); };
    signal.addEventListener('abort', stop, { once: true });
    this.lifetime.signal.addEventListener('abort', stop, { once: true });
    this.element.pause(); this.element.controls = false;
    try {
      const track = await supportedTrack(input); check();
      const { EncodedPacketSink, VideoSampleSink } = await import('mediabunny');
      const timestamps: number[] = [];
      for await (const packet of new EncodedPacketSink(track).packets(undefined, undefined, { metadataOnly: true })) {
        check(); timestamps.push(packet.timestamp);
      }
      this.verification = new CompleteFrameAudit(timestamps);
      this.canvas.width = this.info.width; this.canvas.height = this.info.height;
      const context = this.canvas.getContext('2d');
      if (!context) throw new Error('Canvas 2D 사용 불가');
      for await (const sample of new VideoSampleSink(track).samples()) {
        try {
          check();
          if (sample.displayWidth !== this.info.width || sample.displayHeight !== this.info.height)
            throw new Error('영상 중간의 프레임 크기 변경은 지원하지 않습니다.');
          const t = this.verification.accept(sample.timestamp);
          sample.draw(context, 0, 0);
          this.audit.add(t, this.verification.count);
          yield { t, pixels: this.canvas, release() { /* Valid until next iteration; single reusable canvas. */ } };
        } finally { sample.close(); }
      }
      check(); this.verification.finish();
    } catch (error) { check(); throw error; }
    finally {
      input.dispose(); signal.removeEventListener('abort', stop); this.lifetime.signal.removeEventListener('abort', stop);
      this.canvas.width = 0; this.canvas.height = 0; this.running = false; this.element.controls = !this.disposed;
    }
  }
  /** Exact decoded frame for inspection; no HTMLVideoElement seeking approximation. */
  async drawFrame(t: number, canvas: HTMLCanvasElement, signal: AbortSignal): Promise<void> {
    const input = await inputFor(this.file);
    const stop = () => input.dispose();
    signal.addEventListener('abort', stop, { once: true });
    try {
      if (signal.aborted || this.disposed) throw cancelled();
      const track = await supportedTrack(input);
      const { VideoSampleSink } = await import('mediabunny');
      const sample = await new VideoSampleSink(track).getSample(t + 0.000001);
      if (!sample) throw new Error('해당 프레임을 찾지 못했습니다.');
      try {
        if (signal.aborted || this.disposed) throw cancelled();
        if (Math.abs(sample.timestamp - t) > 0.0000011) throw new Error('오버레이 프레임 시각 불일치');
        canvas.width = this.info.width; canvas.height = this.info.height;
        const ctx = canvas.getContext('2d'); if (!ctx) throw new Error('Canvas 2D 사용 불가');
        sample.draw(ctx, 0, 0);
      } finally { sample.close(); }
    } finally { signal.removeEventListener('abort', stop); input.dispose(); }
  }
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true; this.lifetime.abort();
    this.element.pause(); this.element.removeAttribute('src'); this.element.load(); this.element.remove();
    URL.revokeObjectURL(this.url); this.canvas.width = 0; this.canvas.height = 0;
  }
}

/** Offline MP4/H.264 또는 HEVC(H.265) decoding; HTML video is used only for preview. No playback fallback. */
export class BrowserVideoSource implements VideoSource {
  constructor(private mount: HTMLElement) {}
  async open(selection: unknown, signal: AbortSignal): Promise<BrowserVideoSession> {
    if (!(selection instanceof File) || !selection.size) throw new Error('비어 있지 않은 영상 파일을 선택하세요.');
    if (signal.aborted) throw cancelled();
    const input = await inputFor(selection);
    if (signal.aborted) { input.dispose(); throw cancelled(); }
    const stop = () => input.dispose();
    signal.addEventListener('abort', stop, { once: true });
    try {
      const track = await supportedTrack(input);
      const width = await track.getDisplayWidth(), height = await track.getDisplayHeight();
      const durationSeconds = await track.computeDuration();
      const rotationAppliedDegrees = await track.getRotation();
      const recordedAtMetadata = koreanMetadataTime((await input.getMetadataTags()).creationTime);
      if (signal.aborted) throw cancelled();
      if (!(width > 0 && height > 0 && Number.isFinite(durationSeconds) && durationSeconds > 0)) throw new Error('유효하지 않은 영상 정보');
      const video = document.createElement('video');
      video.controls = true; video.muted = true; video.playsInline = true; video.preload = 'metadata'; video.setAttribute('aria-label', '선택한 영상 미리보기');
      const url = URL.createObjectURL(selection); video.src = url; this.mount.replaceChildren(video);
      return new BrowserVideoSession(video, { width, height, durationSeconds, recordedAtMetadata, orientation: width >= height ? 'landscape' : 'portrait', rotationAppliedDegrees }, url, selection);
    } catch (error) { if (signal.aborted) throw cancelled(); throw error; }
    finally { signal.removeEventListener('abort', stop); input.dispose(); }
  }
}
