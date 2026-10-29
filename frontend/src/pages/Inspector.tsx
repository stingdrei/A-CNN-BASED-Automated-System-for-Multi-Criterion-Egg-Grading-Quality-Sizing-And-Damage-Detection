import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import { useCamera } from '../hooks/useCamera';
import { CapturedFrame, HealthStatus, PredictionDetail } from '../types';
import { getErrorMessage } from '../utils/errorHandler';
import {
  buildCropThumbnails,
  fileToFrame,
} from '../utils/frames';
import { CameraViewfinder } from '../components/inspector/CameraViewfinder';
import { CaptureBanner } from '../components/inspector/CaptureBanner';
import { ResultsPanel } from '../components/inspector/ResultsPanel';
import { StatusFooter } from '../components/inspector/StatusFooter';

export function Inspector() {
  const camera = useCamera();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analysisInFlightRef = useRef(false);
  const [result, setResult] = useState<PredictionDetail | null>(null);
  const [crops, setCrops] = useState<Record<number, string>>({});
  const [previewDataUrl, setPreviewDataUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    let active = true;
    const loadHealth = async () => {
      try {
        const status = await apiClient.getHealth();
        if (active) setHealth(status);
      } catch {
        if (active) setHealth(null);
      }
    };
    void loadHealth();
    const timer = window.setInterval(loadHealth, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const analyzeFrame = useCallback(async (frame: CapturedFrame) => {
    if (analysisInFlightRef.current) return;
    analysisInFlightRef.current = true;
    setBusy(true);
    setError(null);
    setPreviewDataUrl(frame.dataUrl);
    try {
      const detail: PredictionDetail = await apiClient.gradeImage(frame.file);
      let thumbnails: Record<number, string> = {};
      try {
        thumbnails = await buildCropThumbnails(frame, detail.detection_boxes);
      } catch {
        thumbnails = {};
      }
      setResult(detail);
      setCrops(thumbnails);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      analysisInFlightRef.current = false;
      setBusy(false);
      setPreviewDataUrl(null);
    }
  }, []);

  const handleCapture = useCallback(async () => {
    if (busy || analysisInFlightRef.current) return;
    const frame = await camera.capture();
    if (frame) await analyzeFrame(frame);
  }, [analyzeFrame, busy, camera]);

  const handleUpload = useCallback(() => {
    if (!busy) fileInputRef.current?.click();
  }, [busy]);

  const handleFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      try {
        const frame = await fileToFrame(file);
        await analyzeFrame(frame);
      } catch (err) {
        setError(getErrorMessage(err));
      }
    },
    [analyzeFrame],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      const target = event.target as HTMLElement | null;
      const isTyping =
        target?.tagName === 'INPUT' ||
        target?.tagName === 'TEXTAREA' ||
        target?.isContentEditable;
      if (isTyping) return;

      if (event.code === 'Space') {
        event.preventDefault();
        void handleCapture();
      } else if (event.key.toLowerCase() === 'r') {
        void handleCapture();
      } else if (event.key.toLowerCase() === 'u') {
        handleUpload();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleCapture, handleUpload]);

  const exportBatch = useCallback(() => {
    if (!result) return;
    const payload = {
      tray_id: result.tray_id,
      exported_at: new Date().toISOString(),
      prediction: result,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `egg-tray-${result.tray_id.slice(0, 8)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <main className="w-full bg-[#f8fbfe] bg-[radial-gradient(ellipse_90%_70%_at_20%_-10%,rgba(0,102,255,0.09),transparent_60%),radial-gradient(ellipse_80%_60%_at_85%_15%,rgba(39,223,188,0.11),transparent_55%),radial-gradient(ellipse_90%_60%_at_50%_105%,rgba(16,185,129,0.08),transparent_60%),radial-gradient(circle_at_50%_45%,rgba(255,255,255,0.85),rgba(240,246,253,0.95))] min-h-screen relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-32 left-1/4 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-blue-400/15 to-transparent blur-3xl" />
        <div className="absolute top-1/3 -right-36 w-[540px] h-[540px] rounded-full bg-gradient-to-bl from-emerald-300/15 via-teal-200/10 to-transparent blur-3xl" />
        <div className="absolute -bottom-36 left-10 w-[640px] h-[640px] rounded-full bg-gradient-to-tr from-cyan-300/12 to-transparent blur-3xl" />
      </div>

      <div className="relative flex flex-col w-full">
        <div className="p-4 sm:p-gutter-desktop flex flex-col gap-space-lg max-w-[1720px] mx-auto w-full">
          <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm bg-white/60 backdrop-blur-2xl border border-white/80 rounded-2xl px-space-md py-space-sm shadow-sm">
            <div className="flex items-center gap-space-sm">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#0050cb] to-[#0066ff] flex items-center justify-center text-white shadow-[0_6px_16px_rgba(0,102,255,0.28)]">
                <span className="material-symbols-outlined text-[24px] is-filled">
                  egg
                </span>
              </div>
              <div className="flex flex-col">
                <span className="font-headline-lg text-headline-lg text-[#181c1f] tracking-tight font-bold leading-none">
                  Egg-CV
                </span>
                <span className="font-label-sm text-label-sm text-[#727687] uppercase tracking-widest mt-1">
                  Optical Grading Station
                </span>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-space-xs">
              <span className="rounded-full bg-blue-50 border border-blue-200/70 px-3 py-1 font-code-metric text-label-sm text-[#0050cb] font-bold">
                MVP // LOCAL OPERATOR
              </span>
              <span className="rounded-full bg-emerald-50 border border-emerald-200/70 px-3 py-1 font-code-metric text-label-sm text-emerald-700 font-bold">
                AUTH-FREE
              </span>
            </div>
          </header>

          <section className="flex flex-col bg-white/70 backdrop-blur-2xl border border-white/80 rounded-3xl p-space-md shadow-[0_12px_36px_0_rgba(0,80,203,0.07)] relative overflow-hidden">
            <CameraViewfinder
              videoRef={camera.videoRef}
              status={camera.status}
              error={camera.error}
              busy={busy}
              previewDataUrl={previewDataUrl}
              onEnable={() => void camera.start()}
              onUpload={handleUpload}
            />
            <CaptureBanner
              canCapture={camera.status === 'ready'}
              busy={busy}
              onCapture={() => void handleCapture()}
            />
          </section>

          {error && (
            <div className="rounded-2xl border border-red-200 bg-red-50/90 px-4 py-3 shadow-sm">
              <div className="flex items-start gap-2">
                <span className="material-symbols-outlined text-[20px] text-red-600">
                  error
                </span>
                <p className="font-body-md text-body-md text-red-800">{error}</p>
              </div>
            </div>
          )}

          <ResultsPanel result={result} crops={crops} />
          <StatusFooter health={health} result={result} onExport={exportBatch} />
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        className="hidden"
        onChange={(event) => void handleFileChange(event)}
      />
    </main>
  );
}
