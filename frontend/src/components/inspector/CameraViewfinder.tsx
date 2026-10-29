import { RefObject } from 'react';
import { CameraStatus } from '../../hooks/useCamera';

interface CameraViewfinderProps {
  videoRef: RefObject<HTMLVideoElement>;
  status: CameraStatus;
  error: string | null;
  busy: boolean;
  previewDataUrl: string | null;
  onEnable: () => void;
  onUpload: () => void;
}

export function CameraViewfinder({
  videoRef,
  status,
  error,
  busy,
  previewDataUrl,
  onEnable,
  onUpload,
}: CameraViewfinderProps) {
  const showPermissionCard = status !== 'ready';

  return (
    <div className="relative w-full aspect-[16/9] bg-gradient-to-b from-[#eaf1f8] to-[#dde7f2] rounded-2xl overflow-hidden flex items-center justify-center group select-none border border-blue-100/60 shadow-inner">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 h-full w-full object-cover"
      />

      {!previewDataUrl && (
        <div className="absolute inset-0 bg-gradient-to-b from-[#eaf1f8]/25 to-[#dde7f2]/60" />
      )}
      <div className="absolute inset-0 bg-gradient-to-t from-white/40 via-transparent to-white/30 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(0,102,255,0.05)_100%)] pointer-events-none" />

      <div className="absolute top-6 left-6 w-8 h-8 border-t-2 border-l-2 border-[#0066ff] pointer-events-none drop-shadow-sm" />
      <div className="absolute top-6 right-6 w-8 h-8 border-t-2 border-r-2 border-[#0066ff] pointer-events-none drop-shadow-sm" />
      <div className="absolute bottom-6 left-6 w-8 h-8 border-b-2 border-l-2 border-[#0066ff] pointer-events-none drop-shadow-sm" />
      <div className="absolute bottom-6 right-6 w-8 h-8 border-b-2 border-r-2 border-[#0066ff] pointer-events-none drop-shadow-sm" />

      {status === 'ready' && !previewDataUrl && (
        <>
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-50">
            <div className="w-16 h-16 rounded-full border border-blue-400/40 flex items-center justify-center">
              <div className="w-1.5 h-1.5 bg-[#0066ff] rounded-full" />
            </div>
            <div className="absolute w-24 h-px bg-blue-400/40" />
            <div className="absolute h-24 w-px bg-blue-400/40" />
          </div>
          <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full bg-white/80 border border-emerald-200 px-3 py-1 backdrop-blur-md">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-code-metric text-label-sm text-emerald-700 font-bold">
              LIVE OPTICAL FEED
            </span>
          </div>
          <div className="absolute bottom-4 left-4 rounded bg-black/45 px-2 py-1 backdrop-blur-sm">
            <span className="font-code-metric text-label-sm text-white/90">
              FRAME READY // HUD ALIGNED
            </span>
          </div>
        </>
      )}

      {previewDataUrl && busy && (
        <div className="absolute inset-0 z-20">
          <img
            src={previewDataUrl}
            alt="Captured tray frame"
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-blue-950/25" />
          <div className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-[#27dfbc]/50 to-transparent scanline" />
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white">
            <span className="material-symbols-outlined text-[42px] animate-pulse">
              center_focus_strong
            </span>
            <div className="rounded-full bg-black/55 px-4 py-2 backdrop-blur-md">
              <span className="font-code-metric text-label-lg text-white font-bold tracking-wide">
                ANALYZING FRAME...
              </span>
            </div>
          </div>
        </div>
      )}

      {showPermissionCard && (
        <div
          id="viewport-camera-permission"
          className="absolute inset-0 flex items-center justify-center p-space-md z-10"
        >
          <div className="flex flex-col items-center text-center max-w-md bg-white/85 border border-white/90 backdrop-blur-2xl rounded-3xl p-space-lg shadow-[0_16px_40px_rgba(0,80,203,0.12)] ring-1 ring-blue-500/10 gap-space-md">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-[#0050cb] border border-blue-400/30 flex items-center justify-center text-white shadow-[0_8px_20px_rgba(0,102,255,0.3)]">
              <span className="material-symbols-outlined text-[32px]">
                videocam
              </span>
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="font-headline-lg text-headline-lg text-[#181c1f] tracking-tight">
                {error ? 'Camera Unavailable' : 'Camera Access Required'}
              </span>
              <p className="font-body-md text-body-md text-[#525b6a]">
                {error ??
                  'Please allow camera permissions to initialize the live optical feed and automated egg defect inspection stream.'}
              </p>
            </div>
            <div className="flex flex-col sm:flex-row items-center gap-space-sm w-full mt-1">
              <button
                className="w-full sm:w-auto flex-1 px-space-lg py-2.5 bg-gradient-to-r from-[#10b981] to-[#059669] text-white font-headline-md text-headline-md rounded-full hover:shadow-[0_0_24px_rgba(16,185,129,0.4)] transition-all flex items-center justify-center gap-space-xs shadow-[0_4px_16px_rgba(16,185,129,0.3)] border border-emerald-300/40 active:scale-95 disabled:opacity-70 disabled:cursor-wait"
                type="button"
                onClick={onEnable}
                disabled={status === 'starting'}
              >
                <span className="material-symbols-outlined text-[20px]">
                  {status === 'starting' ? 'progress_activity' : 'videocam'}
                </span>
                <span className="font-semibold text-sm">
                  {status === 'starting'
                    ? 'Starting Camera...'
                    : error
                      ? 'Retry Camera'
                      : 'Enable Camera'}
                </span>
              </button>
              <button
                className="w-full sm:w-auto px-space-lg py-2.5 bg-white/80 text-[#0050cb] border border-blue-200 font-headline-md text-headline-md rounded-full hover:bg-blue-50 transition-all flex items-center justify-center gap-space-xs active:scale-95"
                type="button"
                onClick={onUpload}
              >
                <span className="material-symbols-outlined text-[20px]">
                  upload
                </span>
                <span className="font-semibold text-sm">Upload Image</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
