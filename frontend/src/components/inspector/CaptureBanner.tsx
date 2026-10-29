interface CaptureBannerProps {
  canCapture: boolean;
  busy: boolean;
  onCapture: () => void;
}

export function CaptureBanner({
  canCapture,
  busy,
  onCapture,
}: CaptureBannerProps) {
  return (
    <div className="mt-space-md p-space-sm sm:p-space-md bg-white/60 border border-white/80 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-space-md shadow-sm">
      <div className="flex items-center gap-space-md w-full sm:w-auto">
        <div className="w-10 h-10 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center shrink-0 text-amber-600 shadow-sm">
          <span className="material-symbols-outlined text-[24px]">
            priority_high
          </span>
        </div>
        <div className="flex flex-col">
          <div className="flex items-center flex-wrap gap-space-xs">
            <span className="font-headline-md text-headline-md text-[#181c1f] tracking-tight font-semibold">
              Ready for Frame Classification
            </span>
            <span
              className={`font-code-metric text-label-sm border px-space-xs py-0.5 rounded font-bold ${
                canCapture
                  ? 'text-[#0066ff] bg-blue-50 border-blue-200/60'
                  : 'text-[#727687] bg-slate-50 border-slate-200'
              }`}
            >
              {canCapture ? 'TRIGGER READY' : 'CAMERA OFFLINE'}
            </span>
          </div>
          <div className="flex items-center flex-wrap gap-space-xs mt-0.5 text-[#525b6a] font-body-md text-body-md">
            <span>Press</span>
            <span className="inline-flex items-center justify-center px-space-sm py-0.5 bg-white border border-slate-200 rounded text-[#181c1f] font-code-metric text-label-sm shadow-[0_1px_2px_rgba(0,0,0,0.06)]">
              Spacebar
            </span>
            <span>to capture &amp; analyze active frame</span>
          </div>
        </div>
      </div>

      <button
        className="w-full sm:w-auto px-space-xl py-2.5 rounded-full bg-gradient-to-r from-[#10b981] to-[#059669] text-white font-headline-md text-headline-md hover:shadow-[0_0_24px_rgba(16,185,129,0.45)] transition-all flex items-center justify-center gap-space-sm border border-emerald-300/40 shadow-[0_4px_16px_rgba(16,185,129,0.3)] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
        id="btn-capture"
        type="button"
        onClick={onCapture}
        disabled={!canCapture || busy}
      >
        <span className="material-symbols-outlined text-[20px]">
          {busy ? 'progress_activity' : 'camera'}
        </span>
        <span className="font-semibold text-sm">
          {busy ? 'Analyzing...' : 'Capture Frame'}
        </span>
      </button>
    </div>
  );
}
