import { PredictionDetail } from '../../types';
import { EggResultCard, EggResultPlaceholder } from './EggResultCard';

interface ResultsPanelProps {
  result: PredictionDetail | null;
  crops: Record<number, string>;
}

const GRID_CLASS =
  'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-space-md';

export function ResultsPanel({ result, crops }: ResultsPanelProps) {
  const boxes = result?.detection_boxes ?? [];
  const trayLabel = result
    ? `Tray #${result.tray_id.slice(0, 8).toUpperCase()}`
    : 'Awaiting Capture';
  const processingTime = result?.processing_time_ms
    ? `${(result.processing_time_ms / 1000).toFixed(2)}s`
    : '—';
  const cardCount = Math.max(boxes.length, 5);

  return (
    <section className="flex flex-col bg-white/70 backdrop-blur-2xl border border-white/80 rounded-3xl p-space-md shadow-[0_12px_36px_0_rgba(0,80,203,0.06)] gap-space-md">
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-space-sm bg-gradient-to-r from-blue-50/80 to-emerald-50/50 border border-blue-100/60 p-space-md rounded-2xl">
        <div className="flex items-center gap-space-sm min-w-0">
          <span className="material-symbols-outlined text-[#0066ff] text-[22px]">
            analytics
          </span>
          <span className="font-headline-lg text-headline-lg text-[#181c1f] uppercase tracking-tight font-bold truncate">
            Last Capture Results — {trayLabel}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-space-xs">
          <span className="rounded-full bg-white/80 border border-blue-200/70 px-3 py-1 font-code-metric text-label-sm text-[#0050cb] font-bold">
            {boxes.length} / 5 DETECTED
          </span>
          <span className="rounded-full bg-white/80 border border-emerald-200/70 px-3 py-1 font-code-metric text-label-sm text-emerald-700 font-bold">
            {processingTime} INFERENCE
          </span>
          {result && result.needs_review_count > 0 && (
            <span className="rounded-full bg-amber-50 border border-amber-300 px-3 py-1 font-code-metric text-label-sm text-amber-800 font-bold">
              {result.needs_review_count} REVIEW
            </span>
          )}
        </div>
      </div>

      {result && boxes.length === 0 && (
        <div className="rounded-2xl border border-dashed border-amber-300 bg-amber-50/70 px-4 py-3">
          <p className="font-body-md text-body-md text-amber-900">
            No eggs were detected in this frame. Realign the tray and capture
            again, or lower the confidence threshold on the backend.
          </p>
        </div>
      )}

      <div className={GRID_CLASS} id="egg-cards-grid">
        {boxes.length > 0
          ? boxes.map((box, index) => (
              <EggResultCard
                key={box.id}
                box={box}
                index={index + 1}
                thumbnail={crops[box.id]}
              />
            ))
          : Array.from({ length: cardCount }, (_, index) => (
              <EggResultPlaceholder key={`empty-${index}`} index={index + 1} />
            ))}
      </div>
    </section>
  );
}
