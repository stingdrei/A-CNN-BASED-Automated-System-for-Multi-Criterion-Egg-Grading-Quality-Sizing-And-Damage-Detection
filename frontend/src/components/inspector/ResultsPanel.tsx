import { useState } from 'react';
import { PredictionDetail } from '../../types';
import { EggResultCard, EggResultPlaceholder } from './EggResultCard';

interface ResultsPanelProps {
  result: PredictionDetail | null;
  crops: Record<number, string>;
}

const GRID_CLASS =
  'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-space-md';

function AnnotatedPreview({ url, trayId }: { url: string; trayId: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;

  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-space-md rounded-2xl border border-slate-200 bg-slate-50/60 p-space-sm">
      <img
        src={url}
        alt={`Annotated capture for ${trayId}`}
        className="h-32 w-auto max-w-full rounded-xl border border-slate-200 bg-white object-contain"
        loading="lazy"
        onError={() => setFailed(true)}
      />
      <div className="flex flex-col gap-space-xs">
        <span className="font-code-metric text-label-sm text-[#424656] font-bold">
          ANNOTATED FRAME
        </span>
        <span className="font-body-md text-body-md text-[#727687]">
          Detector boxes overlaid on the captured tray.
        </span>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="font-code-metric text-label-sm text-[#0050cb] underline"
        >
          Open full size
        </a>
      </div>
    </div>
  );
}

export function ResultsPanel({ result, crops }: ResultsPanelProps) {
  const boxes = result?.detection_boxes ?? [];
  const trayLabel = result
    ? `Tray #${result.tray_id.slice(0, 8).toUpperCase()}`
    : 'Awaiting Capture';
  const processingTime = result?.processing_time_ms
    ? `${(result.processing_time_ms / 1000).toFixed(2)}s`
    : '—';
  const metadata = (result?.inference_metadata ?? {}) as Record<
    string,
    unknown
  >;
  const cleanlinessApproved =
    metadata.cleanliness_release_approved === true;
  const calibrationReady =
    metadata.calibration_version !== 'uncalibrated' &&
    metadata.size_calibrated === true &&
    metadata.weight_calibrated === true;
  const maxTrayEggs = Number(metadata.max_tray_eggs) || 5;
  const cardCount = Math.max(boxes.length, maxTrayEggs);

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
            {boxes.length} / {maxTrayEggs} DETECTED
          </span>
          <span className="rounded-full bg-white/80 border border-emerald-200/70 px-3 py-1 font-code-metric text-label-sm text-emerald-700 font-bold">
            {processingTime} INFERENCE
          </span>
          {result && result.needs_review_count > 0 && (
            <span className="rounded-full bg-amber-50 border border-amber-300 px-3 py-1 font-code-metric text-label-sm text-amber-800 font-bold">
              {result.needs_review_count} REVIEW
            </span>
          )}
          {result && !cleanlinessApproved && (
            <span
              className="rounded-full bg-amber-50 border border-amber-300 px-3 py-1 font-code-metric text-label-sm text-amber-800 font-bold"
              title="Cleanliness model has not passed physical release validation"
            >
              BETA · CLEANLINESS UNVALIDATED
            </span>
          )}
          {result && !calibrationReady && (
            <span
              className="rounded-full bg-slate-100 border border-slate-300 px-3 py-1 font-code-metric text-label-sm text-slate-700 font-bold"
              title="Size and weight require camera calibration"
            >
              SIZE/WEIGHT UNCALIBRATED
            </span>
          )}
        </div>
      </div>

      {result?.annotated_url && (
        <AnnotatedPreview url={result.annotated_url} trayId={trayLabel} />
      )}

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
