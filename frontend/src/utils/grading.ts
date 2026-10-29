import { DetectionBox, Grade } from '../types';

export interface GradeMeta {
  label: string;
  badgeClass: string;
  cardClass: string;
  thumbClass: string;
}

export const GRADE_META: Record<Grade, GradeMeta> = {
  A: {
    label: 'Grade A (Pass)',
    badgeClass: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    cardClass: 'border-slate-200/70 hover:border-blue-400',
    thumbClass: 'border-emerald-200 text-emerald-700',
  },
  B: {
    label: 'Grade B (Warning)',
    badgeClass: 'bg-amber-50 border-amber-300 text-amber-800',
    cardClass: 'border-amber-200 hover:border-amber-400',
    thumbClass: 'border-amber-300 text-amber-800',
  },
  C: {
    label: 'Grade C (Review)',
    badgeClass: 'bg-orange-50 border-orange-300 text-orange-800',
    cardClass: 'border-orange-200 hover:border-orange-400',
    thumbClass: 'border-orange-300 text-orange-800',
  },
  Reject: {
    label: 'Reject (Damaged)',
    badgeClass: 'bg-red-50 border-red-200 text-red-700',
    cardClass: 'border-red-200 hover:border-red-400',
    thumbClass: 'border-red-300 text-red-700',
  },
  'N/A': {
    label: 'Needs Review',
    badgeClass: 'bg-slate-50 border-slate-200 text-slate-600',
    cardClass: 'border-slate-200/70 hover:border-slate-400',
    thumbClass: 'border-slate-300 text-slate-600',
  },
};

export function resolveGrade(box: DetectionBox): {
  grade: Grade;
  meta: GradeMeta;
  isProvisional: boolean;
} {
  const finalGrade = box.grade && box.grade !== 'N/A' ? box.grade : null;
  const grade = finalGrade ?? box.provisional_grade ?? box.grade ?? 'N/A';
  return {
    grade,
    meta: GRADE_META[grade],
    isProvisional: finalGrade === null,
  };
}

export function formatMetric(
  value: string | number | null | undefined,
  suffix = '',
  fallback = '—',
): string {
  if (value === null || value === undefined || value === '') return fallback;
  return `${value}${suffix}`;
}
