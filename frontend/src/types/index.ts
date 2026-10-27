export interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
  is_active: boolean;
}

export interface Token {
  access_token: string;
  token_type: string;
}

export type Grade = 'A' | 'B' | 'C' | 'N/A' | 'Reject';
export type Size = 'Small' | 'Medium' | 'Large' | 'Unknown';
export type DamageStatus = 'Damaged' | 'Not Damaged' | 'Needs Review';
export type CleanlinessStatus = 'Clean' | 'Stained' | 'Needs Review' | 'Unavailable' | 'Unknown';

export interface EggConfidence {
  detection: number;
  damage?: number | null;
  cleanliness?: number | null;
}

export interface EggGeometry {
  length_mm?: number | null;
  breadth_mm?: number | null;
  quality?: string | null;
}

export interface ModelVersions {
  detector?: string | null;
  damage?: string | null;
  cleanliness?: string | null;
  calibration?: string | null;
}

export interface DetectionBox {
  id: number;
  prediction_id: number;
  egg_id?: number | null;
  class_name: string;
  confidence: EggConfidence;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  annotated: boolean;
  grade?: Grade | null;
  provisional_grade?: Grade | null;
  size?: Size | null;
  weight_g?: number | null;
  damage_status?: DamageStatus | null;
  cleanliness_status?: CleanlinessStatus | null;
  requires_review: boolean;
  review_reason?: string | null;
  geometry: EggGeometry;
  model_versions: ModelVersions;
}

export interface Prediction {
  id: number;
  file_name: string;
  file_type: string;
  tray_id: string;
  captured_at?: string | null;
  operator?: string | null;
  inference_metadata?: Record<string, unknown> | null;
  status: string;
  total_detections: number;
  damaged_count: number;
  not_damaged_count: number;
  needs_review_count: number;
  processing_time_ms: number | null;
  created_at: string;
  completed_at: string | null;
}

export interface PredictionDetail extends Prediction {
  result_json: Record<string, unknown> | null;
  annotated_path: string | null;
  detection_boxes: DetectionBox[];
}

export interface DashboardStats {
  total_predictions: number;
  total_images: number;
  total_detections: number;
  damaged_percentage: number;
  avg_processing_time_ms: number;
  recent_predictions: Prediction[];
  grade_distribution?: GradeDistribution;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  damage_model_loaded: boolean;
  cleanliness_model_loaded: boolean;
  database_connected: boolean;
}

export interface GradeDistribution {
  grade_a: number;
  grade_b: number;
  grade_c: number;
  grade_na: number;
  grade_reject: number;
  total: number;
}

export interface UserSettings {
  mm_per_pixel: number;
}
