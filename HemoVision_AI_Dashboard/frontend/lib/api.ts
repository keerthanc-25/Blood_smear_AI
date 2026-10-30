export type CellClass = 'RBC' | 'WBC' | 'Platelet';

export type MorphologyEntry = {
  count: number;
  percentage: number;
};

export type Detection = {
  id: number;
  class: CellClass;
  detection_confidence: number;
  bbox: number[];
  morphology?: string;
  morphology_confidence?: number;
  morphology_probabilities?: Record<string, number>;
};

export type AnalysisResult = {
  success: boolean;
  sample_id: string;
  elapsed_seconds: number;
  image_size: {
    width: number;
    height: number;
  };

  model?: {
    detector?: string;
    detector_weights?: string;
    rbc_morphology?: string;
    rbc_morphology_weights?: string;
    wbc_morphology?: string;
    wbc_morphology_weights?: string;
  };

  counts: {
    RBC: number;
    WBC: number;
    Platelet: number;
  };

  total_cells: number;

  percentages: {
    RBC: number;
    WBC: number;
    Platelet: number;
  };

  average_confidence: {
    RBC: number;
    WBC: number;
    Platelet: number;
  };

  morphology: Record<string, MorphologyEntry>;

  rbc_morphology?: Record<string, MorphologyEntry>;
  wbc_morphology?: Record<string, MorphologyEntry>;

  rbc_morphology_average_confidence?: number;
  wbc_morphology_average_confidence?: number;

  detections: Detection[];
  annotated_image: string;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function analyzeImage(
  file: File,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch(`${API_URL}/api/analyze`, {
    method: 'POST',
    body: form,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'Analysis failed.');
  }

  return data as AnalysisResult;
}
