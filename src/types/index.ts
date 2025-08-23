export interface ClassificationResult {
  label: string
  percent: number
}

export interface PredictionResult {
  class: string
  confidence: number
  classes: ClassificationResult[]
  image?: string
  heatmap?: string
}

export interface ApiResponse {
  prediction?: string
  confidence?: number
  probabilities?: Record<string, number>
  is_mri?: boolean
  message?: string
  error?: string
}