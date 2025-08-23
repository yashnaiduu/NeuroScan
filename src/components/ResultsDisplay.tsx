'use client'

import { useState } from 'react'
import { Brain, TrendingUp, Eye, Loader2, AlertCircle } from 'lucide-react'
import { PredictionResult } from '@/types'

interface ResultsDisplayProps {
  result: PredictionResult | null
  isLoading: boolean
}

export default function ResultsDisplay({ result, isLoading }: ResultsDisplayProps) {
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [heatmapLoading, setHeatmapLoading] = useState(false)

  const generateHeatmap = async () => {
    setHeatmapLoading(true)
    // Simulate heatmap generation
    await new Promise(resolve => setTimeout(resolve, 2000))
    setShowHeatmap(true)
    setHeatmapLoading(false)
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getProgressBarColor = (label: string, percent: number) => {
    if (percent < 1) return 'bg-gray-300'
    
    switch (label.toLowerCase()) {
      case 'glioma':
        return 'bg-red-500'
      case 'meningioma':
        return 'bg-blue-500'
      case 'no tumor':
      case 'notumor':
        return 'bg-green-500'
      case 'pituitary':
        return 'bg-purple-500'
      default:
        return 'bg-gray-400'
    }
  }

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="w-12 h-12 text-primary-600 animate-spin mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Analyzing MRI Scan</h3>
          <p className="text-gray-600 text-center">
            Our AI is processing your image and generating predictions<span className="loading-dots"></span>
          </p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="card">
        <div className="text-center py-12">
          <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready for Analysis</h3>
          <p className="text-gray-600">
            Upload an MRI scan to get started with AI-powered brain tumor classification
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <TrendingUp className="w-6 h-6 text-primary-600" />
          <h3 className="text-xl font-semibold text-gray-900">Classification Result</h3>
        </div>

        <div className="text-center mb-6">
          <div className="inline-flex items-center space-x-2 bg-primary-50 px-4 py-2 rounded-full mb-3">
            <Brain className="w-5 h-5 text-primary-600" />
            <span className="text-primary-700 font-medium">Prediction</span>
          </div>
          <h4 className="text-2xl font-bold text-gray-900 mb-2">{result.class}</h4>
          <p className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}% Confidence
          </p>
        </div>

        {/* Confidence Interpretation */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h5 className="font-medium text-gray-900 mb-1">Confidence Interpretation</h5>
              <p className="text-sm text-gray-600">
                {result.confidence >= 0.8 
                  ? 'High confidence prediction. The model is very certain about this classification.'
                  : result.confidence >= 0.6
                  ? 'Moderate confidence. Consider additional medical evaluation.'
                  : 'Low confidence prediction. Further analysis recommended.'}
              </p>
            </div>
          </div>
        </div>

        {/* Probability Breakdown */}
        <div>
          <h5 className="font-medium text-gray-900 mb-4">Probability Breakdown</h5>
          <div className="space-y-3">
            {result.classes.map((cls, index) => (
              <div key={index} className="flex items-center space-x-3">
                <div className="w-20 text-sm font-medium text-gray-700 flex-shrink-0">
                  {cls.label}
                </div>
                <div className="flex-1">
                  <div className="progress-bar">
                    <div
                      className={`progress-fill ${getProgressBarColor(cls.label, cls.percent)}`}
                      style={{ width: `${Math.max(cls.percent, 0.5)}%` }}
                    />
                  </div>
                </div>
                <div className="w-12 text-sm font-medium text-gray-900 text-right">
                  {cls.percent.toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap Section */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Eye className="w-6 h-6 text-purple-600" />
            <h3 className="text-xl font-semibold text-gray-900">Grad-CAM Visualization</h3>
          </div>
          {!showHeatmap && (
            <button
              onClick={generateHeatmap}
              disabled={heatmapLoading}
              className="btn-primary flex items-center space-x-2"
            >
              {heatmapLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  <span>Generate Heatmap</span>
                </>
              )}
            </button>
          )}
        </div>

        {showHeatmap ? (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-500 to-red-500 h-64 rounded-lg flex items-center justify-center">
              <p className="text-white font-medium">Heatmap Visualization</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <h5 className="font-medium text-blue-900 mb-2">How to Read the Heatmap</h5>
              <div className="text-sm text-blue-800 space-y-1">
                <p>• <span className="font-medium text-red-600">Red areas</span>: High influence on prediction</p>
                <p>• <span className="font-medium text-yellow-600">Yellow areas</span>: Moderate influence</p>
                <p>• <span className="font-medium text-blue-600">Blue areas</span>: Low influence</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <Eye className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-600">
              Generate a heatmap to see which brain regions influenced the AI's decision
            </p>
          </div>
        )}
      </div>

      {/* Medical Information */}
      {result.class !== 'Not MRI' && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Medical Information</h3>
          <div className="bg-gray-50 rounded-lg p-4">
            {result.class === 'Glioma' && (
              <div>
                <h5 className="font-medium text-gray-900 mb-2">About Glioma</h5>
                <p className="text-sm text-gray-700">
                  Gliomas are the most common primary brain tumors, arising from glial cells. 
                  They can be benign or malignant and require immediate medical attention for proper staging and treatment planning.
                </p>
              </div>
            )}
            {result.class === 'Meningioma' && (
              <div>
                <h5 className="font-medium text-gray-900 mb-2">About Meningioma</h5>
                <p className="text-sm text-gray-700">
                  Meningiomas arise from the meninges (brain covering) and are usually benign, slow-growing tumors. 
                  Treatment depends on size, location, and symptoms.
                </p>
              </div>
            )}
            {result.class === 'Pituitary' && (
              <div>
                <h5 className="font-medium text-gray-900 mb-2">About Pituitary Tumors</h5>
                <p className="text-sm text-gray-700">
                  Pituitary tumors can affect hormone production and may cause various symptoms. 
                  Most are benign but may require treatment depending on size and hormone effects.
                </p>
              </div>
            )}
            {result.class === 'No Tumor' && (
              <div>
                <h5 className="font-medium text-gray-900 mb-2">Healthy Brain Tissue</h5>
                <p className="text-sm text-gray-700">
                  The scan appears to show normal brain tissue without signs of tumor. 
                  However, this AI analysis should not replace professional medical evaluation.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}