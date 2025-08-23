'use client'

import { useState, useCallback } from 'react'
import { Upload, Brain, Zap, Eye, Shield, Github, Coffee } from 'lucide-react'
import ImageUpload from '@/components/ImageUpload'
import ResultsDisplay from '@/components/ResultsDisplay'
import FeatureCard from '@/components/FeatureCard'
import { PredictionResult } from '@/types'

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handlePrediction = useCallback((predictionResult: PredictionResult) => {
    setResult(predictionResult)
    setIsLoading(false)
  }, [])

  const handleLoadingStart = useCallback(() => {
    setIsLoading(true)
    setResult(null)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">NeuroScan</h1>
                <p className="text-sm text-gray-600">Brain Tumor Classification</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
                title="View on GitHub"
              >
                <Github className="w-6 h-6" />
              </a>
              <a
                href="https://buymeacoffee.com/yahnaiduu"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-600 hover:text-orange-600 transition-colors"
                title="Buy me a coffee"
              >
                <Coffee className="w-6 h-6" />
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            AI-Powered Brain Tumor
            <span className="text-primary-600"> Classification</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Advanced deep learning model for accurate brain tumor detection and classification 
            with Grad-CAM visualization and AI-powered MRI validation.
          </p>
          
          {/* Medical Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-w-4xl mx-auto mb-8">
            <div className="flex items-start space-x-3">
              <Shield className="w-6 h-6 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-left">
                <h3 className="font-semibold text-yellow-800 mb-1">Medical Disclaimer</h3>
                <p className="text-sm text-yellow-700">
                  This tool is for research and educational purposes only. It should not be used for actual medical diagnosis. 
                  Always consult qualified medical professionals for medical advice and diagnosis.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <FeatureCard
            icon={<Brain className="w-8 h-8 text-primary-600" />}
            title="4 Tumor Types"
            description="Classifies Glioma, Meningioma, Pituitary tumors, and healthy brain tissue"
          />
          <FeatureCard
            icon={<Zap className="w-8 h-8 text-green-600" />}
            title="95% Accuracy"
            description="High-precision MobileNet model optimized for medical imaging"
          />
          <FeatureCard
            icon={<Eye className="w-8 h-8 text-purple-600" />}
            title="Grad-CAM Viz"
            description="Visual heatmaps showing which brain regions influenced predictions"
          />
          <FeatureCard
            icon={<Shield className="w-8 h-8 text-blue-600" />}
            title="AI Validation"
            description="Gemini Vision API validates MRI authenticity before classification"
          />
        </div>

        {/* Main Application */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-6 h-6 text-primary-600 mr-2" />
                Upload MRI Scan
              </h3>
              <ImageUpload
                onPrediction={handlePrediction}
                onLoadingStart={handleLoadingStart}
                isLoading={isLoading}
              />
            </div>

            {/* Classification Info */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Categories</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div>
                    <span className="font-medium text-gray-900">Glioma:</span>
                    <span className="text-gray-600 ml-1">Primary brain tumors from glial cells</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <div>
                    <span className="font-medium text-gray-900">Meningioma:</span>
                    <span className="text-gray-600 ml-1">Tumors from brain covering (meninges)</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <div>
                    <span className="font-medium text-gray-900">No Tumor:</span>
                    <span className="text-gray-600 ml-1">Healthy brain tissue</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <div>
                    <span className="font-medium text-gray-900">Pituitary:</span>
                    <span className="text-gray-600 ml-1">Tumors in pituitary gland</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div>
            <ResultsDisplay result={result} isLoading={isLoading} />
          </div>
        </div>

        {/* Technology Stack */}
        <div className="mt-16 text-center">
          <h3 className="text-2xl font-bold text-gray-900 mb-8">Built with Modern Technology</h3>
          <div className="flex flex-wrap justify-center items-center gap-8 text-gray-600">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 font-bold text-sm">TF</span>
              </div>
              <span>TensorFlow</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-green-600 font-bold text-sm">FL</span>
              </div>
              <span>Flask</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 font-bold text-sm">AI</span>
              </div>
              <span>Gemini Vision</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center">
                <span className="text-orange-600 font-bold text-sm">CV</span>
              </div>
              <span>OpenCV</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold">NeuroScan</span>
            </div>
            <p className="text-gray-400 mb-6">
              Made with ❤️ for Medical AI Research
            </p>
            <div className="flex justify-center space-x-6">
              <a
                href="https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                GitHub
              </a>
              <a
                href="https://buymeacoffee.com/yahnaiduu"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                Support
              </a>
            </div>
            <div className="mt-8 pt-8 border-t border-gray-800 text-sm text-gray-400">
              <p>© 2025 Yash Naidu. Licensed under MIT License.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}