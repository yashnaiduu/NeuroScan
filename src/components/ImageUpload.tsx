'use client'

import { useState, useCallback, useRef } from 'react'
import { Upload, X, Loader2, Image as ImageIcon } from 'lucide-react'
import { PredictionResult } from '@/types'

interface ImageUploadProps {
  onPrediction: (result: PredictionResult) => void
  onLoadingStart: () => void
  isLoading: boolean
}

export default function ImageUpload({ onPrediction, onLoadingStart, isLoading }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const validateFile = (file: File): string | null => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
    const maxSize = 16 * 1024 * 1024 // 16MB

    if (!allowedTypes.includes(file.type)) {
      return 'Please upload a valid image file (JPEG, PNG, BMP)'
    }

    if (file.size > maxSize) {
      return 'File size must be less than 16MB'
    }

    return null
  }

  const processFile = async (file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    onLoadingStart()

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    // Simulate API call (replace with actual backend integration)
    try {
      // This would be your actual API call to the Flask backend
      // For demo purposes, we'll simulate a response
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      const mockResult: PredictionResult = {
        class: 'Glioma',
        confidence: 0.87,
        classes: [
          { label: 'Glioma', percent: 87.3 },
          { label: 'Meningioma', percent: 8.2 },
          { label: 'No Tumor', percent: 3.1 },
          { label: 'Pituitary', percent: 1.4 },
          { label: 'Not MRI', percent: 0.0 }
        ],
        image: selectedImage || undefined
      }
      
      onPrediction(mockResult)
    } catch (err) {
      setError('Failed to process image. Please try again.')
      console.error('Prediction error:', err)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const clearImage = () => {
    setSelectedImage(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const tryRandomSample = async () => {
    setError(null)
    onLoadingStart()
    
    try {
      // Simulate random sample API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const mockResult: PredictionResult = {
        class: 'Meningioma',
        confidence: 0.92,
        classes: [
          { label: 'Meningioma', percent: 92.1 },
          { label: 'Glioma', percent: 4.8 },
          { label: 'No Tumor', percent: 2.3 },
          { label: 'Pituitary', percent: 0.8 },
          { label: 'Not MRI', percent: 0.0 }
        ],
        image: '/api/placeholder/400/400' // This would be the base64 image from backend
      }
      
      setSelectedImage('/api/placeholder/400/400')
      onPrediction(mockResult)
    } catch (err) {
      setError('Failed to fetch random sample. Please try again.')
      console.error('Random sample error:', err)
    }
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        } ${isLoading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept="image/*"
          onChange={handleChange}
          disabled={isLoading}
        />

        {isLoading ? (
          <div className="flex flex-col items-center space-y-4">
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
            <div>
              <p className="text-lg font-medium text-gray-900">Processing Image</p>
              <p className="text-sm text-gray-600">
                Analyzing MRI scan<span className="loading-dots"></span>
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-center">
              <Upload className="w-12 h-12 text-gray-400" />
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900">
                Drop your MRI scan here, or click to browse
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Supports JPEG, PNG, BMP up to 16MB
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Selected Image Preview */}
      {selectedImage && !isLoading && (
        <div className="relative">
          <div className="relative bg-gray-100 rounded-lg p-4">
            <button
              onClick={clearImage}
              className="absolute top-2 right-2 p-1 bg-white rounded-full shadow-md hover:bg-gray-50 transition-colors"
            >
              <X className="w-4 h-4 text-gray-600" />
            </button>
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <ImageIcon className="w-8 h-8 text-gray-600" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  MRI scan uploaded successfully
                </p>
                <p className="text-sm text-gray-600">Ready for analysis</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Try Random Sample Button */}
      <div className="text-center">
        <button
          onClick={tryRandomSample}
          disabled={isLoading}
          className="text-primary-600 hover:text-primary-700 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Try a random sample from dataset
        </button>
      </div>
    </div>
  )
}