import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NeuroScan - Brain Tumor Classification',
  description: 'AI-powered brain tumor classification with advanced MRI analysis and Grad-CAM visualization',
  keywords: 'brain tumor, MRI, AI, classification, medical imaging, deep learning',
  authors: [{ name: 'Yash Naidu' }],
  openGraph: {
    title: 'NeuroScan - Brain Tumor Classification',
    description: 'AI-powered brain tumor classification with advanced MRI analysis',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}