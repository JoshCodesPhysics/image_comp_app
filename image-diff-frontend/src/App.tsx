import { useState } from 'react'
import './App.css'
import type { Comparison } from './types'
import UploadForm from './UploadForm'

export default function App() {
  const [currentComparison, setCurrentComparison] = useState<Comparison | null>(null)
  const [history, setHistory] = useState<Comparison[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = (beforeFile: File, afterFile: File) => {
    console.log('Files submitted:', { beforeFile, afterFile })
    setLoading(true)
    setError(null)
    
    // TODO: Implement actual API call here
    // For now, just simulate processing
    setTimeout(() => {
      setLoading(false)
      console.log('Processing complete')
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto py-8">
        <UploadForm onSubmit={handleSubmit} />
        
        {loading && (
          <div className="mt-8 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Processing images...</p>
          </div>
        )}
        
        {error && (
          <div className="mt-8 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            Error: {error}
          </div>
        )}
      </div>
    </div>
  )
}
