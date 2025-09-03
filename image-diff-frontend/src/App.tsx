import { useState, useEffect } from 'react'
import './App.css'
import type { Comparison } from './types'
import UploadForm from './UploadForm'
import ComparisonViewer from './ComparisonViewer'
import HistoryList from './HistoryList'

export default function App() {
  const [currentComparison, setCurrentComparison] = useState<Comparison | null>(null)
  const [history, setHistory] = useState<Comparison[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [sliderValue, setSliderValue] = useState(5)

  // Load comparison history from localStorage
  // useEffect takes a function and a dependency array (like a lambda in python)
  // Runs once when the component is mounted since dependency array is empty
  useEffect(() => {
    // Ping the local browser API
    const stored = localStorage.getItem('imageDiffHistory')
    if (stored) {
      setHistory(JSON.parse(stored))
    }
  }, [])

  // Save comparison history to localStorage
  // Runs when the history state changes (when a new comparison is added)
  useEffect(() => {
    localStorage.setItem('imageDiffHistory', JSON.stringify(history))
  }, [history])

  const handleUpload = async (beforeFile: File, afterFile: File) => {
    setLoading(true)
    setError(null)

    try {
      // POST request - FormData mimics HTML form submission, holds key-value pairs, handles files, designed for HTTP requests
      const formData = new FormData()
      formData.append("file1", beforeFile)
      formData.append("file2", afterFile)

      // Post to comparison endpoint
      const postRes = await fetch("http://localhost:8000/comparison", {
        method: "POST",
        // Can be uploaded, but a regular stringified object containing files can't be
        body: formData,
      })

      if (!postRes.ok) {
        throw new Error(`POST failed: ${postRes.status}`)
      }

      // Obtain JSON with all of the ComparisonResponse data (uuid, timestamp, before/after urls, scores, diff images)
      const postData = await postRes.json()

      // Immediately GET to confirm persistence
      const getRes = await fetch(`http://localhost:8000/comparison/${postData.id}`)
      if (!getRes.ok) {
        throw new Error(`GET failed: ${getRes.status}`)
      }

      const getData = await getRes.json()


      const backend_url = "http://localhost:8000"

      const newComparison: Comparison = {
        id: getData.id,
        createdAt: getData.created_at,
        beforeUrl: URL.createObjectURL(beforeFile),
        afterUrl: URL.createObjectURL(afterFile),
        scores: {
          correlation: getData.correlation,
          chiSquare: getData.chi_square,
          bhattacharyya: getData.bhattacharyya,
        },
        diffImages: Object.fromEntries(
          Object.entries(getData.diff_image_urls).map(([k, v]) => [Number(k), backend_url + v])
        ),
      }

      // Update state
      setCurrentComparison(newComparison)
      setHistory(prev => [newComparison, ...prev].slice(0, 10)) // keep max 10 items

    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    // First div: Minimum height (100% of viewport height), gray background, 1.5 rem padding

    // Second: Tailwind CSS h1 element with 2xl font size, bold, 1.5 rem bottom margin, centred 
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-6 text-center">Image Comparison Tool</h1>

      <UploadForm onSubmit={handleUpload} />

      <HistoryList
        history={history}
        onSelect={setCurrentComparison}
      />

      {loading && <p className="text-center text-blue-500">Processing...</p>}
      {error && <p className="text-center text-red-500">{error}</p>}

      {currentComparison && (
        <ComparisonViewer
          comparison={currentComparison}
          sliderValue={sliderValue}
          setSliderValue={setSliderValue}
        />
      )}
    </div>
  )
}
