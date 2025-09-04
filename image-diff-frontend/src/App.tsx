import { useState, useEffect } from 'react';
import './App.css';
import type { Comparison } from './types';
import UploadForm from './UploadForm';
import ComparisonViewer from './ComparisonViewer';
import HistoryList from './HistoryList';

// Storing base64 previews of image instead of URLS to survive refresh
const fileToBase64 = (file: File): Promise<string> => {
  // Converts a File object to a base64-encoded data URL string using the FileReader API.
  // Returns a Promise that resolves with the base64 string when the file is successfully read,
  // or rejects if an error occurs during reading.
  // - Promise: Represents the eventual completion (or failure) of an asynchronous operation.
  // - resolve: Called when the file is read successfully, passing the result (base64 string).
  // - reject: Called if an error occurs while reading the file.
  // - FileReader: A web API that asynchronously reads the contents of files (such as images).
  // - reader.onload: Event handler triggered when the file has been read successfully.
  // - reader.onerror: Event handler triggered if an error occurs during the read operation.
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export default function App() {
  const [currentComparison, setCurrentComparison] = useState<Comparison | null>(
    null
  );
  const [history, setHistory] = useState<Comparison[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [sliderValue, setSliderValue] = useState(5);

  // Load comparison history from localStorage
  // useEffect takes a function and a dependency array (like a lambda in python)
  // Runs once when the component is mounted since dependency array is empty
  useEffect(() => {
    // Ping the local browser API
    const stored = localStorage.getItem('imageDiffHistory');
    if (stored) {
      setHistory(JSON.parse(stored));
    }
  }, []);

  // Save comparison history to localStorage
  // Runs when the history state changes (when a new comparison is added)
  useEffect(() => {
    localStorage.setItem('imageDiffHistory', JSON.stringify(history));
  }, [history]);

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('imageDiffHistory');
  };

  const handleUpload = async (beforeFile: File, afterFile: File) => {
    setLoading(true);
    setError(null);

    try {
      // POST request - FormData mimics HTML form submission, holds key-value pairs, handles files, designed for HTTP requests
      const formData = new FormData();
      formData.append('file1', beforeFile);
      formData.append('file2', afterFile);

      // Post to comparison endpoint
      const postRes = await fetch('http://localhost:8000/comparison', {
        method: 'POST',
        // Can be uploaded, but a regular stringified object containing files can't be
        body: formData,
      });

      if (!postRes.ok) {
        throw new Error(`POST failed: ${postRes.status}`);
      }

      // Obtain JSON with all of the ComparisonResponse data (uuid, timestamp, before/after urls, scores, diff images)
      const postData = await postRes.json();

      // Immediately GET to confirm persistence
      const getRes = await fetch(
        `http://localhost:8000/comparison/${postData.id}`
      );
      if (!getRes.ok) {
        throw new Error(`GET failed: ${getRes.status}`);
      }

      // Inside handleUpload, after GET response:
      const getData = await getRes.json();

      const beforeBase64 = await fileToBase64(beforeFile);
      const afterBase64 = await fileToBase64(afterFile);

      const backend_url = 'http://localhost:8000';

      const newComparison: Comparison = {
        id: getData.id,
        createdAt: getData.created_at,
        beforeUrl: beforeBase64, // base64 instead of blob
        afterUrl: afterBase64,
        scores: {
          correlation: getData.correlation,
          chiSquare: getData.chi_square,
          bhattacharyya: getData.bhattacharyya,
        },
        // Object.entries(getData.diff_image_urls) returns an array of [key, value] pairs from the diff_image_urls object.
        // Object.fromEntries(...) reconstructs an object from these [key, value] pairs after mapping.
        diffImages: Object.fromEntries(
          Object.entries(getData.diff_image_urls).map(([k, v]) => [
            Number(k),
            backend_url + v,
          ])
        ),
      };

      // Update state
      setCurrentComparison(newComparison);
      setHistory((prev) => [newComparison, ...prev].slice(0, 10)); // keep max 10 items
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    // First div: Minimum height (100% of viewport height), gray background, 1.5 rem padding

    // Second: Tailwind CSS h1 element with 2xl font size, bold, 1.5 rem bottom margin, centred
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-6 text-center">
        Image Comparison Tool
      </h1>

      <UploadForm onSubmit={handleUpload} />

      <HistoryList
        history={history}
        onSelect={setCurrentComparison}
        onClear={handleClearHistory}
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
  );
}
