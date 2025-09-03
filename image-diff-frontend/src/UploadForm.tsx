import './App.css'
import { useState } from 'react'
import { useDropzone } from "react-dropzone"

type UploadFormSets = {
    onSubmit : (beforeFile: File, afterFile: File) => void
}

export default function UploadForm({ onSubmit }: UploadFormSets) {
    const [beforeFile, setBeforeFile] = useState<File | null>(null)
    const [afterFile, setAfterFile] = useState<File | null>(null)

    const beforeDrop = (acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            setBeforeFile(acceptedFiles[0])
        }
    }

    const afterDrop = (acceptedFiles: File[]) => {
        setAfterFile(acceptedFiles[0])
    }

    const handleSubmit = () => {
        if (!beforeFile || !afterFile) {
            alert("Please upload both before and after images")
            return
        }

        onSubmit(beforeFile, afterFile)
    }

    const {
        getRootProps: getBeforeRoot,
        getInputProps: getBeforeInput
    } = useDropzone({ onDrop: beforeDrop, accept: { "image/*": [] } })

    const {
        getRootProps: getAfterRoot,
        getInputProps: getAfterInput
    } = useDropzone({ onDrop: afterDrop, accept: { "image/*": [] } })

  return (
    <div className="mb-6 flex flex-col items-center gap-4">
      {/* Before Image Dropzone */}
      <div
        {...getBeforeRoot()}
        className="w-64 h-32 border-2 border-dashed border-gray-400 rounded-lg flex items-center justify-center cursor-pointer bg-white hover:border-blue-500"
      >
        <input {...getBeforeInput()} />
        {beforeFile ? (
          <p className="text-sm text-gray-700">{beforeFile.name}</p>
        ) : (
          <p className="text-gray-500">Drop Before Image here</p>
        )}
      </div>

      {/* After Image Dropzone */}
      <div
        {...getAfterRoot()}
        className="w-64 h-32 border-2 border-dashed border-gray-400 rounded-lg flex items-center justify-center cursor-pointer bg-white hover:border-blue-500"
      >
        <input {...getAfterInput()} />
        {afterFile ? (
          <p className="text-sm text-gray-700">{afterFile.name}</p>
        ) : (
          <p className="text-gray-500">Drop After Image here</p>
        )}
      </div>

             {/* Submit Button */}
       <button
         onClick={handleSubmit}
         disabled={!beforeFile || !afterFile}
         className={`px-6 py-2 rounded-lg shadow transition-colors ${
           beforeFile && afterFile
             ? 'bg-blue-500 text-white hover:bg-blue-600'
             : 'bg-gray-300 text-gray-500 cursor-not-allowed'
         }`}
       >
         Submit
       </button>

       {/* File Information Display */}
       {(beforeFile || afterFile) && (
         <div className="mt-6 w-full max-w-2xl">
           <h3 className="text-lg font-semibold mb-4 text-center">File Information</h3>
           
           <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             {/* Before File Info */}
             {beforeFile && (
               <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                 <h4 className="font-semibold text-green-800 mb-2">Before Image</h4>
                 <div className="space-y-1 text-sm">
                   <p><strong>Name:</strong> {beforeFile.name}</p>
                   <p><strong>Size:</strong> {(beforeFile.size / 1024 / 1024).toFixed(2)} MB</p>
                   <p><strong>Type:</strong> {beforeFile.type}</p>
                   <p><strong>Last Modified:</strong> {new Date(beforeFile.lastModified).toLocaleString()}</p>
                   <p><strong>Size (bytes):</strong> {beforeFile.size.toLocaleString()}</p>
                 </div>
               </div>
             )}

             {/* After File Info */}
             {afterFile && (
               <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                 <h4 className="font-semibold text-blue-800 mb-2">After Image</h4>
                 <div className="space-y-1 text-sm">
                   <p><strong>Name:</strong> {afterFile.name}</p>
                   <p><strong>Size:</strong> {(afterFile.size / 1024 / 1024).toFixed(2)} MB</p>
                   <p><strong>Type:</strong> {afterFile.type}</p>
                   <p><strong>Last Modified:</strong> {new Date(afterFile.lastModified).toLocaleString()}</p>
                   <p><strong>Size (bytes):</strong> {afterFile.size.toLocaleString()}</p>
                 </div>
               </div>
             )}
           </div>

           {/* Comparison Summary */}
           {beforeFile && afterFile && (
             <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
               <h4 className="font-semibold text-gray-800 mb-2">Comparison Summary</h4>
               <div className="grid grid-cols-2 gap-4 text-sm">
                 <div>
                   <p><strong>Total Size:</strong> {((beforeFile.size + afterFile.size) / 1024 / 1024).toFixed(2)} MB</p>
                   <p><strong>Size Difference:</strong> {((afterFile.size - beforeFile.size) / 1024 / 1024).toFixed(2)} MB</p>
                 </div>
                 <div>
                   <p><strong>Same Type:</strong> {beforeFile.type === afterFile.type ? 'Yes' : 'No'}</p>
                   <p><strong>Same Name:</strong> {beforeFile.name === afterFile.name ? 'Yes' : 'No'}</p>
                 </div>
               </div>
             </div>
           )}
         </div>
       )}
     </div>
   )
}