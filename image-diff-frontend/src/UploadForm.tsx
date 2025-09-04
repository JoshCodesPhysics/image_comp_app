import './App.css';
import { useState } from 'react';
import { useDropzone } from 'react-dropzone';

type UploadFormSets = {
  onSubmit: (beforeFile: File, afterFile: File) => void;
};

export default function UploadForm({ onSubmit }: UploadFormSets) {
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);

  const beforeDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setBeforeFile(acceptedFiles[0]);
    }
  };

  const afterDrop = (acceptedFiles: File[]) => {
    setAfterFile(acceptedFiles[0]);
  };

  const handleSubmit = () => {
    if (!beforeFile || !afterFile) {
      alert('Please upload both before and after images');
      return;
    }

    onSubmit(beforeFile, afterFile);
  };

  // Hook (function with persistent 'use' variable states called at top level of component or other hooks, that can trigger a re-render) from react-dropzone
  // Inputs indicate to call beforeDrop (set the before file) when a file is dropped (image only accepted)
  // Returns the properties (members of an object) getRootProps:
  // used to handle the drop, drag, click events for the before image,and getInputProps: used to handle the input type, number of inputs, file selection, display style
  const { getRootProps: getBeforeRoot, getInputProps: getBeforeInput } =
    useDropzone({ onDrop: beforeDrop, accept: { 'image/*': [] } });

  const { getRootProps: getAfterRoot, getInputProps: getAfterInput } =
    useDropzone({ onDrop: afterDrop, accept: { 'image/*': [] } });

  return (
    // Tailwind CSS div element with 1.5 rem botom margin, flexible container for child element arrangement, stack elements vertically, 1 rem (16px) gap between elements
    <div className="mb-6 flex flex-col items-center gap-4">
      {/* Before Image Dropzone */}
      <div
        {...getBeforeRoot()}
        // Tailwind CSS div element with 16 rem width, 8 rem height, 2px dashed border, gray-400 color, rounded corners, flex container for alignment, center children horizontally and vertically, cursor pointer, background color white, hover effect changes border color to blue-500
        className="w-64 h-32 border-2 border-dashed border-gray-400 rounded-lg flex items-center justify-center cursor-pointer bg-white hover:border-blue-500"
      >
        <input {...getBeforeInput()} />
        {beforeFile ? (
          // Tailwind CSS p element with 0.875 rem font size, gray-700 color
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
        // Tailwind CSS button element with blue-500 background, white text, LR 1.5rem padding, UD 0.5rem padding, rounded corners, shadow, hover effect changes background color to blue-600
        className="bg-blue-500 text-white px-6 py-2 rounded-lg shadow hover:bg-blue-600"
      >
        Submit
      </button>
    </div>
  );
}
