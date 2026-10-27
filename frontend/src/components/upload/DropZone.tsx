import { useCallback, useState } from 'react';
import { Upload as UploadIcon, FileImage, X } from 'lucide-react';
import { clsx } from 'clsx';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSizeMB?: number;
}

export function DropZone({ onFileSelect, accept = 'image/*', maxSizeMB = 50 }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validateFile = useCallback((file: File) => {
    setError(null);
    const maxBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`File size must be less than ${maxSizeMB}MB`);
      return false;
    }
    if (!file.type.startsWith('image/')) {
      setError('Only image files are supported');
      return false;
    }
    return true;
  }, [maxSizeMB]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [validateFile, onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [validateFile, onFileSelect]);

  const removeFile = useCallback(() => {
    setSelectedFile(null);
    setError(null);
  }, []);

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return FileImage;
    return UploadIcon;
  };

  if (selectedFile) {
    const FileIcon = getFileIcon(selectedFile.type);
    return (
      <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <FileIcon className="w-8 h-8 text-gray-400" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">{selectedFile.name}</p>
          <p className="text-xs text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
        <button
          onClick={removeFile}
          className="p-1 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-200"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
    );
  }

  return (
    <div>
      <label
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={clsx(
          'flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-colors',
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
        )}
      >
        <input
          type="file"
          className="hidden"
          accept={accept}
          onChange={handleFileInput}
        />
        <UploadIcon className={clsx('w-10 h-10 mb-3', isDragging ? 'text-primary-500' : 'text-gray-400')} />
        <p className="text-sm text-gray-600">
          <span className="font-medium text-primary-600">Click to upload</span> or drag and drop
        </p>
        <p className="text-xs text-gray-500 mt-1">JPG, PNG, WEBP, BMP (max {maxSizeMB}MB)</p>
      </label>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}