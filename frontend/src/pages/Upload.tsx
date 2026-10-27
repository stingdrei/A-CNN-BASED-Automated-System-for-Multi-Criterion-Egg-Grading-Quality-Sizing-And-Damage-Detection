import { useState, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Camera, Upload as UploadIcon } from 'lucide-react';
import { DropZone } from '../components/upload/DropZone';
import { CameraCapture } from '../components/upload/CameraCapture';
import { Button } from '../components/common/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { apiClient } from '../api/client';

type CaptureMode = 'camera' | 'upload';

export function UploadPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const mode: CaptureMode =
    searchParams.get('mode') === 'camera' ? 'camera' : 'upload';
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.75);
  const [saveAnnotated, setSaveAnnotated] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const switchMode = useCallback(
    (nextMode: CaptureMode) => {
      setFile(null);
      setError(null);
      setSearchParams(nextMode === 'camera' ? { mode: 'camera' } : {});
    },
    [setSearchParams],
  );

  const submitFile = useCallback(
    async (selectedFile: File) => {
      setUploading(true);
      setError(null);
      try {
        const result = await apiClient.uploadFile(
          selectedFile,
          confidence,
          saveAnnotated,
        );
        navigate(`/result/${result.id}`);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Upload failed';
        setError(message);
      } finally {
        setUploading(false);
      }
    },
    [confidence, navigate, saveAnnotated],
  );

  const handleUpload = useCallback(() => {
    if (file) {
      void submitFile(file);
    }
  }, [file, submitFile]);

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-2xl font-bold text-gray-900">
          {mode === 'camera' ? 'Capture Tray' : 'Upload for Prediction'}
        </h1>
        <div className="inline-flex self-start rounded-lg bg-gray-100 p-1">
          <button
            type="button"
            onClick={() => switchMode('camera')}
            className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              mode === 'camera'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            aria-pressed={mode === 'camera'}
          >
            <Camera className="h-4 w-4" />
            Camera
          </button>
          <button
            type="button"
            onClick={() => switchMode('upload')}
            className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              mode === 'upload'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            aria-pressed={mode === 'upload'}
          >
            <UploadIcon className="h-4 w-4" />
            File
          </button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{mode === 'camera' ? 'Camera' : 'Select File'}</CardTitle>
        </CardHeader>
        <CardContent>
          {mode === 'camera' ? (
            <CameraCapture onCapture={(captured) => void submitFile(captured)} busy={uploading} />
          ) : (
            <DropZone onFileSelect={setFile} />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">Confidence Threshold</label>
              <span className="text-sm text-gray-500">{Math.round(confidence * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.99"
              step="0.01"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>More detections</span>
              <span>Higher confidence</span>
            </div>
          </div>

          <div className="p-3 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg">
            Beta: cleanliness grading is not configured. Final grades will be N/A;
            provisional grades use damage and size only.
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-700">Save Annotated Image</p>
              <p className="text-xs text-gray-500">Draw bounding boxes on output</p>
            </div>
            <button
              type="button"
              onClick={() => setSaveAnnotated(!saveAnnotated)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                saveAnnotated ? 'bg-primary-600' : 'bg-gray-200'
              }`}
              aria-pressed={saveAnnotated}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  saveAnnotated ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {error && (
            <div className="p-3 text-sm text-red-600 bg-red-50 rounded-lg">
              {error}
            </div>
          )}

          {mode === 'upload' && (
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="w-full"
              size="lg"
              isLoading={uploading}
            >
              {uploading ? 'Processing...' : 'Start Prediction'}
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
