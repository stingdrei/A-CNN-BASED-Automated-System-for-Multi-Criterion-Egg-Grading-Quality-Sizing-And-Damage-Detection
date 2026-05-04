import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { DropZone } from '../components/upload/DropZone';
import { Button } from '../components/common/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { apiClient } from '../api/client';
import { getErrorMessage } from '../utils/errorHandler';

export function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.75);
  const [saveAnnotated, setSaveAnnotated] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const result = await apiClient.uploadFile(file, confidence, saveAnnotated);
      navigate(`/result/${result.id}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
    } finally {
      setUploading(false);
    }
  }, [file, confidence, saveAnnotated, navigate]);

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Upload for Prediction</h1>

      <Card>
        <CardHeader>
          <CardTitle>Select File</CardTitle>
        </CardHeader>
        <CardContent>
          <DropZone onFileSelect={setFile} />
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

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-700">Save Annotated Image</p>
              <p className="text-xs text-gray-500">Draw bounding boxes on output</p>
            </div>
            <button
              onClick={() => setSaveAnnotated(!saveAnnotated)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                saveAnnotated ? 'bg-primary-600' : 'bg-gray-200'
              }`}
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

          <Button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="w-full"
            size="lg"
            isLoading={uploading}
          >
            {uploading ? 'Processing...' : 'Start Prediction'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}