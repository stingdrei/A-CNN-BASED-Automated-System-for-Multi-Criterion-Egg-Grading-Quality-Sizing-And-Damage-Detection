import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Camera, RefreshCw } from 'lucide-react';
import { Button } from '../common/Button';

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  busy?: boolean;
}

const CAMERA_ERRORS: Record<string, string> = {
  NotAllowedError: 'Camera permission denied. Allow camera access and try again.',
  NotFoundError: 'No camera found. Connect a camera or use file upload.',
  NotReadableError: 'Camera is busy or unavailable. Close other camera apps and retry.',
  OverconstrainedError: 'Camera does not support the requested resolution.',
  SecurityError: 'Camera access requires HTTPS or localhost.',
};

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException && CAMERA_ERRORS[error.name]) {
    return CAMERA_ERRORS[error.name];
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return 'Could not start the camera. Use file upload instead.';
}

export function CameraCapture({ onCapture, busy = false }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [starting, setStarting] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setReady(false);
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    setError(null);
    setStarting(true);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera API unavailable. Use HTTPS/localhost or file upload.');
      setStarting(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setReady(true);
      }
    } catch (err) {
      stopCamera();
      setError(cameraErrorMessage(err));
    } finally {
      setStarting(false);
    }
  }, [stopCamera]);

  useEffect(() => {
    void startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  const handleCapture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !ready || busy || capturing) return;

    setCapturing(true);
    setError(null);
    try {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) {
        throw new Error('Camera frame is not ready yet. Try again.');
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Could not capture a camera frame.');
      }
      context.drawImage(video, 0, 0, width, height);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.92);
      });
      if (!blob) {
        throw new Error('Could not encode the camera frame.');
      }

      onCapture(
        new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' }),
      );
    } catch (err) {
      setError(cameraErrorMessage(err));
    } finally {
      setCapturing(false);
    }
  }, [busy, capturing, onCapture, ready]);

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-black">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="h-full w-full object-contain"
        />

        {(!ready || starting) && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm text-white">
            {starting ? 'Starting camera...' : 'Waiting for camera...'}
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80 px-6 text-center">
            <AlertTriangle className="h-10 w-10 text-amber-400" />
            <p className="max-w-sm text-sm text-white">{error}</p>
            <Button type="button" variant="outline" onClick={() => void startCamera()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry Camera
            </Button>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-gray-500">
          Camera requires HTTPS or localhost. Capture one frame per tray.
        </p>
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => void startCamera()}
            disabled={starting || busy}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Restart
          </Button>
          <Button
            type="button"
            onClick={() => void handleCapture()}
            disabled={!ready || busy || capturing}
            isLoading={capturing}
          >
            <Camera className="mr-2 h-4 w-4" />
            {busy ? 'Analyzing...' : 'Capture & Analyze'}
          </Button>
        </div>
      </div>
    </div>
  );
}
