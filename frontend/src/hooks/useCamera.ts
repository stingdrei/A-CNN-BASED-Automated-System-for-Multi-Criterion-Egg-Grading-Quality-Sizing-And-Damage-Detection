import { useCallback, useEffect, useRef, useState } from 'react';
import { CapturedFrame } from '../types';
import { captureVideoFrame } from '../utils/frames';

export type CameraStatus = 'idle' | 'starting' | 'ready' | 'error';

const CAMERA_ERRORS: Record<string, string> = {
  NotAllowedError:
    'Camera permission denied. Allow camera access and try again.',
  NotFoundError: 'No camera found. Connect a camera or upload an image.',
  NotReadableError:
    'Camera is busy or unavailable. Close other camera apps and retry.',
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
  return 'Could not start the camera. Upload an image instead.';
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus('idle');
  }, []);

  const start = useCallback(async () => {
    stop();
    setError(null);
    setStatus('starting');

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera API unavailable. Use HTTPS/localhost or upload an image.');
      setStatus('error');
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
      if (!videoRef.current) {
        throw new Error('Camera preview is not mounted.');
      }
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus('ready');
    } catch (err) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setError(cameraErrorMessage(err));
      setStatus('error');
    }
  }, [stop]);

  const capture = useCallback(async (): Promise<CapturedFrame | null> => {
    const video = videoRef.current;
    if (!video || status !== 'ready') return null;
    try {
      setError(null);
      return await captureVideoFrame(video);
    } catch (err) {
      setError(cameraErrorMessage(err));
      return null;
    }
  }, [status]);

  useEffect(() => stop, [stop]);

  return { videoRef, status, error, start, stop, capture };
}
