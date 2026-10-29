import { CapturedFrame, DetectionBox } from '../types';

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Could not read the selected image.'));
    reader.readAsDataURL(file);
  });
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not decode the image.'));
    image.src = src;
  });
}

export async function fileToFrame(file: File): Promise<CapturedFrame> {
  const dataUrl = await readFileAsDataUrl(file);
  const image = await loadImage(dataUrl);
  return {
    file,
    dataUrl,
    width: image.naturalWidth,
    height: image.naturalHeight,
  };
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error('Could not encode the camera frame.'));
      },
      'image/jpeg',
      0.92,
    );
  });
}

export async function captureVideoFrame(
  video: HTMLVideoElement,
): Promise<CapturedFrame> {
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
  const blob = await canvasToBlob(canvas);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const file = new File([blob], `egg-tray-${stamp}.jpg`, {
    type: 'image/jpeg',
  });

  return {
    file,
    dataUrl: canvas.toDataURL('image/jpeg', 0.92),
    width,
    height,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export async function buildCropThumbnails(
  frame: CapturedFrame,
  boxes: DetectionBox[],
): Promise<Record<number, string>> {
  if (boxes.length === 0) return {};

  const image = await loadImage(frame.dataUrl);
  return boxes.reduce<Record<number, string>>((crops, box) => {
    const x = clamp(box.x1, 0, frame.width);
    const y = clamp(box.y1, 0, frame.height);
    const right = clamp(box.x2, 0, frame.width);
    const bottom = clamp(box.y2, 0, frame.height);
    const width = right - x;
    const height = bottom - y;
    if (width < 2 || height < 2) return crops;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) return crops;

    context.drawImage(image, x, y, width, height, 0, 0, width, height);
    crops[box.id] = canvas.toDataURL('image/jpeg', 0.9);
    return crops;
  }, {});
}
