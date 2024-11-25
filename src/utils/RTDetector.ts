import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export class RTDetector {
  private model: cocoSsd.ObjectDetection | null = null;
  private isScanning: boolean = false;

  constructor() {
    this.initModel();
  }

  private async initModel() {
    try {
      this.model = await cocoSsd.load();
    } catch (error) {
      console.error('Failed to load AI model:', error);
    }
  }

  private async processMapSection(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number) {
    if (!this.model) return [];

    const imageData = ctx.getImageData(x, y, width, height);
    const tensor = tf.browser.fromPixels(imageData);
    
    try {
      const predictions = await this.model.detect(tensor);
      return predictions
        .filter(pred => pred.score > 0.6) // Only high confidence predictions
        .map(pred => ({
          x: x + pred.bbox[0],
          y: y + pred.bbox[1],
          confidence: pred.score,
          type: this.classifyRTType(pred)
        }));
    } finally {
      tensor.dispose();
    }
  }

  private classifyRTType(prediction: cocoSsd.DetectedObject) {
    // Analyze the visual characteristics to determine RT type
    const area = prediction.bbox[2] * prediction.bbox[3];
    const aspectRatio = prediction.bbox[2] / prediction.bbox[3];

    if (area < 500 && aspectRatio > 0.8 && aspectRatio < 1.2) {
      return 'RT Точка';
    } else if (area < 1000 && aspectRatio > 1.2) {
      return 'RT Зона';
    } else {
      return 'RT Неизвестно';
    }
  }

  public async detectRT(mapElement: HTMLElement): Promise<Array<{
    pos: [number, number];
    name: string;
    confidence: number;
  }>> {
    if (this.isScanning || !this.model) return [];

    this.isScanning = true;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return [];

    // Set canvas size to match map
    canvas.width = mapElement.clientWidth;
    canvas.height = mapElement.clientHeight;

    // Draw map to canvas
    ctx.drawImage(mapElement, 0, 0, canvas.width, canvas.height);

    // Process map in sections for better performance
    const sectionSize = 512;
    const rtPoints: Array<{
      pos: [number, number];
      name: string;
      confidence: number;
    }> = [];

    for (let y = 0; y < canvas.height; y += sectionSize) {
      for (let x = 0; x < canvas.width; x += sectionSize) {
        const width = Math.min(sectionSize, canvas.width - x);
        const height = Math.min(sectionSize, canvas.height - y);

        const predictions = await this.processMapSection(ctx, x, y, width, height);
        
        predictions.forEach(pred => {
          // Convert canvas coordinates to map coordinates
          const mapX = (pred.x / canvas.width) * 1000;
          const mapY = (pred.y / canvas.height) * 1000;

          rtPoints.push({
            pos: [mapY, mapX], // Leaflet uses [lat, lng] format
            name: `${pred.type} (${Math.round(pred.confidence * 100)}%)`,
            confidence: pred.confidence
          });
        });

        // Update progress
        const progress = ((y * canvas.width + x) / (canvas.width * canvas.height)) * 100;
        const progressElement = document.getElementById('rtProgress');
        if (progressElement) {
          progressElement.style.width = `${progress}%`;
        }
      }
    }

    this.isScanning = false;
    return rtPoints;
  }
}