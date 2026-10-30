import { Maximize2 } from 'lucide-react';
import type { Detection } from '../lib/api';

export default function SmearPreview({
  image,
  detections,
}: {
  image: string;
  detections: Detection[];
}) {
  return (
    <section className="image-panel">
      <div className="panel-head image-head">
        <div>
          <h3>AI detection preview</h3>
          <p>YOLO11n • {detections.length} cells detected</p>
        </div>
      </div>

      <div className="smear-preview-frame">
        {image ? (
          <img
            src={image}
            alt="AI annotated blood smear"
            className="smear-preview-image"
          />
        ) : (
          <div className="smear-preview-empty">
            No annotated image available
          </div>
        )}
      </div>

      <div className="legend-line">
        <span><i className="r" /> RBC</span>
        <span><i className="w" /> WBC</span>
        <span><i className="p" /> Platelet</span>
      </div>
    </section>
  );
}