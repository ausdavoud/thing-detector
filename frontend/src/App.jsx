import { useState, useMemo } from "react";
import YOLOWebcam from "./YOLOWebcam";
import "./App.css";

export default function App() {
  const [detections, setDetections] = useState([]);

  const classCounts = useMemo(() => {
    const c = {};
    detections.forEach((d) => {
      c[d.cls] = (c[d.cls] || 0) + 1;
    });
    return c;
  }, [detections]);

  return (
    <div className="meet-container">
      <div className="video-section">
        <YOLOWebcam onDetections={setDetections}/>
      </div>

      <div className="sidebar">
        <h2>Detected Objects</h2>

        <div className="object-list">
          {Object.keys(classCounts).length === 0 && (
            <p className="no-objects">No objects detected</p>
          )}

          {Object.entries(classCounts).map(([cls, count]) => (
            <div key={cls} className="object-item">
              <span className="object-name">{cls}</span>
              <span className="object-count">{count}</span>
            </div>
          ))}
        </div>

        <div className="total-footer">
          Total Objects: {detections.length}
        </div>
      </div>
    </div>
  );
}
