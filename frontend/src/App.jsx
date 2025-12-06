import React, { useState, useMemo } from "react";
import YOLOWebcam from "./YOLOWebcam";
import "./App.css";

export default function App() {
  const [detections, setDetections] = useState([]);
  const [snapshots, setSnapshots] = useState([]);

  // Count total and count per object class
  const classCounts = useMemo(() => {
    const c = {};
    detections.forEach((d) => {
      c[d.cls] = (c[d.cls] || 0) + 1;
    });
    return c;
  }, [detections]);

  return (
    <div className="meet-container">
      {/* LEFT SIDE — Video */}
      <div className="video-section">
        <YOLOWebcam
          onDetections={setDetections}
          onSnapshot={(img) => setSnapshots((prev) => [img, ...prev])}
        />
      </div>

      {/* RIGHT SIDE — Object List (Like Google Meet Participants) */}
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

        {/* Bottom total like Google Meet person count */}
        <div className="total-footer">
          Total Objects: {detections.length}
        </div>
      </div>
    </div>
  );
}
