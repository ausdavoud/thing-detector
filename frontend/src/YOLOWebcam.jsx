import React, { useEffect, useRef } from "react";

export default function YOLOWebcam({ onDetections, onSnapshot }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  let ws = null;

  useEffect(() => {
    const wsScheme = import.meta.env.VITE_WS_SCHEME || "wss";
    const wsHost = import.meta.env.VITE_WS_HOST || "localhost:8000";
    const wsUrl = `${wsScheme}://${wsHost}/ws/yolo/`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "detections") {
        const dets = data.detections || [];
        onDetections(dets);
        drawDetections(dets);
        if (dets.length > 0) saveSnapshot();
      }
    };

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setInterval(() => sendFrame(ws), 150);
    });
  }, []);

  function sendFrame(ws) {
    if (!videoRef.current || ws.readyState !== WebSocket.OPEN) return;
    const v = videoRef.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0);

    c.toBlob((blob) => {
      blob.arrayBuffer().then((buf) => ws.send(buf));
    }, "image/jpeg");
  }

  function drawDetections(dets) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";

    dets.forEach((d) => {
      ctx.strokeStyle = "red";
      ctx.strokeRect(d.xmin, d.ymin, d.xmax - d.xmin, d.ymax - d.ymin);
      ctx.fillStyle = "red";
      ctx.fillText(d.cls, d.xmin, d.ymin - 4);
    });
  }

  function saveSnapshot() {
    const v = videoRef.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0);
    onSnapshot(c.toDataURL("image/jpeg"));
  }

  return (
    <div style={{ position: "relative", width: "100%"}}>
      <video ref={videoRef} style={{ width: "100%" }} />
      <canvas
        ref={canvasRef}
        style={{ width: "100%", position: "absolute", top: 0, left: 0 }}
      />
    </div>
  );
}
