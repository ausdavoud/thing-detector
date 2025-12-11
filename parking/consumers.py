import asyncio
import json

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO  # if using ultralytics YOLOv8

# Load model once (global). Consider loading on startup in production.
MODEL = YOLO("yolov8n.pt")  # yolov8n is small and fast; replace as needed


class YOLOConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send_json({"type": "info", "msg": "connected"})
        # Optionally give server model info
        # Note: heavy init should be outside of connect in production.

    async def disconnect(self, close_code):
        # cleanup if needed
        pass

    async def receive(self, bytes_data=None, text_data=None, **kwargs):
        """
        We expect binary frames (JPEG bytes). Channels will pass binary to `bytes_data`.
        """
        if bytes_data:
            # Offload CPU/GPU inference to threadpool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                None, self.process_frame, bytes_data
            )
            # detections is list of dicts
            await self.send_json({"type": "detections", "detections": detections})
        else:
            # ignore text data, or handle control messages
            pass

    def process_frame(self, jpg_bytes):
        # Decode JPEG bytes to BGR image
        nparr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return []
        # Optionally resize to model preferred size for speed:
        # img = cv2.resize(img, (640, 640))

        # Convert BGR->RGB for ultralytics
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        # ultralytics returns results object; we'll parse it
        results = MODEL.predict(
            source=img_rgb, conf=0.35, device="cpu", verbose=False
        )  # device=0 for GPU, or 'cpu'
        # results is a list; results[0] contains boxes, scores, classes
        res0 = results[0]
        boxes = res0.boxes  # ultralytics object
        dets = []
        # boxes.xyxy, boxes.conf, boxes.cls
        if boxes is not None and len(boxes) > 0:
            xyxy_arr = (
                boxes.xyxy.cpu().numpy()
                if hasattr(boxes.xyxy, "cpu")
                else boxes.xyxy.numpy()
            )
            confs = (
                boxes.conf.cpu().numpy()
                if hasattr(boxes.conf, "cpu")
                else boxes.conf.numpy()
            )
            cls_arr = (
                boxes.cls.cpu().numpy()
                if hasattr(boxes.cls, "cpu")
                else boxes.cls.numpy()
            )
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy_arr, confs, cls_arr):
                dets.append(
                    {
                        "xmin": float(x1),
                        "ymin": float(y1),
                        "xmax": float(x2),
                        "ymax": float(y2),
                        "conf": float(conf),
                        "cls": str(res0.names[int(cls_id)])
                        if hasattr(res0, "names")
                        else str(int(cls_id)),
                    }
                )
        return dets

    async def send_json(self, obj):
        # helper to send JSON messages
        await self.send(text_data=json.dumps(obj))
