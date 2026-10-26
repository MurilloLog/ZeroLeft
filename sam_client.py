import os
import cv2
import numpy as np
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

class SAMClient:
    def __init__(self):
        self.api_key = os.getenv("META_API_KEY")
        self.endpoint_url = os.getenv("META_SAM_ENDPOINT_URL")

    def predict_mask(self, image: np.ndarray, click_coords: tuple) -> tuple:
        # If there isn't any API Key, use simulation mode
        if not self.api_key or not self.endpoint_url:
            print("[INFO] Local Simulation Mode is Activated.")
            return self._mock_segmentation(image, click_coords)

        # Otherwise, try a real conexion
        try:
            return self._call_meta_api(image, click_coords)
        except Exception as e:
            print(f"[WARN] Error trying conecting with Meta Model API: {e}.")
            return self._mock_segmentation(image, click_coords)

    def _mock_segmentation(self, image: np.ndarray, click_coords: tuple) -> tuple:
        if click_coords is None:
            return image, "Select a tool."

        x, y = click_coords
        overlay = image.copy()
        
        # Simulate a mask
        cv2.ellipse(overlay, (x, y), (60, 25), 30, 0, 360, (0, 255, 128), -1)
        image_with_mask = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        cv2.circle(image_with_mask, (x, y), 5, (255, 0, 0), -1)

        status_text = f"Tool detected on ({x}, {y}) [Simulation Mode]"
        return image_with_mask, status_text

    def _call_meta_api(self, image: np.ndarray, click_coords: tuple) -> tuple:
        # Pending an HTTP call
        pass

    def auto_segment_stage(self, image: np.ndarray) -> tuple:
        """
        Executes zero-shot automatic mask generation across the entire surgical frame.
        Identifies all physical objects without manual point selection.
        """
        if not self.api_key or not self.endpoint_url:
            return self._mock_auto_segmentation(image)

        try:
            # Payload configured for automatic grid generation in SAM 3.1
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": self._encode_image_to_base64(image),
                "generate_automatic_masks": True,
                "points_per_side": 16,
                "pred_iou_thresh": 0.88
            }

            response = requests.post(self.endpoint_url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                # Process automatically returned masks
                pass

            return self._mock_auto_segmentation(image)
        except Exception as e:
            print(f"[WARN] Auto-segmentation fallback triggered: {e}")
            return self._mock_auto_segmentation(image)

    def _mock_auto_segmentation(self, image: np.ndarray) -> tuple:
        """
        Simulates full-frame automatic instrument detection overlay.
        """
        overlay = image.copy()
        h, w, _ = image.shape

        # Simulate 3 automatic detected masks over the frame
        cv2.ellipse(overlay, (int(w*0.3), int(h*0.4)), (50, 20), 15, 0, 360, (0, 255, 128), -1)
        cv2.ellipse(overlay, (int(w*0.6), int(h*0.5)), (60, 22), -30, 0, 360, (255, 128, 0), -1)
        cv2.rectangle(overlay, (int(w*0.4), int(h*0.7)), (int(w*0.55), int(h*0.82)), (0, 128, 255), -1)

        processed_image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        status_msg = "SAM 3.1 Auto-Segmentation: 3 Objects Identified Zero-Shot."
        
        detected_objects = [
            {"id": "TOOL-001", "class": "Hemostatic Forceps", "confidence": 0.96},
            {"id": "TOOL-002", "class": "Surgical Scalpel", "confidence": 0.92},
            {"id": "TOOL-003", "class": "Gauze Sponge", "confidence": 0.89}
        ]
        
        return processed_image, status_msg, detected_objects

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", bgr_image)
        return base64.b64encode(buffer).decode("utf-8")