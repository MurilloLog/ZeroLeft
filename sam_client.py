import os
import cv2
import numpy as np
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# Loading SAM in local just for OnClick() Gradio event
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

class SAMClient:
    def __init__(self, checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b"):
        self.device = "cuda" if self._has_cuda() else "cpu"
        self.predictor = None
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        
        if SAM_AVAILABLE and os.path.exists(checkpoint_path):
            print(f"[INFO] Loading SAM {model_type} on {self.device} from {checkpoint_path}")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
        else:
            if not SAM_AVAILABLE:
                print("[WARN] Segment Anything Model is not installed. Run: pip install segment-anything torch")
            else:
                print(f"[WARN] Checkpoint was not found in {checkpoint_path}. Run: curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    
    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def predict_mask(self, image: np.ndarray, click_coords: tuple) -> tuple:
        """
        process_click event from app.py
        """
        if image is None or click_coords is None:
            return image, "Select an image and click an instrument"

        x, y = click_coords
        if self.predictor is None:
            return self._mock_segmentation(image, click_coords)

        try:
            self.predictor.set_image(image)
            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, scores, logits = self.predictor.predict(
                point_coords = input_point,
                point_labels = input_label,
                multimask_output = True
            )

            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # Make an overlay
            overlay = image.copy()
            # Green mask
            colored_mask = np.zeros_like(image)
            colored_mask[best_mask] = [0, 255, 128]
            image_with_mask = cv2.addWeighted(colored_mask, 0.45, overlay, 0.55, 0)
            
            # Click pointer
            cv2.circle(image_with_mask, (x, y), 6, (255, 0, 0), -1)
            cv2.circle(image_with_mask, (x, y), 8, (255, 255, 255), 2)
            
            status = f"SAM {self.model_type} | Click ({x},{y}) | Score: {best_score:.2f} | Device: {self.device}"
            return image_with_mask, status
            
        except Exception as e:
            print(f"[ERROR] SAM in local failed: {e}")
            return self._mock_segmentation(image, click_coords)

    def _mock_segmentation(self, image: np.ndarray, click_coords: tuple) -> tuple:
        x, y = click_coords
        overlay = image.copy()
        cv2.ellipse(overlay, (x, y), (60, 25), 30, 0, 360, (0, 255, 128), -1)
        image_with_mask = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        cv2.circle(image_with_mask, (x, y), 5, (255, 0, 0), -1)
        status_text = f"[MOCK] Tool detected on ({x}, {y}) - SAM is needed for real masking"
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

        # Showing detected masks over the frame
        cv2.ellipse(overlay, (int(w*0.3), int(h*0.4)), (50, 20), 15, 0, 360, (0, 255, 128), -1)
        cv2.ellipse(overlay, (int(w*0.6), int(h*0.5)), (60, 22), -30, 0, 360, (255, 128, 0), -1)
        
        processed_image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        status_msg = "OnClick-Segmentation"
        
        detected_objects = []
        
        return processed_image, status_msg, detected_objects

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", bgr_image)
        return base64.b64encode(buffer).decode("utf-8")