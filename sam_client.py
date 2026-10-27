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

# Masking colors for tools classes segmentation
CLASS_COLORS = {
    "Episiotomy scissors": (0, 255, 128),   # green
    "Hemostatic forceps": (255, 128, 0),    # orange
    "Scalpel": (0, 128, 255),               # blue
    "Stitch scissors": (255, 0, 128),       # magenta
    "Gauze / Sponge": (255, 255, 0),        # yellow
    "Retractor": (128, 0, 255),             # purple
    "Syringe": (0, 255, 255),               # cyan
    "Globes": (128, 128, 128),              # gray
}

class SAMClient:
    def __init__(self, checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b"):
        self.device = "cuda" if self._has_cuda() else "cpu"
        self.predictor = None
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.last_image = None
        self_last_image_hash = None

        # Masking history
        self.annotations = [] # {mask, class, color, points, labels, centroid}
        
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

    def _image_changed(self, image):
        # Avoiding a set_image in every click using a simple hash instead
        if self.last_image is None:
            return True
        if image.shape != self.last_image.shape:
            return True
        
        return not np.array_equal(image[::20, ::20], self.last_image[::20, ::20])
    
    def predict_mask(self, image: np.ndarray, click_coords: tuple, tool_type: str = "Instrument") -> tuple:
        """
        process_click event from app.py
        """
        if image is None or click_coords is None:
            return image, "Select an image and click on an instrument"

        x, y = click_coords
        color = CLASS_COLORS.get(tool_type, (0, 255, 128))
        
        if self.predictor is None:
            return self._mock_add(image, click_coords, tool_type)

        if self._image_changed(image):
            self.predictor.set_image(image)
            self.last_image = image.copy()
            self.annotations = [] # If there is a new image, history is reseted
            print("[INFO] New image") 

        should_refine = False
        if len(self.annotations) > 0:
            last = self.annotations[-1]
            if last["class"] == tool_type:
                cx, cy = last["centroid"]
                dist = np.hypot(x - cx, y - cy)
                if dist < 150:  # we are selecting the same tool, it is not a new one
                    should_refine = True

        if should_refine:
            # Adding a new point to the previous tool
            last = self.annotations[-1]
            last["points"].append([x, y])
            last["labels"].append(1)
            points = np.array(last["points"])
            labels = np.array(last["labels"])
            print(f"[INFO] Segmentation of {tool_type} improved with {len(points)} points")
        else:
            points = np.array([[x, y]])
            labels = np.array([1])

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        # Labeling the tool segmentation using the center of the mask
        ys, xs = np.where(best_mask)
        if len(xs) > 0:
            centroid = (int(xs.mean()), int(ys.mean()))
        else:
            centroid = (x, y)

        if should_refine:
            # Updating current mask
            self.annotations[-1]["mask"] = best_mask
            self.annotations[-1]["centroid"] = centroid
            self.annotations[-1]["score"] = float(best_score)
        else:
            # New annotation
            self.annotations.append({
                "mask": best_mask,
                "class": tool_type,
                "color": color,
                "points": points.tolist() if isinstance(points, np.ndarray) else [[x,y]],
                "labels": labels.tolist(),
                "centroid": centroid,
                "score": float(best_score)
            })

        # Render tool segmentation by clase
        composite, status = self._render_composite(image)
        return composite, status

    def _render_composite(self, base_image):
        """
        1. Keep previous mask using the color class scheme
        2. Every tool is labeled
        """
        overlay = base_image.copy()
        # Mask layer base
        mask_layer = np.zeros_like(base_image)

        for ann in self.annotations:
            color = ann["color"]
            mask = ann["mask"]
            mask_layer[mask] = color

        # Blend
        composite = cv2.addWeighted(mask_layer, 0.45, overlay, 0.55, 0)

        # Drawing labels and clicks
        for idx, ann in enumerate(self.annotations):
            cx, cy = ann["centroid"]
            color = ann["color"]
            label = ann["class"]
            # Visual border
            cv2.putText(composite, f"{label}", (cx+10, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(composite, f"{label}", (cx+10, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            # Central point (x,y)
            cv2.circle(composite, (cx, cy), 4, color, -1)
            cv2.circle(composite, (cx, cy), 6, (255,255,255), 1)
            # Writing the tool ID
            cv2.putText(composite, f"#{idx+1}", (cx+10, cy+8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # Status
        counts = {}
        for ann in self.annotations:
            counts[ann["class"]] = counts.get(ann["class"], 0) + 1
        breakdown = ", ".join([f"{k}: {v}" for k,v in counts.items()])
        status = f"Tools: {len(self.annotations)} | {breakdown} | Last score: {self.annotations[-1]['score']:.2f}" if self.annotations else "No tools"
        
        return composite, status

    def _mock_add(self, image, click_coords, tool_type):
        # Mock adder
        x, y = click_coords
        color = CLASS_COLORS.get(tool_type, (0,255,128))
        h,w,_ = image.shape
        
        fake_mask = np.zeros((h,w), dtype=bool)
        cv2.ellipse(fake_mask.astype(np.uint8)*255, (x,y), (60,25), 30, 0, 360, 255, -1)
        fake_mask = fake_mask | (cv2.ellipse(np.zeros((h,w), np.uint8), (x,y), (60,25), 30, 0, 360, 255, -1).astype(bool))
        
        mask = np.zeros((h,w), dtype=bool)
        cv2.ellipse(mask.astype(np.uint8), (x,y), (60,25), 0, 0, 360, 1, -1)
        
        self.annotations.append({
            "mask": np.zeros((h,w), dtype=bool), # temp render
            "class": tool_type,
            "color": color,
            "points": [[x,y]],
            "labels": [1],
            "centroid": (x,y),
            "score": 0.0,
            "mock_xy": (x,y)
        })
        
        composite = image.copy()
        for ann in self.annotations:
            mx, my = ann.get("mock_xy", ann["centroid"])
            cv2.ellipse(composite, (mx,my), (60,25), 30, 0, 360, ann["color"], -1)
            cv2.putText(composite, ann["class"], (mx+10, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return composite, f"[MOCK] {tool_type} en ({x},{y}) - Total: {len(self.annotations)}"

    def clear(self):
        self.annotations = []
        self.last_image = None

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

    def _mock_auto_segmentation(self, image):
        """
        Simulates full-frame automatic instrument detection overlay.
        """
        return image, "OnClick Gradio event", []

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", bgr_image)
        return base64.b64encode(buffer).decode("utf-8")