import os
import cv2
import numpy as np
import base64
import asyncio

try:
    from openai import AsyncOpenAI
    from meta_sam_parser import (
        parse_responses_stream,
        decode_mask_to_raster,
        CompletedOutcome,
        image_segmentation_format,
        video_segmentation_format
    )
    # Try to import record types - they may vary by version
    try:
        from meta_sam_parser import SegmentationMaskRecord, SegmentationBoxRecord
    except ImportError:
        SegmentationMaskRecord = None
        SegmentationBoxRecord = None
    PARSER_AVAILABLE = True
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    PARSER_AVAILABLE = False
    print("[WARN] pip install openai meta-sam-parser")

# Loading SAM in local just for OnClick() Gradio event
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Masking colors for tools classes segmentation
CLASS_COLORS = {
    "Episiotomy scissors": (0, 255, 128),   # green
    "Hemostatic forceps": (255, 128, 0),    # orange
    "Scalpel": (0, 128, 255),               # blue
    "Stitch scissors": (255, 0, 128),       # magenta
    "Gauze / Sponge": (255, 255, 0),        # yellow
    "Retractor": (128, 0, 255),             # purple
    "Syringe": (0, 255, 255),               # cyan
    "Gloves": (255, 255, 255),              # white
}

class SAMClient:
    def __init__(self, checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b"):
        self.device = "cuda" if self._has_cuda() else "cpu"
        self.predictor = None
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.last_image = None

        # Masking history
        self.annotations = [] # {mask, class, color, points, labels, centroid}
        
        # Meta API credentials for text prompts
        self.api_key = os.getenv("META_API_KEY")
        self.api_base = os.getenv("META_API_BASE")
        self.text_model = os.getenv("SAM_TEXT_MODEL")

        if SAM_AVAILABLE and os.path.exists(checkpoint_path):
            print(f"[INFO] Loading SAM {model_type} on {self.device} from {checkpoint_path}")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=16,
                pred_iou_thresh=0.90,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=500.0
            )
        else:
            if not SAM_AVAILABLE:
                print("[WARN] Segment Anything Model is not installed. Run: pip install segment-anything torch")
                self.mask_generator = None
            else:
                print(f"[WARN] Checkpoint was not found in {checkpoint_path}. Run: curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                self.mask_generator = None
    
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

    #region Local inference
    def predict_mask(self, image: np.ndarray, click_coords: tuple, tool_type: str = "Instrument", point_label: int = 1) -> tuple:
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
            if point_label == 0:
                # If removing shadow, always refine last object regardless of distance
                should_refine = True
            elif last["class"] == tool_type:
                cx, cy = last["centroid"]
                dist = np.hypot(x - cx, y - cy)
                if dist < 150:  # we are selecting the same tool, it is not a new one
                    should_refine = True

        if should_refine:
            # Adding a new point to the previous tool (positive or negative)
            last = self.annotations[-1]
            last["points"].append([x, y])
            last["labels"].append(point_label)  # 1 = add, 0 = remove shadow
            points = np.array(last["points"])
            labels = np.array(last["labels"])
            #fg = sum(np.array(last["labels"]) == 1)
            #bg = sum(np.array(last["labels"]) == 0)
            #print(f"[INFO] Segmentation of {tool_type} improved with {len(points)} points (fg={fg}, bg={bg})")
        else:
            if point_label == 0:
                # Cannot remove shadow without existing object
                composite, status = self._render_composite(image)
                return composite, "To remove shadow, first add a positive point on instrument, then use Remove mode"
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
                "points": points.tolist(),
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

        # Drawing labels and clicks - positive green, negative red X
        for idx, ann in enumerate(self.annotations):
            cx, cy = ann["centroid"]
            color = ann["color"]
            label = ann["class"]
            # Visual border for label
            offset_y = idx * 18
            cv2.putText(composite, f"{label}", (cx+10, cy-10 - offset_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(composite, f"{label}", (cx+10, cy-10 - offset_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            # Draw all points used
            for pt, lb in zip(ann["points"], ann["labels"]):
                px, py = int(pt[0]), int(pt[1])
                if lb == 1:
                    cv2.circle(composite, (px, py), 5, (0, 255, 0), -1)
                    cv2.circle(composite, (px, py), 7, (255,255,255), 1)
                else:
                    cv2.circle(composite, (px, py), 5, (0, 0, 255), -1)
                    cv2.line(composite, (px-4, py-4), (px+4, py+4), (255,255,255), 2)
                    cv2.line(composite, (px+4, py-4), (px-4, py+4), (255,255,255), 2)
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

    def undo_last_point(self, image: np.ndarray) -> tuple:
        """Remove last point from last annotation, for bad clicks"""
        if len(self.annotations) == 0:
            return image, "No objects to undo"
        ann = self.annotations[-1]
        if len(ann["points"]) <= 1:
            return self.undo_last(image)
        ann["points"].pop()
        ann["labels"].pop()
        points = np.array(ann["points"])
        labels = np.array(ann["labels"])
        masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
        best_idx = np.argmax(scores)
        ann["mask"] = masks[best_idx]
        ann["score"] = float(scores[best_idx])
        ys, xs = np.where(ann["mask"])
        ann["centroid"] = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else ann["centroid"]
        composite, status = self._render_composite(image)
        return composite, f"Point removed. {status}"

    def undo_last(self, image: np.ndarray) -> tuple:
        """Remove last full instrument"""
        if len(self.annotations) > 0:
            removed = self.annotations.pop()
            print(f"[INFO] Removed {removed['class']}")
        composite, status = self._render_composite(image) if len(self.annotations) > 0 else (image, "No objects")
        return composite, f"Removed last. {status}"
    #endregion

    #region Meta API
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
            # SAM API expects RGB PNG base64
            _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode("utf-8")
    
    async def _call_sam_api_async(self, image: np.ndarray, text_prompt: str):
        """
        Official Meta way: /v1/responses streaming + meta-sam-parser
        Docs: learnopencv.com SAM 3.1 - Object Multiplex
        """
        if not PARSER_AVAILABLE:
            raise RuntimeError("Install: pip install openai==2.21.0 meta-sam-parser==0.0.5")
        
        b64_img = self._encode_image_to_base64(image)
        
        async with AsyncOpenAI(base_url=self.api_base, api_key=self.api_key) as client:
            # Important: One concept per request per Meta docs - "dog, cat" is NOT supported as multi-label
            # Use single prompt like "Scalpel"
            events = await client.responses.create(
                model=self.text_model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64_img}"}
                    ]
                }],
                stream=True,
                metadata={"mask_encoding": "one_bit"}
            )
            
            # For image, use image_segmentation_format, for video use video_segmentation_format
            try:
                fmt = image_segmentation_format()
            except:
                fmt = video_segmentation_format()
            
            parsed = parse_responses_stream(events, fmt)
            async with parsed:
                result = await parsed.final_result()
            
            if not isinstance(result.outcome, CompletedOutcome):
                print(f"[WARN] Outcome not completed: {result.outcome}, diagnostics: {result.diagnostics}")
            
            return result

    def predict_with_text(self, image: np.ndarray, text_prompts: list) -> tuple:
        """
        Synchronous wrapper that handles one prompt at a time (per Meta spec: one concept per request)
        """
        if image is None:
            return None, "No image", []
        if not text_prompts:
            return image, "No prompts", []
        if not self.api_key:
            return self._mock_text_segmentation(image, text_prompts)
        
        # Meta says: commas are NOT a documented multi-label interface. Use one phrase per request.
        # So we loop prompts
        self.annotations = []
        all_annotations = []
        h, w = image.shape[:2]
        
        for prompt in text_prompts:
            prompt = prompt.strip()
            if not prompt:
                continue
            try:
                print(f"[INFO] SAM 3.1 API call for concept: '{prompt}'")
                result = asyncio.run(self._call_sam_api_async(image, prompt))
                if not isinstance(result.outcome, CompletedOutcome):
                    print(f"[WARN] Outcome: {result.outcome}, diag: {result.diagnostics}")

                # Properly handle box + mask records
                # Each object id has one box and one mask
                boxes_by_id = {}
                masks_by_id = {}
                for rec in result.records:
                    if hasattr(rec, 'mask') and rec.mask is not None:
                        # Box has bounds with x,y,right,bottom and frame size
                        oid = getattr(rec, 'object_id', '0')
                        rev = getattr(rec, 'revision', 0)
                        prev = masks_by_id.get(oid)
                        if prev is None or rev > getattr(prev, 'revision', 0):
                            masks_by_id[oid] = rec
                    elif hasattr(rec, 'bounds'):
                        oid = getattr(rec, 'object_id', '0')
                        boxes_by_id[oid] = rec
                
                print(f"[INFO] Found {len(boxes_by_id)} boxes, {len(masks_by_id)} masks for '{prompt}'")
                
                for obj_id, mask_rec in masks_by_id.items():
                    try:
                        # Decode raster bytes
                        raster_bytes = decode_mask_to_raster(mask_rec.mask)
                        # According to docs, mask has width, height, and bounds has position
                        # SegmentationMask has width,height ; bounds has x,y ?
                        # Try to get x,y from mask_rec.bounds or mask_rec.mask
                        # From pypi docs: SegmentationMaskRecord has bounds same as box
                        # bounds is half-open box
                        try:
                            # Try bounds attribute on record
                            bounds = getattr(mask_rec, 'bounds', None) or getattr(mask_rec, 'box', None)
                            if bounds is None:
                                # Try box record for same id
                                box_rec = boxes_by_id.get(obj_id)
                                bounds = getattr(box_rec, 'bounds', None) if box_rec else None
                        except:
                            bounds = None
                        
                        mh = mask_rec.mask.height
                        mw = mask_rec.mask.width
                        
                        # x,y placement - from bounds if available, else 0,0
                        x0, y0 = 0, 0
                        if bounds is not None:
                            try:
                                # bounds has left, top, right, bottom or x,y
                                x0 = int(getattr(bounds, 'left', getattr(bounds, 'x', 0)))
                                y0 = int(getattr(bounds, 'top', getattr(bounds, 'y', 0)))
                            except Exception as be:
                                print(f"[WARN] bounds parse failed {be}, bounds={bounds}")
                        
                        arr = np.frombuffer(raster_bytes, dtype=np.uint8).reshape(mh, mw) > 0
                        full_mask = np.zeros((h, w), dtype=bool)
                        x1 = min(w, x0 + mw)
                        y1 = min(h, y0 + mh)
                        ah = y1 - y0
                        aw = x1 - x0
                        if ah > 0 and aw > 0:
                            full_mask[y0:y1, x0:x1] = arr[:ah, :aw]
                        
                        ys, xs = np.where(full_mask)
                        centroid = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else (w//2, h//2)
                        
                        all_annotations.append({
                            "mask": full_mask,
                            "class": prompt,
                            "color": CLASS_COLORS.get(prompt, (0, 128, 255)),
                            "points": [],
                            "labels": [],
                            "centroid": centroid,
                            "score": 0.95,
                            "source": "sam3.1_api",
                            "object_id": obj_id
                        })
                        print(f"[INFO] Decoded mask id={obj_id} at ({x0},{y0}) size {mw}x{mh} - pixels {arr.sum()}")
                    except Exception as e:
                        print(f"[WARN] Decode failed for {prompt} id {obj_id}: {e} - {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                        continue
            except Exception as e:
                print(f"[ERROR] API call for '{prompt}' failed: {e}")
                import traceback
                traceback.print_exc()
                #mock_comp, _, mock_anns = self._mock_text_segmentation(image, [prompt])
                #all_annotations.extend(mock_anns)
                print(f"[ERROR] API failed for '{prompt}', keeping empty")
                continue
        
        if not all_annotations:
            print(f"[WARN] No masks for {text_prompts} decoded")
            self.annotations = []
            status = f"SAM 3.1 API: No objects found for {text_prompts} (0 masks)"
            #return self._mock_text_segmentation(image, text_prompts)
            return image, status, []
        
        self.annotations = all_annotations
        #self.annotations.extend(all_annotations)
        self.last_image = image.copy()
        composite, _ = self._render_composite(image)
        status = f"SAM 3.1 API: Found {len(all_annotations)} objects for {text_prompts}"
        return composite, status, all_annotations

    #region Llama report
    async def _call_llama_async(self, system_prompt: str, user_prompt: str):
        from openai import AsyncOpenAI
        llama_model = os.getenv("LLAMA_MODEL", "muse-spark-1.3-contributor")
        async with AsyncOpenAI(base_url=self.api_base, api_key=self.api_key) as client:
            # Muse Spark/Wonder usan responses API igual que SAM, no chat.completions
            resp = await client.responses.create(
                model=llama_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
                ],
            )
            # Extraer texto de responses
            text = ""
            try:
                text = resp.output_text
            except:
                try:
                    text = resp.output[0].content[0].text
                except:
                    text = str(resp)
            print(f"[LLAMA] Raw response length: {len(text)}")
            return text

    def generate_llama_report(self, before_counts: dict, after_counts: dict, prompts: list, traceability_text: str):
        """Sync wrapper for Llama"""
        if not self.api_key:
            return "No API key - using mock", "[MOCK] Llama is working in local mode"

        system_prompt = (
            "You are a clinical surgical auditor with expertise in the WHO Safe Surgery protocol. "
            "You analyze the traceability of surgical instruments detected by SAM 3.1. "
            "Generate a concise clinical report in English, including the risk level and the possible location of the retained instrument, "
            "and recommendations. Do not invent instruments that are not listed."
        )
        user_prompt = f"""
        Input prompt: {prompts}
        Pre-operation counts: {before_counts}
        Post-operation counts: {after_counts}
        SAM Technical Report:
        {traceability_text}

        Generate:
        1. Executive summary (2 lines)
        2. Risk analysis (high/medium/low) and justification
        3. Missing instruments and possible location (intra-corporeal or in the field)
        4. Recommended action according to the WHO checklist
        5. Note: Limitations of SAM 3.1 in distinguishing between gauze and gloves due to similar colors
        """

        try:
            print(f"[LLAMA] Calling to {os.getenv('LLAMA_MODEL','llama-4-maverick')}")
            report = asyncio.run(self._call_llama_async(system_prompt, user_prompt))
            if not report or len(report.strip()) < 10:
                report = f"[WARN] Llama answer is empty, using a base report instead:\n{traceability_text}"
            return report, f"Llama report created: {len(report)} chars"
        except Exception as e:
            print(f"[ERROR] Llama call failed: {e}")
            import traceback
            traceback.print_exc()
            fallback = f"CLINICAL REPORT (fallback without Llama - error {e}):\n{traceability_text}\n\nRisk: Check manually the remaining instruments."
            return fallback, f"Error using Llama API"
        
    def _mock_text_segmentation(self, image, text_prompts):
        h, w = image.shape[:2]
        self.annotations = []
        import random
        random.seed(42)
        for prompt in text_prompts:
            for j in range(random.randint(1,2)):
                cx = random.randint(int(w*0.2), int(w*0.8))
                cy = random.randint(int(h*0.2), int(h*0.8))
                temp = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(temp, (cx, cy), (50, 20), random.randint(0,90), 0, 360, 255, -1)
                mask = temp > 0
                ys, xs = np.where(mask)
                centroid = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else (cx, cy)
                self.annotations.append({"mask": mask, "class": prompt, "color": CLASS_COLORS.get(prompt, (0,128,255)), "points": [], "labels": [], "centroid": centroid, "score": 0.9, "source": "mock"})
        composite, _ = self._render_composite(image)
        status = f"[MOCK TEXT] {len(self.annotations)} for {text_prompts}"
        return composite, status, self.annotations
    
    def auto_generate_all_masks(self, image: np.ndarray):
        """Mode 4: SAM sugests all masks without prompts, then Muse Spark clasify them"""
        if image is None:
            return None, "No image", []
        if not SAM_AVAILABLE or self.mask_generator is None:
            print("[WARN] SamAutomaticMaskGenerator not available")
            return image, "SAM auto not available", []
        
        print(f"[AUTO] Auto-masking the image {image.shape}")
        self.annotations = []
        masks_data = self.mask_generator.generate(image)
        # Filtering masks over 70% of the imagen or <0.3% (noise)
        H, W = image.shape[:2]
        total_area = H*W
        filtered = []
        for m in masks_data:
            area = m['area']
            x, y, w, h = m['bbox']
            # Filter 1: By area
            if area > total_area*0.6:  # all image
                continue
            if area < total_area*0.003:  # dust <0.3%
                continue
            # Filter 2: By bbox
            if w > W*0.85 or h > H*0.85:
                    continue
            # Filter 3: By border colission
            if (x < 5 or y < 5) and (w > W*0.5 or h > H*0.5):
                continue
            filtered.append(m)
        masks_data = filtered
        
        all_annotations = []
        for idx, m in enumerate(masks_data):
            seg = m['segmentation']  # bool array HxW
            bbox = m['bbox']  # x,y,w,h
            x, y, w, h = bbox
            cx, cy = int(x + w/2), int(y + h/2)
            all_annotations.append({
                "mask": seg,
                "bbox": bbox,
                "class": f"object-{idx}",
                "color": (128, 128, 128),
                "points": [],
                "labels": [],
                "centroid": (cx, cy),
                "score": float(m.get('stability_score', 0.9)),
                "source": "sam_auto",
                "area": int(m.get('area', 0)),
                "predicted_iou": float(m.get('predicted_iou', 0))
            })
        
        self.annotations = all_annotations
        self.last_image = image.copy()
        composite, _ = self._render_composite(image)
        print(f"[AUTO] Auto-masking {len(all_annotations)} created")
        return composite, f"Auto: {len(all_annotations)} masks (SAM segmentation + Muse classify)", all_annotations

    def _encode_crop_to_base64(self, crop: np.ndarray):
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    async def _classify_crop_muse_async(self, crop_b64: str, options: list):
        from openai import AsyncOpenAI
        model_id = os.getenv("LLAMA_MODEL")
        system_prompt = (
            "You are an expert in surgical instruments. Classify the image clip. "
            f"Possible options: {', '.join(options)}. "
            "Respond ONLY in JSON format: {\"instrument\": \"exact name\", \"confidence\": 0-100, \"reason\": \"brief\"}"
        )
        user_prompt = "Classify this surgical instrument. If it is not an instrument, explain. {\"instrument\": \"background\", \"confidence\": 0}"
        
        async with AsyncOpenAI(base_url=self.api_base, api_key=self.api_key) as client:
            resp = await client.responses.create(
                model=model_id,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{crop_b64}"}
                    ]}
                ],
            )
            try:
                text = resp.output_text
            except:
                text = resp.output[0].content[0].text
            return text

    def classify_masks_with_muse(self, image: np.ndarray, options: list):
        """Mode 4: Take the 7 auto-generated masks, trim each bbox, and classify them using Muse"""
        if len(self.annotations) == 0:
            return image, "First, run `auto_generate_all_masks`", []
        
        print(f"[MUSE-CLASSIFY] Sorting {len(self.annotations)} crops with {os.getenv('LLAMA_MODEL')}")
        options = [
            "Kelly hemostatic forceps curved 14cm with serrated jaws",
            "Pean forceps straight 16cm", 
            "Metzenbaum dissecting scissors curved",
            "Mayo scissors straight",
            "Adson tissue forceps with teeth",
            "scalpel handle number 4 with blade",
            "gauze sponge 2x3 and 5x5",
            "latex glove",
            "plastic syringe 10ml"
        ]
        classified = []
        for idx, ann in enumerate(self.annotations):
            x, y, w, h = ann["bbox"]
            x, y = max(0, int(x)), max(0, int(y))
            w, h = int(w), int(h)
            # Recorte con padding 10px
            x1, y1 = max(0, x-10), max(0, y-10)
            x2, y2 = min(image.shape[1], x+w+10), min(image.shape[0], y+h+10)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            b64 = self._encode_crop_to_base64(crop)
            try:
                json_text = asyncio.run(self._classify_crop_muse_async(b64, options))
                # Parse JSON
                import json as js
                # Reading JSON input from Spark
                start = json_text.find("{")
                end = json_text.rfind("}")+1
                if start!=-1:
                    data = js.loads(json_text[start:end])
                    instrument = data.get("instrument", "unknown")
                    conf = data.get("confidence", 0)
                else:
                    instrument = json_text.strip()[:30]
                    conf = 50
                
                ann["class"] = instrument
                ann["score"] = float(conf)/100.0
                ann["color"] = CLASS_COLORS.get(instrument, CLASS_COLORS.get(instrument.split()[0], (0,255,128)))
                print(f"[MUSE] Mask {idx} bbox {w}x{h} -> {instrument} {conf}%")
                classified.append(ann)
            except Exception as e:
                print(f"[MUSE] Error classifying mask {idx}: {e}")
                ann["class"] = "unknown"
                classified.append(ann)

        classified = [a for a in classified if "background" not in a["class"].lower() and a["score"] > 0.3]
        self.annotations = classified
        composite, _ = self._render_composite(image)
        return composite, f"Muse classified {len(classified)} instruments", classified


    # Mode 4: Exemplars segmentation
    async def _classify_crop_with_exemplars_async(self, crop_b64, exemplar_b64s, exemplar_labels, options):
        from openai import AsyncOpenAI
        model_id = os.getenv("LLAMA_MODEL")
        exemplar_desc = "\n".join([f"- {lbl}: ref {i+1}" for i,lbl in enumerate(exemplar_labels)])
        system_prompt = (
            f"You're an expert in surgical instruments. You have {len(exemplar_labels)} exemplars in white background 512x512:\n{exemplar_desc}\n"
            f"Options: {', '.join(options)}. "
            "Compare the actual crop image (blue background, blood) with the exemplars. Return JSON ONLY: {\"instrument\": \"name\", \"confidence\": 0-100, \"matched_exemplar\": \"label\"}"
        )
        user_content = [{"type":"input_text","text":"Classify crop using exemplars:"}]
        for eb64 in exemplar_b64s:
            user_content.append({"type":"input_image","image_url": f"data:image/jpeg;base64,{eb64}"})
        user_content.append({"type":"input_text","text":"Prueba bandeja sucia:"})
        user_content.append({"type":"input_image","image_url": f"data:image/jpeg;base64,{crop_b64}"})
        async with AsyncOpenAI(base_url=self.api_base, api_key=self.api_key) as client:
            resp = await client.responses.create(
                model=model_id,
                input=[
                    {"role":"system","content":[{"type":"input_text","text":system_prompt}]},
                    {"role":"user","content": user_content}
                ],
            )
            try:
                return resp.output_text
            except:
                return resp.output[0].content[0].text

    def classify_masks_with_exemplars(self, image, exemplar_folder="exemplars_clean", options=None):
        from pathlib import Path
        if len(self.annotations) == 0:
            return image, "run auto_generate_all_masks", []
        if options is None:
            options = ["Babcock tissue forceps","DeBakey dissector","Metzenbaum scissors","forceps","Episiotomy scissors","Mayo needle holder","Stitch scissors","Syringe","gauze","glove"]
        exemplar_paths = list(Path(exemplar_folder).glob("*.jpg")) + list(Path(exemplar_folder).glob("*.png"))
        if not exemplar_paths:
            print(f"[WARN] No exemplars en {exemplar_folder}, fallback")
            return self.classify_masks_with_muse(image, options)
        exemplar_b64s=[]; exemplar_labels=[]
        for p in exemplar_paths[:4]:
            img = cv2.imread(str(p))
            if img is None: continue
            exemplar_b64s.append(self._encode_crop_to_base64(img))
            exemplar_labels.append(p.stem)
        print(f"[MUSE-EXEMPLAR] {len(self.annotations)} crops con {len(exemplar_b64s)} exemplars")
        classified=[]
        for idx, ann in enumerate(self.annotations):
            x,y,w,h = ann["bbox"]
            x,y = max(0,int(x)), max(0,int(y)); w,h = int(w), int(h)
            x1,y1 = max(0,x-10), max(0,y-10)
            x2,y2 = min(image.shape[1], x+w+10), min(image.shape[0], y+h+10)
            crop = image[y1:y2, x1:x2]
            if crop.size==0: continue
            b64 = self._encode_crop_to_base64(crop)
            try:
                import asyncio, json as js
                json_text = asyncio.run(self._classify_crop_with_exemplars_async(b64, exemplar_b64s, exemplar_labels, options))
                start = json_text.find("{"); end = json_text.rfind("}")+1
                if start!=-1:
                    data = js.loads(json_text[start:end])
                    instrument = data.get("instrument","unknown")
                    conf = data.get("confidence",0)
                else:
                    instrument = json_text.strip()[:40]; conf=50
                    data = {}
                ann["class"]=instrument; ann["score"]=float(conf)/100.0
                ann["matched_exemplar"] = data.get("matched_exemplar","")
                ann["reason"] = data.get("reason","")
                if "background" not in instrument.lower() and conf>=30:
                    classified.append(ann)
            except Exception as e:
                print(f"[EXEMPLAR] Error {idx}: {e}")
                continue
        self.annotations=classified
        composite,_=self._render_composite(image)
        return composite, f"Muse exemplar {len(classified)} using {len(exemplar_b64s)} refs", classified

    def auto_segment_stage(self, image: np.ndarray):
        return self.auto_generate_all_masks(image)

    def _mock_auto_segmentation(self, image):
        """
        Simulates full-frame automatic instrument detection overlay.
        """
        return image, "OnClick Gradio event", []

    