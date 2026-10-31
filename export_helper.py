
"""
Save a JSON file for evaluation
"""
import json, os
from datetime import datetime

def export_json(before_counts, after_counts, annotations_before, annotations_after, mode="SAM 3.1 + exemplar batch", time_s=15, sam_boxes=None):
    os.makedirs("exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "timestamp": ts,
        "mode": mode,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "time_s": time_s,
        "sam_boxes": sam_boxes or {},
        "annotations_before": [{k: str(v) if k=="bbox" else v for k,v in ann.items() if k!="mask"} for ann in annotations_before],
        "annotations_after": [{k: str(v) if k=="bbox" else v for k,v in ann.items() if k!="mask"} for ann in annotations_after],
    }
    path = f"exports/mode4_{ts}.json"
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[EXPORT] {path}")
    return path