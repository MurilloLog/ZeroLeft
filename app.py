import gradio as gr
import numpy as np
import pandas as pd
import json
import os
import cv2
from datetime import datetime
from collections import Counter
from sam_client import SAMClient

# API cost control
USE_REAL_API = os.getenv("USE_REAL_API").lower() == "true"

# Clients
# Two separate clients to keep before/after annotations independent
client_before_click = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
client_after_click = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
# Client for text mode (uses same checkpoint but text method calls API)
client_before_text = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
client_after_text = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
client_single_text = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
client_text = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")

# Global inventory data to keep track of detected instruments across the session
inventory_before = []
inventory_after = []

def _rebuild_inventory(client):
    """Helper: build inventory list from client.annotations"""
    inv = []
    for idx, ann in enumerate(client.annotations, 1):
        pts_info = f"{len(ann.get('points', []))} pts" if ann.get('points') else f"{ann.get('source','')} {ann.get('object_id','')}"
        inv.append({
            "ID": f"ITEM-{idx:03d}",
            "Instrument": ann["class"],
            "Points": pts_info,
            "Score": f"{ann.get('score',0):.2f}",
            "Source": ann.get('source','')
        })
    return inv

#region Mode 1: Clicks
def process_click_before(image, tool_type, point_mode, evt: gr.SelectData):
    global inventory_before
    if image is None:
        return None, "Upload the pre-operation instrumental image", pd.DataFrame(inventory_before)
    x, y = evt.index[0], evt.index[1]
    label = 0 if "Remove" in point_mode else 1
    composite, status = client_before_click.predict_mask(image, (x, y), tool_type, point_label=label)
    inventory_before = _rebuild_inventory(client_before_click)
    return composite, status, pd.DataFrame(inventory_before)

def process_click_after(image, tool_type, point_mode, evt: gr.SelectData):
    global inventory_after
    if image is None:
        return None, "Upload the post-operation instrumental image", pd.DataFrame(inventory_after)
    x, y = evt.index[0], evt.index[1]
    label = 0 if "Remove" in point_mode else 1
    composite, status = client_after_click.predict_mask(image, (x, y), tool_type, point_label=label)
    inventory_after = _rebuild_inventory(client_after_click)
    return composite, status, pd.DataFrame(inventory_after)

def undo_before(image):
    global inventory_before
    if image is None:
        return None, "No image uploaded", pd.DataFrame(inventory_before)
    composite, status = client_before_click.undo_last_point(image)
    inventory_before = _rebuild_inventory(client_before_click)
    return composite, status, pd.DataFrame(inventory_before)

def undo_after(image):
    global inventory_after
    if image is None:
        return None, "No image uploaded", pd.DataFrame(inventory_after)
    composite, status = client_after_click.undo_last_point(image)
    inventory_after = _rebuild_inventory(client_after_click)
    return composite, status, pd.DataFrame(inventory_after)

def compare_manual():
    if not inventory_before:
        return "You need to upload your pre-operation instrumental image first"
    if not inventory_after:
        return "You need to upload your post-operation instrumental image first"
    before_counts = Counter([x["Instrument"] for x in inventory_before])
    after_counts = Counter([x["Instrument"] for x in inventory_after])
    all_types = set(before_counts.keys()) | set(after_counts.keys())
    lines = ["ZEROLEFT TRACEABILITY - MANUAL SEGMENTATION", f"PRE-OPERATION: {len(inventory_before)}", str(dict(before_counts)), f"\nPOST-OPERATION: {len(inventory_after)}", str(dict(after_counts)), "\nDIFERENCES:"]
    missing=False
    for t in all_types:
        diff = before_counts.get(t,0) - after_counts.get(t,0)
        if diff>0:
            lines.append(f"  MISSING: {t} x{diff} - An instrument was lost!")
            missing=True
        elif diff<0:
            lines.append(f"  EXTRA: {t} x{-diff}")
        else:
            lines.append(f"  OK: {t}")
    lines.append("\nPASSED" if not missing else "\nFAILED - Check missing instruments")
    return "\n".join(lines)
#endregion

#region Mode 2: API calls
def process_text_single(image, prompts_str, use_real):
    if image is None:
        return None, "Upload your image", pd.DataFrame([])
    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        return image, "Text your input prompt", pd.DataFrame([])
    # Cost guard
    if not use_real and not USE_REAL_API:
        client_single_text.api_key = None
    else:
        client_single_text.api_key = os.getenv("META_API_KEY")
    
    composite, status, anns = client_single_text.predict_with_text(image, prompts)
    # Restore
    client_single_text.api_key = os.getenv("META_API_KEY")
    inv = _rebuild_inventory(client_single_text)
    return composite, status, pd.DataFrame(inv)

def process_phase3_auto(before_img, after_img, prompts_str, use_real):
    if before_img is None or after_img is None:
        return None, None, "Upload your pre-operation and post-operation images", pd.DataFrame([]), pd.DataFrame([]), "Not image uploaded"
    
    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        return None, None, "Text your input prompt", pd.DataFrame([]), pd.DataFrame([]), "Empty prompt"

    # Cost guard
    api_key_backup_before = client_before_text.api_key
    api_key_backup_after = client_after_text.api_key
    if not use_real:
        client_before_text.api_key = None
        client_after_text.api_key = None

    print(f"Starting auto pre/post operation comparison for {prompts}")

    # Pre-operation
    composite_before, status_before, anns_before = client_before_text.predict_with_text(before_img, prompts)
    inv_before = _rebuild_inventory(client_before_text)
    df_before = pd.DataFrame(inv_before)

    # Post-operation
    composite_after, status_after, anns_after = client_after_text.predict_with_text(after_img, prompts)
    inv_after = _rebuild_inventory(client_after_text)
    df_after = pd.DataFrame(inv_after)

    # Restore keys
    client_before_text.api_key = api_key_backup_before
    client_after_text.api_key = api_key_backup_after

    # Auto comparison
    before_counts = Counter([a["class"] for a in client_before_text.annotations])
    after_counts = Counter([a["class"] for a in client_after_text.annotations])
    all_types = set(before_counts.keys()) | set(after_counts.keys()) | set(prompts)

    lines = []
    lines.append("ZEROLEFT - AUTOMATIC TRACEABILITY")
    lines.append(f"Input prompts: {prompts}")
    lines.append(f"\nPre-operation ({len(client_before_text.annotations)} instruments detected):")
    for k in prompts:
        lines.append(f"  - {k}: {before_counts.get(k,0)}")
    lines.append(f"\nPost-operation ({len(client_after_text.annotations)} instruments detected):")
    for k in prompts:
        lines.append(f"  - {k}: {after_counts.get(k,0)}")

    lines.append("\nANALISIS RESUME")
    missing = False
    extra = False
    for tool in prompts:
        b = before_counts.get(tool,0)
        a = after_counts.get(tool,0)
        diff = b - a
        if diff > 0:
            lines.append(f"  MISSING: {tool} x{diff} - PRE-OPERATION={b} POST-OPERATION={a}")
            missing = True
        elif diff < 0:
            lines.append(f"  EXTRA: {tool} x{-diff} - The POST-OPERATION image has more instruments than PRE-OPERATION image")
            extra = True
        else:
            if b==0 and a==0:
                lines.append(f"  NOT FOUND: {tool}")
            else:
                lines.append(f"  OK: {tool} count={b} matches")

    lines.append("")
    if missing:
        lines.append("VERIFICATION FAILED: Some instruments are missing")
    elif extra:
        lines.append("VERIFICATION WARNING: There are extras (possible false positive)")
    else:
        lines.append("VERIFICATION PASSED: No missing instruments")

    status_combined = f"PRE-OPERATION: {status_before} | POST-OPERATION: {status_after}"
    report = "\n".join(lines)

    return composite_before, composite_after, status_combined, df_before, df_after, report

def reset_manually():
    global inventory_before, inventory_after
    inventory_before = []
    inventory_after = []
    for c in [client_before_click, client_after_click]:
        c.clear()
    empty = pd.DataFrame(columns=["ID","Instrument","Points","Score","Source"])
    return None, None, "Reset", empty, None, None, "Reset", empty, ""

def reset_all():
    global inventory_before, inventory_after
    inventory_before = []
    inventory_after = []
    for c in [client_before_click, client_after_click, client_before_text, client_after_text, client_single_text]:
        c.clear()
    empty = pd.DataFrame(columns=["ID","Instrument","Points","Score","Source"])
    return None, None, "Reset", empty, None, None, "Reset", empty, None, None, "Reset", empty, "", None, None, "", empty, empty, ""

def export_phase3(report):
    os.makedirs("exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "timestamp": ts,
        "phase": "3 - PRE/POST OPERATION Automatic",
        "before_inventory": _rebuild_inventory(client_before_text),
        "after_inventory": _rebuild_inventory(client_after_text),
        "before_count": len(client_before_text.annotations),
        "after_count": len(client_after_text.annotations),
        "report": report
    }
    path = f"exports/auto_segmentation_report_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"Exported to {path}\n{report}"
#endregion

# Interface Layout Definition
with gr.Blocks(title="ZeroLeft") as demo:
    gr.Markdown(f"""
    # ZeroLeft: Surgical Tool Traceability
    ### Meta Global AI Hackathon 2026 | Indie Track
    
    - **Mode 1 (Clicks)**: SAM local vit_b, free, not API. Click + Add/Remove shadow + BEFORE/AFTER operation
    - **Mode 2 (Text)**: SAM 3.1 via Meta Model API sam-3.1 + meta-sam-parser. Payment mode with $2.50/1000 imgs. 
    - **Mode 3 (Auto)**: PRE/POST instrumental image segmentation in automatic mode via Meta API sam-3.1. Payment mode with $2.50/1000 imgs.
    
    Use its checkbox to activate them.
    """)

    with gr.Tab("Mode 1: Manual segmentation"):
        with gr.Row():
            tool_sel = gr.Radio(choices=["Episiotomy scissors","Hemostatic forceps","Scalpel","Stitch scissors","Gauze","Retractor","Syringe","Gloves"], value="Gloves", label="Instrument Class")
            point_mode = gr.Radio(choices=["Add (green)", "Remove shadow (red)"], value="Add (green)", label="Click Mode")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### PRE-OPERATION IMAGE")
                in_before = gr.Image(label="PRE-OPERATION - Full setup", type="numpy", interactive=True)
                out_before = gr.Image(label="PRE-OPERATION - Mask output", type="numpy", interactive=False)
                stat_before = gr.Textbox(label="PRE-OPERATION status", interactive=False)
                tbl_before = gr.Dataframe(headers=["ID","Instrument","Points","Score","Source"], label="PRE-OPERATION count", interactive=False)
                btn_undo_b = gr.Button("Undo selection")
            with gr.Column():
                gr.Markdown("### POST-OPERATION IMAGE")
                in_after = gr.Image(label="POST-OPERATION - Full setup", type="numpy", interactive=True)
                out_after = gr.Image(label="POST-OPERATION - Mask output", type="numpy", interactive=False)
                stat_after = gr.Textbox(label="POST-OPERATION status", interactive=False)
                tbl_after = gr.Dataframe(headers=["ID","Instrument","Points","Score","Source"], label="POST-OPERATION count", interactive=False)
                btn_undo_a = gr.Button("Undo selection")
        with gr.Row():
            btn_comp1 = gr.Button("Manual comparison", variant="primary")
            btn_clear = gr.Button("Reset", variant="stop")
        rep1 = gr.Textbox(label="Manual report", lines=12)

        in_before.select(fn=process_click_before, inputs=[in_before, tool_sel, point_mode], outputs=[out_before, stat_before, tbl_before])
        in_after.select(fn=process_click_after, inputs=[in_after, tool_sel, point_mode], outputs=[out_after, stat_after, tbl_after])
        btn_undo_b.click(fn=undo_before, inputs=[in_before], outputs=[out_before, stat_before, tbl_before])
        btn_undo_a.click(fn=undo_after, inputs=[in_after], outputs=[out_after, stat_after, tbl_after])
        btn_comp1.click(fn=compare_manual, inputs=[], outputs=[rep1])
        btn_clear.click(fn=reset_manually, inputs=[], outputs=[in_before, out_before, stat_before, tbl_before, in_after, out_after, stat_after, tbl_after, rep1])

    with gr.Tab("Mode 2: Prompt segmentation"):
        with gr.Row():
            with gr.Column():
                in_single = gr.Image(label="INPUT IMAGE", type="numpy", interactive=True)
                txt_single = gr.Textbox(label="Prompts comma separated", value="glove, syringe, gauze", placeholder="scalpel, forceps, gauze")
                chk_single = gr.Checkbox(label="Use REAL API ($2.50/1k images) - unchecked = MOCK free", value=False)
                btn_single = gr.Button("Search by Text", variant="primary")
            with gr.Column():
                out_single = gr.Image(label="Masks Text", type="numpy", interactive=False)
                stat_single = gr.Textbox(label="Status", lines=4)
                tbl_single = gr.Dataframe(headers=["ID","Instrument","Points","Score","Source"], label="Inventary Count")
        btn_single.click(fn=process_text_single, inputs=[in_single, txt_single, chk_single], outputs=[out_single, stat_single, tbl_single])

    with gr.Tab("Mode 3: Automatic PRE/POST operation comparison"):
        gr.Markdown("""
        **Workflow:**
        1. Upload your pre-operation (tray containing all the instruments before use) and post-operation image (tray containing all the instruments after use)
        2. Write short prompts—one word, lowercase, singular: `glove, gauze, syringe, scalpel`
        3. Click `Run` to analyze and compare every image
        4. Get the final report
        """)
        with gr.Row():
            with gr.Column():
                in_p3_before = gr.Image(label="PRE-OPERATION - Full setup", type="numpy", interactive=True)
                out_p3_before = gr.Image(label="PRE-OPERATION - Masks auto", type="numpy", interactive=False)
                tbl_p3_before = gr.Dataframe(headers=["ID","Instrument","Points","Score","Source"], label="PRE-OPERATION count auto")
            with gr.Column():
                in_p3_after = gr.Image(label="POST-OPERATION - Post intervention", type="numpy", interactive=True)
                out_p3_after = gr.Image(label="POST-OPERATION - Masks auto", type="numpy", interactive=False)
                tbl_p3_after = gr.Dataframe(headers=["ID","Instrument","Points","Score","Source"], label="POST-OPERATION count auto")
        with gr.Row():
            txt_p3 = gr.Textbox(label="Input prompts to segment", value="glove, gauze, syringe, scalpel", placeholder="glove, gauze, syringe")
            chk_p3 = gr.Checkbox(label="Use REAL API - unchecked = MOCK", value=False)
        with gr.Row():
            btn_p3 = gr.Button("Run - Automatic comparison", variant="primary")
            btn_export_p3 = gr.Button("Export to JSON", variant="secondary")
            btn_reset_all = gr.Button("Reset", variant="stop")
        stat_p3 = gr.Textbox(label="Status PRE-OPERATION | POST-OPERATION", interactive=False)
        rep_p3 = gr.Textbox(label="Automatic report - Traceability", lines=18, interactive=False)

        btn_p3.click(fn=process_phase3_auto, inputs=[in_p3_before, in_p3_after, txt_p3, chk_p3], outputs=[out_p3_before, out_p3_after, stat_p3, tbl_p3_before, tbl_p3_after, rep_p3])
        btn_export_p3.click(fn=export_phase3, inputs=[rep_p3], outputs=[rep_p3])
        btn_reset_all.click(fn=reset_all, inputs=[], outputs=[in_before, out_before, stat_before, tbl_before, in_after, out_after, stat_after, tbl_after, rep1, in_single, out_single, stat_single, tbl_single, out_p3_before, out_p3_after, stat_p3, tbl_p3_before, tbl_p3_after, rep_p3])

if __name__ == "__main__":
    demo.launch(share=False)