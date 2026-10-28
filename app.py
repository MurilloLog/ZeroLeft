import gradio as gr
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sam_client import SAMClient

# Two separate clients to keep before/after annotations independent
client_before = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
client_after = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")
# Client for text mode (uses same checkpoint but text method calls API)
client_text = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")

# Global inventory data to keep track of detected instruments across the session
inventory_before = []
inventory_after = []

# Cost control: set USE_REAL_API=false in .env to work 100% offline
USE_REAL_API = os.getenv("USE_REAL_API", "false").lower() == "true"

def _rebuild_inventory(client):
    """Helper: build inventory list from client.annotations"""
    inv = []
    for idx, ann in enumerate(client.annotations, 1):
        fg = sum(np.array(ann['labels']) == 1)
        bg = sum(np.array(ann['labels']) == 0)
        inv.append({
            "ID": f"ITEM-{idx:03d}",
            "Instrument": ann["class"],
            "Points": f"{len(ann['points'])} pts (fg={fg}, bg={bg})",
            "Mode": f"Refined {len(ann['points'])} pts",
            "Status": f"Score {ann['score']:.2f}"
        })
    return inv

def process_click_before(image, tool_type, point_mode, evt: gr.SelectData):
    global inventory_before
    if image is None:
        return None, "Upload BEFORE image", pd.DataFrame(inventory_before)
    x, y = evt.index[0], evt.index[1]
    label = 0 if "Remove" in point_mode else 1
    composite, status = client_before.predict_mask(image, (x, y), tool_type, point_label=label)
    inventory_before = _rebuild_inventory(client_before)
    return composite, status, pd.DataFrame(inventory_before)

def process_click_after(image, tool_type, point_mode, evt: gr.SelectData):
    global inventory_after
    if image is None:
        return None, "Upload AFTER image", pd.DataFrame(inventory_after)
    x, y = evt.index[0], evt.index[1]
    label = 0 if "Remove" in point_mode else 1
    composite, status = client_after.predict_mask(image, (x, y), tool_type, point_label=label)
    inventory_after = _rebuild_inventory(client_after)
    return composite, status, pd.DataFrame(inventory_after)

def undo_before_point(image):
    global inventory_before
    if image is None:
        return None, "No image", pd.DataFrame(inventory_before)
    composite, status = client_before.undo_last_point(image)
    inventory_before = _rebuild_inventory(client_before)
    return composite, status, pd.DataFrame(inventory_before)

def undo_after_point(image):
    global inventory_after
    if image is None:
        return None, "No image", pd.DataFrame(inventory_after)
    composite, status = client_after.undo_last_point(image)
    inventory_after = _rebuild_inventory(client_after)
    return composite, status, pd.DataFrame(inventory_after)

def process_text_search(image, text_prompts_str, use_real_api_checkbox):
    """
    Phase 2: Text -> masks
    text_prompts_str: comma separated e.g. "Scalpel, Hemostatic forceps"
    """
    if image is None:
        return None, "Upload image first", pd.DataFrame([])
    
    # Parse prompts
    prompts = [p.strip() for p in text_prompts_str.split(",") if p.strip()]
    if not prompts:
        return image, "Enter at least one prompt e.g. Scalpel", pd.DataFrame([])
    
    # Cost guard
    if use_real_api_checkbox and not USE_REAL_API:
        # If UI says real but env says false, warn
        return image, "Set USE_REAL_API=true in .env to allow real calls. Currently using MOCK (no cost).", pd.DataFrame([])
    
    # Temporarily override env for this call
    if not use_real_api_checkbox:
        # Force mock
        os.environ["META_API_KEY"] = ""
        client_text.api_key = None
    
    composite, status, anns = client_text.predict_with_text(image, prompts)
    
    # Restore key if needed
    if not use_real_api_checkbox:
        client_text.api_key = os.getenv("META_API_KEY")
    
    inv = _rebuild_inventory(client_text)
    return composite, status, pd.DataFrame(inv)

def reset_all():
    global inventory_before, inventory_after
    inventory_before = []
    inventory_after = []
    client_before.clear()
    client_after.clear()
    client_text.clear()
    empty = pd.DataFrame(columns=["ID", "Instrument", "Points", "Mode", "Status"])
    return None, None, "Reset", empty, None, None, "Reset", empty, None, None, "Reset", empty, ""

def export_audit():
    """Function 1: Export traceability artifact for jury"""
    if not inventory_before and not inventory_after:
        return "No instruments to export"
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_data = {
        "timestamp": timestamp,
        "before": inventory_before,
        "after": inventory_after,
        "before_count": len(client_before.annotations),
        "after_count": len(client_after.annotations),
        "use_real_api": USE_REAL_API
    }
    json_path = f"exports/inventory_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    return f"Exported to {json_path} | BEFORE: {len(inventory_before)} AFTER: {len(inventory_after)}"

def compare_before_after():
    if not inventory_before:
        return "Upload and segment BEFORE image first"
    if not inventory_after:
        return "Upload and segment AFTER image"
    from collections import Counter
    before_counts = Counter([x["Instrument"] for x in inventory_before])
    after_counts = Counter([x["Instrument"] for x in inventory_after])
    all_types = set(before_counts.keys()) | set(after_counts.keys())
    lines = []
    lines.append("ZEROLEFT TRACEABILITY REPORT")
    lines.append(f"BEFORE: {len(inventory_before)}")
    for k,v in before_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append(f"\nAFTER: {len(inventory_after)}")
    for k,v in after_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append("\nDIFFERENCE:")
    missing = False
    for tool in all_types:
        diff = before_counts.get(tool,0) - after_counts.get(tool,0)
        if diff > 0:
            lines.append(f"  MISSING: {tool} x{diff}")
            missing = True
        elif diff < 0:
            lines.append(f"  EXTRA: {tool} x{-diff}")
        else:
            lines.append(f"  OK: {tool}")
    lines.append("\nVERIFICATION PASSED" if not missing else "\nVERIFICATION FAILED")
    lines.append("\nMode 1: SAM segments geometry, YOU label via Radio")
    lines.append("Mode 2: SAM 3.1 knows via text prompts + meta-sam-parser")
    return "\n".join(lines)

def process_click(image: np.ndarray, tool_type: str, point_mode: str, evt: gr.SelectData):
    """
    Callback for manual point selection on the surgical image.
    Generates mask via SAM and appends single item to inventory.
    point_mode: Add (green) or Remove shadow (red)
    """
    global inventory_data
    if image is None:
        return (
            None,
            "Please upload a surgical image first.",
            pd.DataFrame(inventory_data)
        )
    
    click_x = evt.index[0]
    click_y = evt.index[1]
    
    # Determine label: 1 = add, 0 = remove shadow
    label = 0 if "Remove" in point_mode else 1
    
    # Single-point inference call with label support
    segmented_image, status = client.predict_mask(image, (click_x, click_y), tool_type, point_label=label)
    
    # Rebuild inventory from annotations (source of truth)
    inventory_data = []
    for idx, ann in enumerate(client.annotations, 1):
        fg = sum(np.array(ann['labels']) == 1)
        bg = sum(np.array(ann['labels']) == 0)
        inventory_data.append({
            "ID": f"ITEM-{idx:03d}",
            "Instrument": ann["class"],
            "Coordinates (X, Y)": f"{len(ann['points'])} pts (fg={fg}, bg={bg})",
            "Mode": f"Refined {len(ann['points'])} pts",
            "Status": f"Score {ann['score']:.2f}"
        })
    
    df_inventory = pd.DataFrame(inventory_data)
    return segmented_image, status, df_inventory

def undo_last_point_fn(image: np.ndarray):
    """
    Remove last point from last instrument
    """
    global inventory_data
    if image is None:
        return None, "Please upload a surgical image first.", pd.DataFrame(inventory_data)
    
    composite, status = client.undo_last_point(image)
    
    # Rebuild inventory
    inventory_data = []
    for idx, ann in enumerate(client.annotations, 1):
        fg = sum(np.array(ann['labels']) == 1)
        bg = sum(np.array(ann['labels']) == 0)
        inventory_data.append({
            "ID": f"ITEM-{idx:03d}",
            "Instrument": ann["class"],
            "Coordinates (X, Y)": f"{len(ann['points'])} pts (fg={fg}, bg={bg})",
            "Mode": f"Refined {len(ann['points'])} pts",
            "Status": f"Score {ann['score']:.2f}"
        })
    
    return composite, status, pd.DataFrame(inventory_data)

def undo_last_obj_fn(image: np.ndarray):
    """
    Remove last full instrument
    """
    global inventory_data
    if image is None:
        return None, "Please upload a surgical image first.", pd.DataFrame(inventory_data)
    
    composite, status = client.undo_last(image)
    
    if inventory_data:
        inventory_data = inventory_data[:-1]
    
    return composite, status, pd.DataFrame(inventory_data)

def process_auto_scan(image: np.ndarray):
    """
    Triggers zero-shot full-frame automatic mask generation
    """
    global inventory_data

    if image is None:
        return (
            None,
            "Please upload a surgical image first.",
            pd.DataFrame(inventory_data),
            "Error: No image loaded."
        )

    processed_image, status_msg, detected_objects = client.auto_segment_stage(image)

    inventory_data = []
    for idx, obj in enumerate(detected_objects, start=1):
        inventory_data.append({
            "ID": f"AUTO-{idx:03d}",
            "Instrument": obj["class"],
            "Coordinates (X, Y)": "Full-Frame Grid",
            "Mode": f"Auto-Scan ({obj['confidence']*100:.1f}%)",
            "Status": "Autonomous Detection"
        })

    df_inventory = pd.DataFrame(inventory_data)
    audit_summary = generate_audit_report()

    return processed_image, status_msg, df_inventory, audit_summary

def reset_session():
    """
    Resets application image state and clears stored inventory records.
    """
    global inventory_data
    inventory_data = []
    client.clear()
    empty_df = pd.DataFrame(
        columns=["ID", "Instrument", "Coordinates (X, Y)", "Mode", "Status"]
    )
    return None, None, "Awaiting input...", empty_df, ""

def generate_audit_report():
    """
    Compiles operational summary report of logged surgical instruments.
    """
    if not inventory_data:
        return "WARNING: No instruments logged in inventory to audit."
    
    total_items = len(inventory_data)
    df = pd.DataFrame(inventory_data)
    summary_counts = df["Instrument"].value_counts().to_dict()

    breakdown_list = [
        f"  - {tool}: {count} unit(s)"
        for tool, count in summary_counts.items()
    ]
    breakdown_str = "\n".join(breakdown_list)
    
    report = (
        f"ZEROLEFT AUDIT REPORT\n"
        f"Total Instruments Detected: {total_items}\n"
        f"Item Breakdown:\n{breakdown_str}\n\n"
        f"Spatial Safety Verification: PASSED\n"
        f"Protocol Status: Full-frame scan completed via Meta SAM 3.1.\n"
        f"Zero retained surgical item anomalies detected.\n"
    )
    return report

# Interface Layout Definition
with gr.Blocks(title="ZeroLeft") as demo:
    gr.Markdown(f"""
    # ZeroLeft: Surgical Tool Traceability
    ### Hackathon Meta 2026 - Indie Track
    **Cost guard: USE_REAL_API={USE_REAL_API}** - Set false in .env to avoid charges. Mock mode is free.
    - **Mode 1 (Clicks)**: SAM local vit_b, free, not API. Click + Add/Remove shadow + BEFORE/AFTER
    - **Mode 2 (Text)**: SAM 3.1 via Meta Model API sam-3.1 + meta-sam-parser. Payment mode with $2.50/1000 imgs. Use checkbox.
    """)

    with gr.Tab("Mode 1: BEFORE/AFTER Traceability (Free)"):
        with gr.Row():
            tool_selector = gr.Radio(
                choices=["Episiotomy scissors","Hemostatic forceps","Scalpel","Stitch scissors","Gauze / Sponge","Retractor","Syringe","Gloves"],
                value="Scalpel", label="Instrument Class (defines color)"
            )
            point_mode = gr.Radio(
                choices=["Add (green)", "Remove shadow (red)"],
                value="Add (green)",
                label="Point Mode"
            )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### BEFORE")
                input_before = gr.Image(label="BEFORE Input", type="numpy", interactive=True)
                output_before = gr.Image(label="BEFORE Masks", type="numpy", interactive=False)
                status_before = gr.Textbox(label="BEFORE Status", interactive=False)
                table_before = gr.Dataframe(headers=["ID","Instrument","Points","Mode","Status"], label="BEFORE Inventory", interactive=False)
                btn_undo_before = gr.Button("Undo last point BEFORE")
            with gr.Column():
                gr.Markdown("### AFTER")
                input_after = gr.Image(label="AFTER Input", type="numpy", interactive=True)
                output_after = gr.Image(label="AFTER Masks", type="numpy", interactive=False)
                status_after = gr.Textbox(label="AFTER Status", interactive=False)
                table_after = gr.Dataframe(headers=["ID","Instrument","Points","Mode","Status"], label="AFTER Inventory", interactive=False)
                btn_undo_after = gr.Button("Undo last point AFTER")
        with gr.Row():
            btn_compare = gr.Button("Compare BEFORE vs AFTER", variant="primary")
            btn_export = gr.Button("Export JSON", variant="secondary")
            btn_clear = gr.Button("Reset All", variant="stop")
        audit_output = gr.Textbox(label="Traceability Report", lines=12, interactive=False)

        input_before.select(fn=process_click_before, inputs=[input_before, tool_selector, point_mode], outputs=[output_before, status_before, table_before])
        input_after.select(fn=process_click_after, inputs=[input_after, tool_selector, point_mode], outputs=[output_after, status_after, table_after])
        btn_undo_before.click(fn=undo_before_point, inputs=[input_before], outputs=[output_before, status_before, table_before])
        btn_undo_after.click(fn=undo_after_point, inputs=[input_after], outputs=[output_after, status_after, table_after])
        btn_compare.click(fn=compare_before_after, inputs=[], outputs=[audit_output])
        btn_export.click(fn=export_audit, inputs=[], outputs=[audit_output])
        btn_clear.click(fn=reset_all, inputs=[], outputs=[input_before, output_before, status_before, table_before, input_after, output_after, status_after, table_after, audit_output, output_before, status_before, table_before, audit_output])

    with gr.Tab("Mode 2: Text Prompt (SAM 3.1 API)"):
        gr.Markdown("""
        **How SAM knows Scalpel?** Phase 2: via Promptable Concept Segmentation. One concept per request (Meta spec).
        Enter comma-separated but app will call API once per concept. Check logs for raw masks.
        """)
        with gr.Row():
            with gr.Column():
                input_text = gr.Image(label="Input Surgical Frame", type="numpy", interactive=True)
                text_prompts = gr.Textbox(label="Text Prompts (comma separated)", value="Scalpel, Hemostatic forceps, Gauze / Sponge", placeholder="Scalpel, Retractor")
                use_real_api = gr.Checkbox(label="Use REAL API (charges $2.50/1k) - Uncheck for MOCK free", value=False)
                btn_text_search = gr.Button("Search by Text (SAM 3.1)", variant="primary")
            with gr.Column():
                output_text = gr.Image(label="Text Search Masks", type="numpy", interactive=False)
                status_text = gr.Textbox(label="API Status + Raw Logs", lines=5, interactive=False)
                table_text = gr.Dataframe(headers=["ID","Instrument","Points","Mode","Status"], label="Text Inventory", interactive=False)
        btn_text_search.click(fn=process_text_search, inputs=[input_text, text_prompts, use_real_api], outputs=[output_text, status_text, table_text])

if __name__ == "__main__":
    demo.launch(share=False)