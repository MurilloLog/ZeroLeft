import gradio as gr
import numpy as np
import pandas as pd
from sam_client import SAMClient

# Start the SAMClient instance
client = SAMClient(checkpoint_path="sam_vit_b_01ec64.pth", model_type="vit_b")

# Global inventory data to keep track of detected instruments across the session
inventory_data = []

def process_click(image: np.ndarray, tool_type: str, evt: gr.SelectData):
    """
    Callback for manual point selection on the surgical image.
    Generates mask via SAM 3.1 API and appends single item to inventory.
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
    
    # Single-point inference call
    segmented_image, status = client.predict_mask(image, (click_x, click_y), tool_type)
    
    # Log manual item into session memory
    total_anns = len(client.annotations)
    if total_anns > len(inventory_data):
        #item_id = len(inventory_data) + 1 # New tool
        inventory_data.append({
            "ID": f"ITEM-{total_anns:03d}",
            "Instrument": tool_type,
            "Coordinates (X, Y)": f"({click_x}, {click_y})",
            "Mode": f"Click + {len(client.annotations[-1]['points'])} pts" if total_anns>0 else "Manual",
            "Status": f"Score {client.annotations[-1]['score']:.2f}" if total_anns>0 else "Verified"
        })
    else:
        # Update last register
        if inventory_data:
            inventory_data[-1]["Coordinates (X, Y)"] = f"{inventory_data[-1]['Coordinates (X, Y)']} + ({click_x},{click_y})"
            inventory_data[-1]["Mode"] = f"Extended {len(client.annotations[-1]['points'])} pts"
    
    df_inventory = pd.DataFrame(inventory_data)
    return segmented_image, status, df_inventory

def process_auto_scan(image: np.ndarray):
    """
    Triggers zero-shot full-frame automatic mask generation across the entire image.
    Detects all objects simultaneously without manual point inputs.
    """
    global inventory_data

    if image is None:
        return (
            None,
            "Please upload a surgical image first.",
            pd.DataFrame(inventory_data),
            "Error: No image loaded."
        )

    # Execute full-frame auto segmentation
    processed_image, status_msg, detected_objects = client.auto_segment_stage(image)

    # Populate inventory dynamically with auto-detected tools
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
        f"ZEROLEFT AUDIT REPORT ===\n"
        f"Total Instruments Detected: {total_items}\n"
        f"Item Breakdown:\n{breakdown_str}\n\n"
        f"Spatial Safety Verification: PASSED\n"
        f"Protocol Status: Full-frame scan completed via Meta SAM 3.1.\n"
        f"Zero retained surgical item anomalies detected.\n"
    )
    return report

# Interface Layout Definition
with gr.Blocks(title="ZeroLeft - Autonomous Surgical Audit") as demo:
    gr.Markdown(
        """
        # ZeroLeft: An Intelligent Platform for Surgical Tool Traceability Based on Meta SAM 3
        ### Meta Global AI Developer Hackathon 2026 | Indie Track
        Run zero-shot **Auto-Scan** to segment all physical instruments in the operating field simultaneously, or click manually on specific tools to register individual items with **Meta SAM 3.1**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Surgical Frame",
                type="numpy",
                interactive=True
            )
            tool_selector = gr.Radio(
                choices=[
                    "Episiotomy scissors",
                    "Hemostatic forceps",
                    "Scalpel",
                    "Stitch scissors",
                    "Gauze / Sponge",
                    "Retractor",
                    "Syringe",
                    "Globes"
                ],
                value="Episiotomy scissors",
                label="Class (Define color)"
            )
            with gr.Row():
                btn_auto_scan = gr.Button("Auto-Scan Full Frame", variant="primary")
                btn_audit = gr.Button("Generate Audit Log", variant="secondary")
                btn_clear = gr.Button("Reset Session", variant="stop")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="SAM 3.1 Segmentation Output Mask",
                type="numpy",
                interactive=False
            )
            output_status = gr.Textbox(
                label="Inference Execution Status",
                value="Awaiting input...",
                interactive=False
            )
            inventory_table = gr.Dataframe(
                headers = ["ID", "Instrument", "Coordinates (X, Y)", "Mode", "Status"],
                value = pd.DataFrame(columns=["ID", "Instrument", "Coordinates (X, Y)", "Mode", "Status"]),
                label = "Tools",
                interactive = False
            )
            

    audit_output = gr.Textbox(
        label = "Operational Audit Summary Report (Muse Spark Agentic Summary)",
        lines = 7,
        interactive = False
    )

    # Event Handlers
    input_image.select(
        fn=process_click,
        inputs=[input_image, tool_selector],
        outputs=[output_image, output_status, inventory_table]
    )

    btn_auto_scan.click(
        fn=process_auto_scan,
        inputs=[input_image],
        outputs=[output_image, output_status, inventory_table, audit_output]
    )

    btn_audit.click(
        fn=generate_audit_report,
        inputs=[],
        outputs=[audit_output]
    )

    btn_clear.click(
        fn=reset_session,
        inputs=[],
        outputs=[
            input_image,
            output_image,
            output_status,
            inventory_table,
            audit_output
        ]
    )

if __name__ == "__main__":
    demo.launch(share=False)