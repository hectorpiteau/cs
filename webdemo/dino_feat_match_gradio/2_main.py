# This Gradio app allows the user to click on an image and returns the coordinates of the click.
import gradio as gr
import numpy as np
import traceback
import sys

print("Starting script...", flush=True)
print(f"Gradio version: {gr.__version__}", flush=True)
print(f"NumPy version: {np.__version__}", flush=True)

# Define a function that takes a SelectData event and returns the click coordinates.
def handle_click(evt: gr.SelectData):
    print(evt)
    result = f"You clicked at coordinates: {evt.index}"
    
    try:
        print("[DEBUG] handle_click called", flush=True)
        print(f"[DEBUG] evt type: {type(evt)}", flush=True)
        print(f"[DEBUG] evt.index: {evt.index}", flush=True)
        print(f"[DEBUG] dir(evt): {dir(evt)}", flush=True)
        result = f"You clicked at coordinates: {evt.index}"
        print(f"[DEBUG] Returning: {result}", flush=True)
        return result
    except Exception as e:
        print(f"[ERROR] Exception in handle_click: {e}", flush=True)
        traceback.print_exc()
        return f"Error: {str(e)}"

try:
    print("Generating image...", flush=True)
    # Generate a random image.
    # Convert to uint8 format (0-255) to avoid segmentation faults
    image = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}", flush=True)

    print("Creating Gradio interface...", flush=True)
    # Create a Gradio interface that takes an image input, runs it through the handle_click function, and returns output to a textbox.
    # The live parameter is set to True to enable real-time updates.
    # The select method is used to handle the click event and pass the SelectData event to the handle_click function.
    with gr.Blocks() as demo:
        img_input = gr.Image(value=image, label="Click anywhere", type="numpy")
        text_output = gr.Textbox(label="Click Coordinates")
        img_input.select(fn=handle_click, inputs=[], outputs=text_output)
    
    print("Interface created successfully!", flush=True)
    print("Launching interface...", flush=True)
    # Launch the interface.
    demo.launch(show_error=True, server_name="0.0.0.0")
    
except Exception as e:
    print(f"[FATAL ERROR] {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
