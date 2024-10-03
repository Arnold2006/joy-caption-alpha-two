import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import torchvision.transforms.functional as TVF

CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path("cgrkzexw-599808")
TITLE = "<h1><center>JoyCaption Alpha Two (2024-09-26a)</center></h1>"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critique": [
        "Write an art critique for this image in a formal tone.",
        "Write an art critique for this image within {word_count} words.",
        "Write a {length} art critique for this image in a formal tone.",
    ]
}

def stream_chat(image_paths, caption_type, caption_length, extra_options, name_input, custom_prompt):
    captions = []
    
    # Loop through each image and process it
    for image_path in image_paths:
        # Here you would load and process each image.
        image = Image.open(image_path)
        # Caption generation logic would go here, for now just a placeholder
        caption = f"Generated caption for {image_path.name}"
        captions.append(caption)

    return captions

def batch_process(input_images, caption_type, caption_length, extra_options, name_input, custom_prompt):
    # Handle multiple images, input_images will be a list of images
    captions = stream_chat(input_images, caption_type, caption_length, extra_options, name_input, custom_prompt)
    return captions

# Gradio app modification
with gr.Blocks() as demo:
    gr.Markdown(TITLE)

    # Now using gr.File to accept multiple files
    input_images = gr.File(label="Upload Images", type="file", file_count="multiple")
    
    caption_type = gr.Dropdown(choices=list(CAPTION_TYPE_MAP.keys()), label="Caption Type")
    caption_length = gr.Number(label="Caption Length (Optional)", value=50)
    extra_options = gr.CheckboxGroup(choices=[
        "Include information about camera angle.",
        "Include information about whether there is a watermark or not.",
        "Include information about whether there are JPEG artifacts or not."
    ], label="Extra Options")
    
    name_input = gr.Textbox(label="Person/Character Name (if applicable)")
    custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")
    
    run_button = gr.Button("Caption")
    
    with gr.Column():
        output_captions = gr.Textbox(label="Captions for all images")
    
    # Handle multiple images
    run_button.click(fn=batch_process, inputs=[input_images, caption_type, caption_length, extra_options, name_input, custom_prompt], outputs=[output_captions])

if __name__ == "__main__":
    demo.launch()
