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
	# ... (rest of the options remain unchanged)
}

HF_TOKEN = os.environ.get("HF_TOKEN", None)

class ImageAdapter(nn.Module):
    # (No changes here, keeping as it is)
    pass

# Load CLIP, tokenizer, and LLM (unchanged)
# ...

@spaces.GPU()
@torch.no_grad()
def stream_chat(image_paths: list[Image.Image], caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str, custom_prompt: str) -> list[tuple[str, str]]:
    torch.cuda.empty_cache()

    captions = []
    for input_image in image_paths:
        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

        if custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()

        # Preprocess image
        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to('cuda')

        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to('cuda')

        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)

        convo_tokens = convo_tokens.squeeze(0)
        prompt_tokens = prompt_tokens.squeeze(0)

        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))

        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],
            embedded_images.to(dtype=convo_embeds.dtype),
            convo_embeds[:, preamble_len:],
        ], dim=1).to('cuda')

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, suppress_tokens=None)

        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        captions.append((prompt_str, caption.strip()))
    
    return captions

with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column():
            input_images = gr.File(label="Upload Images", type="file", file_count="multiple", file_types=["image"])
            caption_type = gr.Dropdown(choices=list(CAPTION_TYPE_MAP.keys()), label="Caption Type", value="Descriptive")
            caption_length = gr.Dropdown(choices=["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)], label="Caption Length", value="long")
            extra_options = gr.CheckboxGroup(choices=["Include information about camera angle.", "Include lighting details.", "Include watermark presence."], label="Extra Options")
            name_input = gr.Textbox(label="Person/Character Name (if applicable)")
            custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")

            run_button = gr.Button("Caption")

        with gr.Column():
            output_prompts = gr.Textbox(label="Prompts used for all images")
            output_captions = gr.Textbox(label="Generated captions for all images")

    run_button.click(fn=stream_chat, inputs=[input_images, caption_type, caption_length, extra_options, name_input, custom_prompt], outputs=[output_prompts, output_captions])

if __name__ == "__main__":
    demo.launch()
