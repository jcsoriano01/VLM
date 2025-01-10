from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import base64
import io

# Load VLM model and processor
DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16
).to(DEVICE)

def process_frame(frame_data, prompt):
    """
    Process a video frame and generate inference using VLM.
    
    Args:
        frame_data (bytes): The frame data in base64 encoding.
        prompt (str): The prompt text to send with the image.

    Returns:
        str: The response text from the model's inference.
    """
    try:
        if not prompt:
            raise ValueError('Prompt is required.')

        # Decode the image
        frame = Image.open(io.BytesIO(frame_data))

        # Prepare VLM input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        processed_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=processed_prompt, images=[frame], return_tensors="pt").to(DEVICE)

        # Perform inference
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Return the inference result
        return generated_texts[0]

    except Exception as e:
        return f"Error: {str(e)}"
