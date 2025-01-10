from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import base64
import io

app = Flask(__name__)

# Load VLM model and processor
DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16
).to(DEVICE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to process video frames and generate inference using VLM.
    """
    try:
        # Receive the frame and prompt from the client
        data = request.get_json()
        frame_data = base64.b64decode(data['image'])
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'Prompt is required.'}), 400

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
        return jsonify({'response': generated_texts[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, threaded=True)
