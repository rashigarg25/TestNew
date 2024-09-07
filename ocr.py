from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import requests
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

class ImageRequest(BaseModel):
    imagePath: str
    query: str


start = time.process_time()


app = FastAPI()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model = Qwen2VLForConditionalGeneration.from_pretrained(
   "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
).to(device)


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

@app.post("/processImage")
async def process_image(request: ImageRequest):
    
    image_path = request.imagePath
    query = request.query

    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text", 
                    "text": query,
                }    
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    #image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)


    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    print(time.process_time() - start)

    return output_text

