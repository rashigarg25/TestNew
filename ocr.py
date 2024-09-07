import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time


start = time.process_time()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#device = "cpu"
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
   "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
).to(device)

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#      "Qwen/Qwen2-VL-7B-Instruct",
#      torch_dtype=torch.bfloat16,
#      attn_implementation="flash_attention_2",
#      device_map="auto",
# )


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "../wa/18.jpeg",
            },
            {
                "type": "text", 
                "text": "What is the value of uhid of each image? Only the image name with value, don't share text"},
        ],
    }
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
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