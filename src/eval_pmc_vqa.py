from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import pandas as pd
from PIL import Image

DATASET="PMC_VQA"
# MODEL_PATH="qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="qwen_Qwen2.5-VL-3B-Instruct_PMC_VQA/checkpoint-250"
MODEL_NAME = MODEL_PATH.replace("/", "_")
BSZ=8 # reduce it if GPU OOM
OUTPUT_PATH=f"./logs/{DATASET}_{MODEL_NAME}.json"
CSV_PATH="/home/geonhee/workspace/deepspeed-grpo/vqa_datasets/PMC_VQA/test.csv"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)
NUM_EXAMPLES = 100
data = pd.read_csv(CSV_PATH)[:NUM_EXAMPLES]

QUESTION_TEMPLATE = """{Question} 
{Choice_A}
{Choice_B}
{Choice_C}
{Choice_D}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags."""

messages = []

for i in range(len(data)):
    row = data.iloc[i]
    image_path = "/home/geonhee/workspace/deepspeed-grpo/vqa_datasets/PMC_VQA/images/" + row["Figure_path"]
    question = QUESTION_TEMPLATE.format(
        Question=row['Question'].strip(),
        Choice_A=row['Choice A'].strip(),
        Choice_B=row['Choice B'].strip(),
        Choice_C=row['Choice C'].strip(),
        Choice_D=row['Choice D'].strip()
    )
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((480, 480), Image.LANCZOS)    
    message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
    }]
    messages.append(message)

all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

def extract_letter_answer(output_str):
    # Try to find a single letter within <answer> tags; return None if not found
    answer_pattern = r'<answer>\s*([a-zA-Z])\s*</answer>'
    match = re.search(answer_pattern, output_str)

    if match:
        return match.group(1)
    return None

final_output = []
correct_number = 0

for input_example, model_output in zip(data.to_dict(orient='records'), all_outputs):
    original_output = model_output
    ground_truth = input_example['Answer_label']
    model_answer = extract_letter_answer(original_output)
    
    # Create a result dictionary for this example
    result = {
        'question': input_example,
        'ground_truth': ground_truth,
        'model_output': original_output,
        'extracted_answer': model_answer
    }
    final_output.append(result)
    
    # Count correct answers
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2)

print(f"Results saved to {output_path}")