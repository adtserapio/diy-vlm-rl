import os
from datasets import load_dataset
from grpo import QwenGRPOTrainer, accuracy_reward, format_reward
from trl.trainer.grpo_config import GRPOConfig
import torch
import regex as re
import json
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    dataset_name: Optional[str] = field(
        default="default",
        metadata={"help": "Dataset used fo training"}
    )
    

QUESTION_TEMPLATE = """{Question} 
{Choice_A}
{Choice_B}
{Choice_C}
{Choice_D}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags."""

CSV_PATH="/home/geonhee/workspace/deepspeed-grpo/vqa_datasets/PMC_VQA/train.csv"

def main(script_args, training_args, model_args):    
    model_id = model_args.model_name_or_path
    data = pd.read_csv(CSV_PATH)
    train_dataset = []
    for i in range(len(data)):
        if i % 100 == 0: 
            print(f"🔍 {i} data have been loaded") 
        if i == 500:
            break
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
        new_entry = {
            "image": image,
            "solution": row["Answer_label"],
            "prompt": [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "url": image
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }]
        }
        train_dataset.append(new_entry)

    model_name = model_id.replace("/", "_")
    training_args.output_dir = f"{model_name}_{script_args.dataset_name}"
    training_args.run_name = f"{model_name}_{script_args.dataset_name}"

    trainer = QwenGRPOTrainer(
        model_name=model_id,
        reward_funcs=[accuracy_reward, format_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None
    )
    trainer.train() 

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)