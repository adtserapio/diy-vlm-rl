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


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

def main(script_args, training_args, model_args):    
    model_id = "qwen/Qwen2.5-VL-3B-Instruct" 
    dataset = load_dataset("leonardPKU/clevr_cogen_a_train")
    total_samples = len(dataset["train"]) 
    train_dataset = []
    for idx, entry in enumerate(dataset["train"]):
        if idx % 1000 == 0 or idx == total_samples - 1: 
            print(f"🔍 {idx} data have been loaded") 
        if idx == 2000:
            break
        new_entry = {
            "image": entry["image"],
            "solution": entry["solution"],
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": entry["image"]},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=entry["problem"])},
                    ]
                }
            ]
        }
        train_dataset.append(new_entry)

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