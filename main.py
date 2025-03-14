from grpo_trainer import MinimalGRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
import torch
import regex as re

def format_reward(prompt, completion, **kwargs):
    return 1.0 if ("<think>" in completion and "</think>" in completion and "<answer>" in completion and "</answer>" in completion) else -1.0 

if __name__ == "__main__":
    config = GRPOConfig()
    config.num_iterations = 1
    config.num_generations = 2
    config.max_steps = 1000
    config.per_device_train_batch_size = 2
    config.gradient_accumulation_steps = 1
    config.gradient_checkpointing=True
    config.report_to = "wandb"
    config.run_name = "smolvlm2_format_reward"
    config.logging_steps= 1

    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  
    original_train_dataset = [
        {
            "prompt": "Generate a radiology report for this Chest X-ray image. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think><answer> radiology report </answer>",
            "image_path": "/mnt/sohn2022/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
        },
        {
            "prompt": "Generate a radiology report for this Chest X-ray image. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think><answer> radiology report </answer>",
            "image_path": "/mnt/sohn2022/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s56699142/ea030e7a-2e3b1346-bc518786-7a8fd698-f673b44c.jpg"
        }
    ] * 32

    train_dataset = []
    for entry in original_train_dataset:
        new_entry = {
            "prompt": entry["prompt"],
            "image_path": entry["image_path"],
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": entry["image_path"]},
                        {"type": "text", "text": entry["prompt"]},
                    ]
                }
            ]
        }

        train_dataset.append(new_entry)

    trainer = MinimalGRPOTrainer(
        model_name=model_id,
        reward_func=format_reward,
        args=config,
        train_dataset=train_dataset,
    )
    trainer.train()

