import torch
from transformers import AutoProcessor, GenerationConfig, Trainer, AutoModelForImageTextToText
from PIL import Image
from torch.utils.data import Sampler
from accelerate.utils import set_seed
import wandb

class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

class MinimalGRPOTrainer(Trainer):
    def __init__(self, model_name, reward_func, args, train_dataset, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
        )
        self.processing_class = AutoProcessor.from_pretrained(model_name, padding_side="left")
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=1,
            pad_token_id=self.processing_class.tokenizer.pad_token_id
        )
        self.reward_func = reward_func  
        self.global_step = 1

        self.num_generations = args.num_generations  # = G in the GRPO paper
        
        # Multi-step
        self.num_iterations = args.num_iterations 
        self.train_dataset = train_dataset

        super().__init__(
            model=self.model,
            processing_class=self.processing_class, 
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: x,
            **kwargs
        )

    def _get_train_sampler(self):
        effective_batch_size = (
            self.args.per_device_train_batch_size
            # * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )
    
    def _generate_and_score(self, batch):
        prompts = [example["prompt"] for example in batch]
        conversations = [example["conversations"] for example in batch]

        # Replicate each prompt num_generations times
        expanded_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        expanded_conversations = [conv for conv in conversations for _ in range(self.num_generations)]

        inputs = self.processing_class.apply_chat_template(
            expanded_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generated_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, prompt_len:]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for each completion
        rewards = torch.tensor(
            [self.reward_func(prompt, completion) for prompt, completion in zip(expanded_prompts, completions)],
            device=self.model.device
        )

        self.global_step += 1
        # Log model-generated outputs every 10 steps
        if self.global_step % 5 == 0:
            print("="*20)
            print("Step", self.global_step)
            print("-"*20)
            print(expanded_prompts[0])
            print(completions[0])

        return inputs["input_ids"], inputs["attention_mask"], completion_ids, rewards
    
    def compute_loss(self, model, batch, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask, completion_ids, rewards = self._generate_and_score(batch)

        # Reshape tensors to group completions by their original prompt
        batch_size = len(batch)
        prompt_ids = prompt_ids.view(batch_size, self.num_generations, -1)
        prompt_mask = prompt_mask.view(batch_size, self.num_generations, -1)
        completion_ids = completion_ids.view(batch_size, self.num_generations, -1)
        rewards = rewards.view(batch_size, self.num_generations)

        # Log reward statistics
        self.log({
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item()
        })


        # Flatten tensors for model input
        flat_prompt_ids = prompt_ids.view(-1, prompt_ids.size(-1))
        flat_prompt_mask = prompt_mask.view(-1, prompt_mask.size(-1))
        flat_completion_ids = completion_ids.view(-1, completion_ids.size(-1))

        input_ids = torch.cat([flat_prompt_ids, flat_completion_ids], dim=1)
        attention_mask = torch.cat([flat_prompt_mask, torch.ones_like(flat_completion_ids)], dim=1)

        outputs = model(input_ids, attention_mask=attention_mask)
        completion_logits = outputs.logits[:, flat_prompt_ids.shape[1]:, :]
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=flat_completion_ids.unsqueeze(-1)).squeeze(-1)

        # Reshape to compute advantages per group
        target_log_probs = target_log_probs.view(batch_size, self.num_generations, -1)
        mean_log_probs = target_log_probs.mean(dim=-1)

        # Compute advantages
        # advantages = rewards - rewards.mean(dim=1, keepdim=True)
        advantages = rewards

        # Compute loss
        loss = - (mean_log_probs * advantages).mean()
        return loss