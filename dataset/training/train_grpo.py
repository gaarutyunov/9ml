#!/usr/bin/env python3
"""
GRPO training script for Plan 9 function calling agent.

Uses Unsloth for efficient LoRA fine-tuning and TRL for GRPO training.
Supports both Colab (free tier) and cloud GPU environments.

Usage:
    # Basic training
    python train_grpo.py --model unsloth/gemma-3-1b-it --output ./grpo_plan9

    # With custom config
    python train_grpo.py --config gemma3_config.py

    # Resume from checkpoint
    python train_grpo.py --resume ./grpo_plan9/checkpoint-100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required = ["torch", "transformers", "trl", "datasets"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r training/requirements.txt")
        return False

    return True


def get_default_config() -> dict[str, Any]:
    """Get default training configuration."""
    return {
        # Model settings
        "model_name": "unsloth/gemma-3-1b-it",  # Start small for testing
        "max_seq_length": 4096,
        "load_in_4bit": True,

        # LoRA settings
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],

        # GRPO settings
        "num_generations": 4,  # Number of completions per prompt
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.9,

        # Training settings
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
        "logging_steps": 10,
        "save_steps": 100,
        "warmup_steps": 50,

        # Paths
        "output_dir": "./grpo_plan9",
        "tasks_path": None,  # Use built-in tasks if None
        "qemu_dir": None,  # For VM-based rewards
    }


def load_config(config_path: str | None) -> dict[str, Any]:
    """Load configuration from file or return defaults."""
    config = get_default_config()

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            if config_path.endswith(".json"):
                user_config = json.load(f)
            elif config_path.endswith(".py"):
                # Execute Python config file
                exec(open(config_path).read(), user_config := {})
                user_config = user_config.get("CONFIG", {})
            config.update(user_config)

    return config


def setup_model(config: dict[str, Any]):
    """Set up model with LoRA adapter.

    Returns:
        Tuple of (model, tokenizer).
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Unsloth not found. Using standard transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            load_in_4bit=config["load_in_4bit"],
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Add LoRA with PEFT
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        return model, tokenizer

    # Use Unsloth for efficient loading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def create_dataset(config: dict[str, Any]):
    """Create GRPO training dataset.

    Returns:
        HuggingFace Dataset.
    """
    from datasets import Dataset

    from plan9_dataset.grpo_data import (
        GRPO_TASKS,
        create_grpo_dataset,
        load_grpo_tasks,
    )

    # Load tasks
    if config["tasks_path"]:
        tasks = load_grpo_tasks(Path(config["tasks_path"]))
    else:
        tasks = GRPO_TASKS

    # Create dataset
    data = create_grpo_dataset(tasks)

    return Dataset.from_list(data)


def create_reward_function(config: dict[str, Any]):
    """Create reward function for GRPO.

    Returns:
        Reward function callable.
    """
    qemu_dir = config.get("qemu_dir")

    if qemu_dir and os.path.exists(qemu_dir):
        # Use VM-based rewards
        from plan9_dataset.rewards import create_reward_function as create_vm_reward

        disk_image = os.path.join(qemu_dir, "9front.qcow2")
        shared_image = os.path.join(qemu_dir, "shared.img")

        if os.path.exists(disk_image):
            print(f"Using VM-based rewards from {qemu_dir}")
            return create_vm_reward(disk_image, shared_image)

    # Use simple heuristic rewards (no VM)
    print("Using heuristic rewards (no VM)")
    return create_heuristic_reward_function()


def create_heuristic_reward_function():
    """Create a simple heuristic reward function without VM.

    This is useful for quick testing without QEMU setup.
    """
    from plan9_dataset.tools import parse_tool_calls, parse_thinking

    def reward_function(samples, prompts, outputs, **kwargs):
        rewards = []

        for output in outputs:
            reward = 0.0

            # Parse tool calls
            tool_calls = parse_tool_calls(output)

            # Reward for having tool calls
            if tool_calls:
                reward += 0.5 * len(tool_calls)

            # Reward for valid tool calls
            for call in tool_calls:
                name = call.get("name", "")
                params = call.get("params", {})

                if name == "write_file":
                    content = params.get("content", "")
                    # Check for Plan 9 style
                    if "#include <u.h>" in content:
                        reward += 0.5
                    if "#include <libc.h>" in content:
                        reward += 0.3
                    if "nil" in content and "NULL" not in content:
                        reward += 0.3
                    if "print(" in content and "printf(" not in content:
                        reward += 0.2
                    if "exits(" in content:
                        reward += 0.2
                    # Check for rc script style
                    if "#!/bin/rc" in content:
                        reward += 0.5

                elif name == "run_command":
                    command = params.get("command", "")
                    # Reward for compile commands
                    if "6c" in command and "6l" in command:
                        reward += 0.5
                    # Reward for running compiled binary
                    if "./" in command:
                        reward += 0.3

            # Bonus for reasoning
            if parse_thinking(output):
                reward += 0.5

            rewards.append(reward)

        return rewards

    return reward_function


def train(config: dict[str, Any]):
    """Run GRPO training.

    Args:
        config: Training configuration dict.
    """
    from trl import GRPOConfig, GRPOTrainer

    print("Setting up model...")
    model, tokenizer = setup_model(config)

    print("Creating dataset...")
    dataset = create_dataset(config)
    print(f"Dataset size: {len(dataset)} prompts")

    print("Creating reward function...")
    reward_fn = create_reward_function(config)

    print("Configuring trainer...")
    grpo_config = GRPOConfig(
        num_generations=config["num_generations"],
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=config["max_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        output_dir=config["output_dir"],
        report_to="none",  # Disable wandb etc. for now
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {config['output_dir']}")
    trainer.save_model()

    # Cleanup reward function if it has VM
    if hasattr(reward_fn, "cleanup"):
        reward_fn.cleanup()

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Plan 9 agent")

    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path (default: unsloth/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (JSON or Python)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for model",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Path to tasks.json file",
    )
    parser.add_argument(
        "--qemu-dir",
        type=str,
        help="Path to QEMU directory with disk images",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.model:
        config["model_name"] = args.model
    if args.output:
        config["output_dir"] = args.output
    if args.tasks:
        config["tasks_path"] = args.tasks
    if args.qemu_dir:
        config["qemu_dir"] = args.qemu_dir
    if args.max_steps:
        config["max_steps"] = args.max_steps

    if args.dry_run:
        print("Configuration:")
        print(json.dumps(config, indent=2))
        return

    train(config)


if __name__ == "__main__":
    main()
