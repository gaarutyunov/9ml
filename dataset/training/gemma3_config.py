"""
Gemma 3 configuration for GRPO training.

This file defines training configurations for different environments:
- Colab (free tier): Gemma 3 1B with aggressive memory optimization
- Cloud A100: Gemma 3 4B with full precision options
"""

# ============================================================================
# Colab Configuration (Free Tier - T4 16GB)
# ============================================================================

COLAB_CONFIG = {
    # Model - use smallest Gemma for T4
    "model_name": "unsloth/gemma-3-1b-it",
    "max_seq_length": 2048,  # Reduced for memory
    "load_in_4bit": True,

    # LoRA - smaller rank for memory
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # GRPO - fewer generations
    "num_generations": 2,
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.9,

    # Training - conservative settings
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_steps": 100,
    "logging_steps": 5,
    "save_steps": 25,
    "warmup_steps": 10,

    "output_dir": "./grpo_plan9_colab",
}


# ============================================================================
# Cloud A100 Configuration (40GB+)
# ============================================================================

CLOUD_A100_CONFIG = {
    # Model - larger Gemma for A100
    "model_name": "unsloth/gemma-3-4b-it",
    "max_seq_length": 4096,
    "load_in_4bit": True,  # Can set False for A100 80GB

    # LoRA - larger rank
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # GRPO - more generations for better exploration
    "num_generations": 4,
    "max_new_tokens": 1024,
    "temperature": 0.8,
    "top_p": 0.9,

    # Training - more aggressive
    "learning_rate": 5e-6,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 500,
    "logging_steps": 10,
    "save_steps": 100,
    "warmup_steps": 50,

    "output_dir": "./grpo_plan9_a100",
}


# ============================================================================
# Development Configuration (Quick testing)
# ============================================================================

DEV_CONFIG = {
    "model_name": "unsloth/gemma-3-1b-it",
    "max_seq_length": 1024,
    "load_in_4bit": True,

    "lora_r": 4,
    "lora_alpha": 4,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],

    "num_generations": 2,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.9,

    "learning_rate": 1e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "max_steps": 10,
    "logging_steps": 1,
    "save_steps": 5,
    "warmup_steps": 2,

    "output_dir": "./grpo_plan9_dev",
}


# ============================================================================
# Select Configuration
# ============================================================================

# Default to Colab config - change this based on your environment
CONFIG = COLAB_CONFIG

# Alternatively, uncomment one of these:
# CONFIG = CLOUD_A100_CONFIG
# CONFIG = DEV_CONFIG
