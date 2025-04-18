import torch
import argparse
import os
from pathlib import Path
from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM  # Needed for GPTQ models

# ✅ ARGUMENT PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--resolution", type=str, default="1024x1024")
parser.add_argument("--seed", type=int, default=-1)
args = parser.parse_args()

# ✅ Extract arguments
model_type = args.model_type
prompt = args.prompt
output_path = args.output_path
resolution = args.resolution
seed = args.seed

# ✅ Model Paths
MODEL_PREFIX = "azaneko"
LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

def parse_resolution(resolution_str):
    options = {
        "1024x1024": (1024, 1024),
        "768x1360": (768, 1360),
        "1360x768": (1360, 768),
        "880x1168": (880, 1168),
        "1168x880": (1168, 880),
        "1248x832": (1248, 832),
        "832x1248": (832, 1248),
    }
    return options.get(resolution_str, (1024, 1024))

def load_models(model_type):
    print("✅ Loading model config:", model_type)
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    # ✅ Load Hugging Face token securely
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError("❌ HUGGINGFACE_HUB_TOKEN not found in environment variables!")

    # ✅ Load quantized tokenizer
    tokenizer_4 = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        token=token,
        use_fast=False,
        trust_remote_code=True
    )

    # ✅ Load 4-bit quantized LLaMA model with AutoGPTQ
    text_encoder_4 = AutoGPTQForCausalLM.from_quantized(
        LLAMA_MODEL_NAME,
        token=token,
        use_safetensors=True,
        trust_remote_code=True,
        device="cpu",  # ⬅️ Keep on CPU to avoid VRAM overflow
        torch_dtype=torch.float32
    )

    # ✅ Load HiDream Transformer (quantized NF4)
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        config["path"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # ✅ Load HiDream Image Pipeline
    pipe = HiDreamImagePipeline.from_pretrained(
        config["path"],
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)

    pipe.transformer = transformer
    return pipe, config

def generate_image(pipe, model_type, prompt, resolution, seed):
    config = MODEL_CONFIGS[model_type]
    height, width = parse_resolution(resolution)

    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)

    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images[0], seed

# ✅ MAIN EXECUTION
print(f"🔧 Preparing HiDream with model type: {model_type}")
pipe, _ = load_models(model_type)
print("🖼 Generating image...")
image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)

# ✅ SAVE OUTPUT
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"✅ Image saved to: {output_path} (Seed: {used_seed})")
