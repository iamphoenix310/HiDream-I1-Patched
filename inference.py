import torch
import argparse
import os
from pathlib import Path
from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM  # ✅ Use Huggingface now

# ✅ ARGUMENT PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--resolution", type=str, default="1024x1024")
parser.add_argument("--seed", type=int, default=-1)
args = parser.parse_args()

model_type = args.model_type
prompt = args.prompt
output_path = args.output_path
resolution = args.resolution
seed = args.seed

MODEL_PREFIX = "azaneko"
LLAMA_MODEL_NAME = "TheBloke/LLaMA-Pro-8B-Instruct-GPTQ"
LLAMA_TOKENIZER_NAME = "TheBloke/LLaMA-Pro-8B-Instruct-GPTQ"

MODEL_CONFIGS = {
    "dev": {...},
    "full": {...},
    "fast": {...}
}

def parse_resolution(resolution_str):
    options = {...}
    return options.get(resolution_str, (1024, 1024))

def load_models(model_type):
    print(f"✅ Loading model config: {model_type}")
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("❌ HUGGINGFACE_HUB_TOKEN not found!")

    tokenizer_4 = AutoTokenizer.from_pretrained(
        LLAMA_TOKENIZER_NAME,
        token=token,
        use_fast=True,
        trust_remote_code=True
    )

    text_encoder_4 = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        config["path"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")

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

print(f"🔧 Preparing HiDream with model type: {model_type}")
pipe, _ = load_models(model_type)
print("🖼 Generating image...")
image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)

output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"✅ Image saved to: {output_path} (Seed: {used_seed})")
