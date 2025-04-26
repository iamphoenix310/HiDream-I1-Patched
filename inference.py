import torch
import argparse
import os
from pathlib import Path
from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer  # ✅ Only transformers

# ✅ ARGUMENT PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--resolution", type=str, default="1024x1024")
parser.add_argument("--seed", type=int, default=-1)
args = parser.parse_args()

# ✅ Extract args
model_type = args.model_type
prompt = args.prompt
output_path = args.output_path
resolution = args.resolution
seed = args.seed

# ✅ Model paths
MODEL_PREFIX = "azaneko"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # ✅ updated
LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # ✅ updated

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
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",   # ✅ JUST ADD THIS
        model_max_length=77  # ✅ Important fix
    )
    with torch.no_grad():
        dummy = tokenizer_4(prompt, return_tensors="pt").to("cuda")
        print(f"🧠 Prompt token length: {dummy['input_ids'].shape[1]}")
        if dummy['input_ids'].shape[1] > 77:
            print("⚠️ Prompt too long — may crash due to LLaMA max length. Consider trimming.")
        encoder_out = text_encoder_4(**dummy, output_hidden_states=True)
        print("✅ Encoder output shape:", encoder_out.hidden_states[-1].shape)

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
    # ✅ Debug line to inspect output tensor quality
    print("📸 Image tensor std dev:", images[0].std())
    return images[0], seed

# ✅ Main execution
print(f"🔧 Preparing HiDream with model type: {model_type}")
pipe, _ = load_models(model_type)
print("🖼 Generating image...")
image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)

# ✅ Save output
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"✅ Image saved to: {output_path} (Seed: {used_seed})")
