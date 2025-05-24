import torch
import t2v_metrics
from pipeline_stable_diffusion_3_nd import StableDiffusion3NDPipeline

sd_device = "cuda:0"
vqa_model_device = "cuda:1"
pipe = StableDiffusion3NDPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
pipe.to(sd_device)
vqa_model = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=vqa_model_device)
pipe.init_vqa_model(vqa_model, vqa_model_device)

prompt = "a drum on a firetruck"
seed = 50

generator = torch.Generator(sd_device).manual_seed(seed)
image = pipe( prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=512,
        width=512,
        guidance_scale=7.0,
        generator=generator,
        optimization_epoch=50
        ).images

