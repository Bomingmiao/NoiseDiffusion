import torch
import t2v_metrics
from pipeline_stable_diffusion_nd import StableDiffusionNDPipeline

sd_device = "cuda:0"
vqa_model_device = "cuda:0"
pipe = StableDiffusionNDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to(sd_device)
vqa_model = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=vqa_model_device)
pipe.init_vqa_model(vqa_model, vqa_model_device)

prompt = "a book on the sofa"
seed = 33

generator = torch.Generator(sd_device).manual_seed(seed)
image = pipe( prompt=prompt,
        num_inference_steps=50,
        generator=generator,
        optimization_epoch=1
        ).images

