import torch
from diffusers import AutoencoderKL, UNet2DConditionModel,DDIMScheduler,DDPMScheduler,PNDMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from tqdm import tqdm

def generate_img(prompt,tokenizer,text_encoder,vae,unet,scheduler,latents= None, seed=33,size=512, num_inference_steps=50,guidance_scale=7.5,start_step=0,device='cuda:0'):
    device = torch.device(device)
    uncond_input = tokenizer(
        "",
        truncation=True,
        padding="max_length",
        max_length = tokenizer.model_max_length,
        return_tensors = "pt",
    ).input_ids.to(device)

    cond_input = tokenizer(
        prompt,
        truncation = True,
        padding="max_length",
        max_length = tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    uncond_embeddings = text_encoder(uncond_input)[0]
    cond_embeddings = text_encoder(cond_input)[0]
    context = torch.cat([uncond_embeddings,cond_embeddings])
    scheduler.set_timesteps(num_inference_steps)
    if latents is None:
        generator = torch.Generator(device).manual_seed(seed)
        latents = torch.randn(1, unet.config.in_channels, size//8, size//8,generator = generator, device = device)
    latents_init = latents.clone().detach()
    noise_list = []
    all_latents = [latents_init]
    for i,t in tqdm(enumerate(scheduler.timesteps[start_step:]),total=len(scheduler.timesteps[start_step:]),desc="generating images with prompts"):
        latents_input = torch.cat([latents]*2)
        noise_pred = unet(latents_input,t,context)['sample']
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_cond - noise_pred_uncond)
        noise_list.append(noise_pred.detach())
        latents = scheduler.step(noise_pred,t,latents)['prev_sample']
        all_latents.append(latents.clone().detach())
    latents_last = 1/vae.config.scaling_factor *latents
    imgs = vae.decode(latents_last)['sample']
    imgs = imgs * 0.5 + 0.5
    imgs = imgs.clamp(0,1)
    return imgs, latents, all_latents,noise_list,uncond_embeddings
def reverse(latent, noise_list,scheduler,start_step=0):
    for i,t in tqdm(enumerate(scheduler.timesteps[start_step:]),total = len(scheduler.timesteps[start_step:]),desc="reverse noisy sample with predicted noise"):
        noise = noise_list[i]
        latent = scheduler.step(noise,t,latent)['prev_sample']
    return latent
def latent2img(vae,latent):
    latent = 1/vae.config.scaling_factor * latent
    img = vae.decode(latent)['sample']
    img = img *0.5 + 0.5
    img = img.clamp(0,1)
    return img
def load_checkpoint(checkpoint_path,device):
    text_encoder = CLIPTextModel.from_pretrained(
        checkpoint_path,
        subfolder="text_encoder",
        revision=None,
    ).requires_grad_(False)
    text_encoder.to(torch.device(device))

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint_path, subfolder="unet", revision=None
    ).requires_grad_(False)
    unet.to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    vae = AutoencoderKL.from_pretrained(
        checkpoint_path, subfolder="vae", revision=None
    ).requires_grad_(False)
    vae.to(torch.device(device))
    ddim_scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler", revision=None)
    return text_encoder, tokenizer, unet, vae, ddim_scheduler