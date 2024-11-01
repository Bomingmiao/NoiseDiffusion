import t2v_metrics
import torch
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel,DDIMScheduler,DDPMScheduler,PNDMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def generate_logo(prompt,tokenizer,text_encoder,vae,unet,scheduler,latents= None,uncond_embeddings_list=None,seed=33,size=512, num_inference_steps=50,guidance_scale=7.5,start_step=0,device='cuda:0'):
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
        #noise_pred = unet(latents_input,t,encoder_hidden_states=context)['sample']
        if uncond_embeddings_list is not None:
            context = torch.cat([uncond_embeddings_list[i+start_step],cond_embeddings])
        noise_pred = unet(latents_input,t,context)['sample']
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_cond - noise_pred_uncond)
        noise_list.append(noise_pred.detach())
        latents = scheduler.step(noise_pred,t,latents)['prev_sample']
        all_latents.append(latents.clone().detach())
    latents_last = 1/vae.config.scaling_factor *latents
    logos = vae.decode(latents_last)['sample']
    logos = logos * 0.5 + 0.5
    logos = logos.clamp(0,1)
    return logos, latents, all_latents,noise_list,uncond_embeddings
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
    # Load scheduler and models
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

def main():
    device = 'cuda:0'
    result_root='simple'
    to_pil = transforms.ToPILImage()
    vqa_model = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=device)
    #vqa_model = t2v_metrics.VQAScore(model='llava-v1.5-13b')
    #vqa_model = t2v_metrics.VQAScore(model='instructblip-flant5-xxl')
    vqa_model.model.model.requires_grad_(False)
    clip_model = t2v_metrics.CLIPScore(model='laion2b_s32b_b82k:ViT-L-14',device=device)
    clip_model.model.model.requires_grad_(False)
    version_list = {"v-1-4":"CompVis/stable-diffusion-v1-4",
                       "v-1-5":"stable-diffusion-1-5",
                       "v-2-0":"stable-diffusion-2",
                       "v-2-1":"stable-diffusion-2-1"}
    versions = ["v-1-4"]
    #seeds = [33, 5599, 9423 , 7409, 4416, 2359, 1142, 5266, 5696, 2754, 4161]
    seeds  = [22,15,9951,31,9989]
    for version in versions:
        checkpoint_path = version_list[version]
        text_encoder, tokenizer, unet, vae, ddim_scheduler = load_checkpoint(checkpoint_path,device)
        #with open('new_prompts.json', 'r') as f:
        #    prompts = json.load(f)
        #prompts = [dict['prompt']for dict in prompts]

        with open('a.e_prompts.json', 'r') as f:
            dicts = json.load(f)
        for cls in ['animals', 'animals_objects', 'objects']:
            prompts = dicts[cls]
            for dict in prompts:
                prompt = dict['prompt']
                if prompt in ["a bear and a rabbit",
                                "a lion and a monkey"
                                "a purple crown and a blue bench"]:
                    detail = [prompt]
                    for seed in seeds:
                        dir = f"{result_root}/{cls}/{prompt.replace(' ', '_')}/{version}/{seed}/sd"
                        os.makedirs(dir, exist_ok=True)
                        logo, latents, all_latents,noise_list,_ = generate_logo(prompt,tokenizer,text_encoder,vae,unet,ddim_scheduler,latents = None,size=512,seed=seed,num_inference_steps=50,guidance_scale=7.5,device=device)
                        target = to_pil(logo.squeeze(0))
                        target.save(f'{dir}/0.png')
                        latent_init = all_latents[0].detach()
                        latent_init.requires_grad = True
                        latent = reverse(latent_init, noise_list, ddim_scheduler)
                        logo = latent2img(vae, latent)
                        vqa_score = vqa_model(logo, detail)
                        with torch.no_grad():
                            clip_score = clip_model(logo, detail)
                        max_vqa_score = vqa_score.item()
                        max_clip_score = clip_score.item()
                        vqa_scores = [vqa_score.item()]
                        clip_scores = [clip_score.item()]
                        max_vqa_scores = [max_vqa_score]
                        max_clip_scores = [max_clip_score]
                        for i in range(50):
                            vqa_score.backward()
                            step_lr = 1 - vqa_score.item()**(1/2)


                            grad = latent_init.grad
                            noise_pool = torch.randn(50, *grad.shape).to(grad.device)
                            sims = []
                            for noise in noise_pool:
                                vec = ((1 - step_lr) ** 0.5 - 1) * latent_init.detach() + step_lr ** 0.5 * noise
                                sim = (vec  * grad ).sum() / vec.norm(2)**2
                                sims.append(sim.item())
                            noise = noise_pool[np.argmax(sims)]


                            latent_tmp = latent_init.detach()
                            latent_tmp =  (1-step_lr)**0.5 * latent_tmp + step_lr**0.5 * noise


                            latent_init = latent_tmp.detach()
                            _, _, _,noise_list,_ = generate_logo(prompt,tokenizer,text_encoder,vae,unet,ddim_scheduler,latents = latent_init,size=512,seed=seed,num_inference_steps=50,guidance_scale=7.5,device=device)
                            latent_init.requires_grad = True

                            latent = reverse(latent_init, noise_list, ddim_scheduler)
                            logo = latent2img(vae, latent)
                            vqa_score = vqa_model(logo, detail)
                            with torch.no_grad():
                                clip_score = clip_model(logo, detail)
                            image = to_pil(logo.squeeze(0))
                            if vqa_score > max_vqa_score:
                                target = image
                                max_vqa_score = vqa_score.item()
                            if clip_score > max_clip_score:
                                max_clip_score = clip_score.item()
                            vqa_scores.append(vqa_score.item())
                            max_vqa_scores.append(max_vqa_score)
                            clip_scores.append(clip_score.item())
                            max_clip_scores.append(max_clip_score)
                            image.save(f'{dir}/{i+1}.png')
                        target.save(f'{dir}/target.png')
                        np.save(f'{dir}/vqa_scores.npy', np.array(vqa_scores))
                        np.save(f'{dir}/max_vqa_scores.npy', np.array(max_vqa_scores))
                        np.save(f'{dir}/clip_scores.npy', np.array(clip_scores))
                        np.save(f'{dir}/max_clip_scores.npy', np.array(max_clip_scores))
                        del clip_score
                        del vqa_score


if __name__=='__main__':
    main()