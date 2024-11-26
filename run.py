import os
import numpy as np
import torch
from torchvision import transforms
import argparse
import t2v_metrics
from utils import *




def main(args):
    prompt = args.prompt
    to_pil = transforms.ToPILImage()
    detail = [prompt]
    vqa_model = t2v_metrics.VQAScore(model=args.vqa_model,device=args.device)
    vqa_model.model.model.requires_grad_(False)
    text_encoder, tokenizer, unet, vae, ddim_scheduler = load_checkpoint(args.sd_version,args.device)

    for seed in args.seeds:
        dir = f"{args.result_root}/{prompt.replace(' ', '_')}/{seed}"
        os.makedirs(dir, exist_ok=True)
        img, latents, all_latents,noise_list,_ = generate_img(prompt,tokenizer,text_encoder,vae,unet,ddim_scheduler,latents = None,size=512,seed=seed,num_inference_steps=args.num_denoising_steps,guidance_scale=7.5,device=args.device)
        target = to_pil(img.squeeze(0))
        target.save(f'{dir}/0.png')
        latent_init = all_latents[0].detach()
        latent_init.requires_grad = True
        latent = reverse(latent_init, noise_list, ddim_scheduler)
        img = latent2img(vae, latent)
        vqa_score = vqa_model(img, detail)
        max_vqa_score = vqa_score.item()
        for i in range(args.max_epoch):
            vqa_score.backward()
            step_lr = 1 - vqa_score.item()**(1/2)
            grad = latent_init.grad
            noise_pool = torch.randn(args.num_candidate_noises, *grad.shape).to(grad.device)
            sims = []
            for noise in noise_pool:
                vec = ((1 - step_lr) ** 0.5 - 1) * latent_init.detach() + step_lr ** 0.5 * noise
                sim = (vec  * grad ).sum() / vec.norm(2)**2
                sims.append(sim.item())
            noise = noise_pool[np.argmax(sims)]
            latent_tmp =  (1-step_lr)**0.5 * latent_init + step_lr**0.5 * noise
            latent_init = latent_tmp.detach()
            _, _, _,noise_list,_ = generate_img(prompt,tokenizer,text_encoder,vae,unet,ddim_scheduler,latents = latent_init,size=512,seed=seed,num_inference_steps=args.num_denoising_steps,guidance_scale=7.5,device=args.device)
            latent_init.requires_grad = True
            latent = reverse(latent_init, noise_list, ddim_scheduler)
            img = latent2img(vae, latent)
            vqa_score = vqa_model(img, detail)
            image = to_pil(img.squeeze(0))
            image.save(f'{dir}/{i+1}.png')
            if vqa_score > max_vqa_score:
                target = image
                max_vqa_score = vqa_score.item()
        del vqa_score
        target.save(f'{dir}/target.png')
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Execute the Noise Diffusion Process')
    parser.add_argument('--prompt', type=str, default='a book on the sofa', help='The prompt to generate the image')
    parser.add_argument('--seed', type=int, nargs='+', default=[33, 42], help='List of seeds to generate the image')
    parser.add_argument('--sd_version', type=str, default='CompVis/stable-diffusion-v1-4', help='The version of the stable diffusion model')
    parser.add_argument('--vqa_model', type=str, default='clip-flant5-xxl', help='The version of the VQA model')
    parser.add_argument('--num_candidate_noises', type=int, default=50, help='The number of candidate noises')
    parser.add_argument('--num_denoising_steps', type=int, default=50, help='The number of denoising steps')
    parser.add_argument('--max_epoch', type=int, default=50, help='The maximum number of epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the process')
    parser.add_argument('--result_root', type=str, default='results', help='The root directory to save the results')
    args = parser.parse_args()
    main(args)