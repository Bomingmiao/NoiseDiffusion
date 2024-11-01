import os
import json
import torch
import json
from diffusers import DDIMScheduler
import t2v_metrics
from initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline


# ---------
# Arguments
# ---------
SEEDS           = [22,15,9951,31,9989]
SD_VERSION    = "CompVis/stable-diffusion-v1-4"
PROMPT          = "a cat and a rabbit"
token_indices   = [2, 5]
result_root     = "simple"
device = 'cuda:0'


def main():
    vqa_model = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=device)
    vqa_model.model.model.requires_grad_(False)
    clip_model = t2v_metrics.CLIPScore(model='laion2b_s32b_b82k:ViT-L-14', device=device)
    clip_model.model.model.requires_grad_(False)
    pipe = StableDiffusionInitNOPipeline.from_pretrained(SD_VERSION).to("cuda")
    scheduler = DDIMScheduler.from_pretrained(SD_VERSION,subfolder='scheduler',revision=None)
    pipe.scheduler = scheduler
    with open('a.e_prompts.json','r') as f:
        dicts = json.load(f)
    for cls in ['animals','animals_objects','objects']:
        prompts = dicts[cls]
        for dict in prompts:
        # use get_indices function to find out indices of the tokens you want to alter
            PROMPT = dict['prompt']
            if PROMPT in ["a bear and a rabbit",
                                "a lion and a monkey"
                                "a purple crown and a blue bench"]:
                token_indices = dict['indices']
                pipe.get_indices(PROMPT)
                for SEED in SEEDS:

                    print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))

                    generator = torch.Generator("cuda").manual_seed(SEED)
                    images = pipe(
                        prompt=PROMPT,
                        token_indices=token_indices,
                        guidance_scale=7.5,
                        generator=generator,
                        num_inference_steps=50,
                        max_iter_to_alter=25,
                        result_root=result_root,
                        seed=SEED,
                        run_sd=False,
                    ).images

                    image = images[0]
                    dir = f"{result_root}/{cls}/{PROMPT.replace(' ', '_')}/v-1-4/{SEED}/initno"
                    os.makedirs(dir,exist_ok=True)
                    path = f"{dir}/target.png"
                    image.save(f"{dir}/target.png")
                    vqa_score = vqa_model(path, [PROMPT])
                    with torch.no_grad():
                        clip_score = clip_model(path, [PROMPT])
                    score = {'clip_score':clip_score.item(),'vqa_score':vqa_score.item()}
                    with open(f"{dir}/score.json" ,'w') as f:
                        json.dump(score,f)



if __name__ == '__main__':
    main()