import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
'''
dir = 'results'
list = os.listdir(dir)
path = [f'{dir}/{p}/v-2-1/33'for p in list]
total_clip_scores = []
total_vqa_scores = []
total_max_clip_scores = []
total_max_vqa_scores = []
for p in path:
    clip_scores = os.path.join(p,'clip_scores.npy')
    vqa_scores = os.path.join(p,'vqa_scores.npy')
    max_clip_scores = os.path.join(p,'max_clip_scores.npy')
    max_vqa_scores = os.path.join(p, 'max_vqa_scores.npy')
    clip_scores = np.load(clip_scores)
    vqa_scores = np.load(vqa_scores)
    max_clip_scores = np.load(max_clip_scores)
    max_vqa_scores = np.load(max_vqa_scores)

    total_clip_scores.append(clip_scores)
    total_max_clip_scores.append(max_clip_scores)
    total_vqa_scores.append(vqa_scores)
    total_max_vqa_scores.append(max_vqa_scores)
total_clip_scores = np.array(total_clip_scores)
total_vqa_scores = np.array(total_vqa_scores)
total_max_clip_scores = np.array(total_max_clip_scores)
total_max_vqa_scores = np.array(total_max_vqa_scores)

avg_clip_scores = total_clip_scores.mean(axis=0)
avg_vqa_scores = total_vqa_scores.mean(axis=0)
avg_max_clip_scores = total_max_clip_scores.mean(axis=0)
avg_max_vqa_scores = total_max_vqa_scores.mean(axis=0)


plt.plot(range(len(avg_clip_scores)),avg_clip_scores)
plt.plot(range(len(avg_max_clip_scores)),avg_max_clip_scores)
plt.plot(range(len(avg_vqa_scores)),avg_vqa_scores)
plt.plot(range(len(avg_max_vqa_scores)),avg_max_vqa_scores)
plt.show()
'''
seeds = [453,462]
root = 'simple'
clip_scores = []
vqa_scores = []
max_clip_scores = []
max_vqa_scores = []
for cls in ['animals','animals_objects','objects'][2:3]:
    dir = f"{root}/{cls}"
    for prompt in os.listdir(dir):

        if prompt.replace('_',' ') in ["a purple crown and a blue bench"]:
            for seed in seeds:
                file = f"{dir}/{prompt}/v-1-4/{seed}/initno/score.json"
                with open(file,'r') as f:
                    score = json.load(f)
                clip_scores.append(score['clip_score'])
                vqa_scores.append(score['vqa_score'])
                file = f"{dir}/{prompt}/v-1-4/{seed}/sd"
                max_clip_score = np.load(f"{file}/max_clip_scores.npy")
                max_vqa_score = np.load(f"{file}/max_vqa_scores.npy")
                max_clip_scores.append(max_clip_score)
                max_vqa_scores.append(max_vqa_score)

max_clip_scores = np.array(max_clip_scores).mean(axis=0).astype(np.float32)
max_vqa_scores = np.array(max_vqa_scores).mean(axis=0).astype(np.float32)
clip_scores = np.array(clip_scores).mean()
vqa_scores = np.array(vqa_scores).mean()
clip_scores = np.array([clip_scores]*len(max_clip_scores))
vqa_scores = np.array([vqa_scores]*len(max_vqa_scores))
plt.plot(range(len(max_clip_scores)),max_clip_scores)
plt.plot(range(len(max_vqa_scores)),max_vqa_scores)
plt.plot(range(len(clip_scores)),clip_scores)
plt.plot(range(len(vqa_scores)),vqa_scores)
plt.show()

