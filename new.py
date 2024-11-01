import os
import json
import numpy as np
import matplotlib.pyplot as plt

path = 'simple/animals/an_elephant_and_a_turtle/v-1-4/33/sd'
clip_score = np.load(f"{path}/clip_scores.npy")
vqa_score = np.load(f"{path}/vqa_scores.npy")
plt.plot(range(len(vqa_score)),vqa_score)
plt.plot(range(len(clip_score)),clip_score)
plt.show()
