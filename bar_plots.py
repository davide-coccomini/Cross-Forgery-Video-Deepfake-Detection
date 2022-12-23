import matplotlib.pyplot as plt
import numpy as np

import os

labels = ['Avg_Max', 'Max_Max', 'Max_Top']
results = {"1": {"ViT": [0.509, 0.444, 0.488] , 
                 "EfficientNet": [0.508, 0.482, 0.374]},
           "2": {"ViT": [0.512, 0.495, 0.500] , 
                 "EfficientNet": [0.446, 0.444, 0.427]}, 
           "3": {"ViT": [0.489, 0.479, 0.479] , 
                 "EfficientNet": [0.446, 0.444, 0.505]},  
           "4": {"ViT": [0.476, 0.447, 0.453] , 
                 "EfficientNet": [0.505, 0.439, 0.402]},  
           "6": {"ViT": [0.498, 0.486, 0.485] , 
                 "EfficientNet": [0.508, 0.444, 0.372]},   
           "7": {"ViT": [0.521, 0.500, 0.503], 
                 "EfficientNet": [0.527, 0.444, 0.405]},   
           "8": {"ViT": [0.529, 0.520, 0.518], 
                 "EfficientNet": [0.535, 0.442, 0.419]}}

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
for key in results.keys():
    values = results[key]
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values["ViT"], width, label='ViT-Base')
    rects2 = ax.bar(x + width/2, values["EfficientNet"], width, label='EfficientNet-V2-M')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Accuracies of models trained on method ' + str(key))
    ax.set_xticks(x, labels)

    ax.legend(bbox_to_anchor = (0.65, -0.1))
    ax.bar_label(rects1, padding=1)
    ax.bar_label(rects2, padding=1)

    #fig.tight_layout()


    plt.savefig(os.path.join("plots/bar", key + ".png"), bbox_inches='tight')
    plt.clf()


