import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import pandas as pd
from utils import approaches, groups
import numpy as np
CROPS_DIR = "data/forgerynet/Training/crops"
STATS_DIR = "stats"
LIST_FILE_PATH = "data/forgerynet/training_image_list.txt"
'''
all_classes = dict.fromkeys([i for i in range(1, 20)], 0)
two_classes = dict.fromkeys(["Real", "Fake"], 0)
for i in range(1, 20):
    print("Checking", i)
    tmp = os.path.join(CROPS_DIR, str(i))
    folder_names = os.listdir(tmp)
    for folder_name in folder_names:
        folder_path = os.path.join(tmp, folder_name)
        
        if os.path.isdir(folder_path): # We need to search more
            internal_folders = os.listdir(folder_path)
            
            for internal_folder in internal_folders: # We found files
                internal_path = os.path.join(folder_path, internal_folder)
                if os.path.isdir(internal_path): 
                    file_names = os.listdir(internal_path)
                    all_classes[i] += len(file_names)
                    if i > 15:
                        two_classes["Real"] += len(file_names)
                    else:
                        two_classes["Fake"] += len(file_names)

                else:
                    all_classes[i] += len(internal_folders)
                    if i > 15:
                        two_classes["Real"] += len(internal_folders)
                    else:
                        two_classes["Fake"] += len(internal_folders)
                    break

        else: #  We found the files
            all_classes[i] += len(folder_names)
            if i > 15:
                two_classes["Real"] += len(folder_names)
            else:
                two_classes["Fake"] += len(folder_names)
            break


print(all_classes)
print(two_classes)

names = list(all_classes.keys())
values = list(all_classes.values())
plt.bar(range(len(all_classes)),values, tick_label=names)

plt.savefig(os.path.join(STATS_DIR, 'all_classes.png'))

plt.clf()

names = list(two_classes.keys())
values = list(two_classes.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'two_classes.png'))



df = pd.read_csv(LIST_FILE_PATH, sep=' ', usecols=[0, 1, 3])
binary_counters = dict(df['binary_cls_label'].value_counts())
all_classes_counters = dict(df['16cls_label'].value_counts())
groups_counters = dict.fromkeys(list(groups.keys()), 0)

for key in all_classes_counters:
    for group_key in groups:
        if key in groups[group_key]:
            groups_counters[group_key] += all_classes_counters[key]

print(groups_counters)

names = list(all_classes_counters.keys())
values = list(all_classes_counters.values())
plt.bar(range(len(all_classes_counters)),values, tick_label=names)

plt.savefig(os.path.join(STATS_DIR, 'all_classes_from_csv.png'))

plt.clf()

names = list(binary_counters.keys())
values = list(binary_counters.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'two_classes_from_csv.png'))


plt.clf()

names = list(groups_counters.keys())
values = list(groups_counters.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'groups_from_csv.png'))

plt.clf()


plt.bar(range(len(groups_counters)),values, tick_label=names)
plt.savefig(os.path.join(STATS_DIR, 'two_classes_from_csv_bar.png'))



'''


# LINE PLOTS
results =  {"1": {"ViT": [0.742,0.44,0.448,0.198,0.249,0.226,0.266,0.209,0.496] , 
                 "EfficientNet": [0.129,0.603,0.799,0.635,0.516,0.502,0.613,0.318,0.991]},
            "2": {"ViT": [0.854,0.13,0.878,0.102,0.16,0.128,0.145,0.122,0.389],
                 "EfficientNet": [0.029,0.927,0.995,0.909,0.919,0.848,0.871,0.821,1.0]},
            "3": {"ViT": [0.675,0.322,0.303,0.408,0.31,0.288,0.468,0.289,0.476],
                "EfficientNet": [0.127,0.615,0.793,0.584,0.508,0.479,0.637,0.339,0.988]},
            "4": {"ViT": [0.796,0.18,0.386,0.13,0.256,0.179,0.149,0.09,0.166],
                "EfficientNet": [0,0,0,0,0,0,0,0]},
            "6": {"ViT": [0.726,0.26,0.306,0.287,0.255,0.202,0.496,0.267,0.459],
                "EfficientNet": [0.186,0.491,0.736,0.552,0.38,0.354,0.488,0.249,0.948]},
            "7": {"ViT": [0.745,0.28,0.355,0.214,0.243,0.156,0.343,0.388,0.829],
                "EfficientNet": [0.324,0.474,0.701,0.423,0.405,0.3,0.472,0.268,0.992]},
            "8": {"ViT": [0.889,0.119,0.306,0.091,0.102,0.086,0.129,0.137,0.962],
                "EfficientNet": [0,0,0,0,0,0,0,0]}}
           #"1,2,3": {"ViT": [0.593,0.622,0.603,0.64,0.491,0.517,0.539,0.488,0.51,0.568,0.459,0.497,0.462,0.509,0.367,0.568],
           #             "EfficientNet": [0.798,0.47,0.57,0.811,0.328,0.346,0.476,0.27,0.516,0.477,0.265,0.389,0.441,0.342,0.273,0.507]},
           #"7,8,10": {"ViT": [0.631,0.496,0.471,0.36,0.431,0.465,0.57,0.741,0.775,0.455,0.627,0.577,0.455,0.416,0.41,0.5],
           #             "EfficientNet": [0.716,0.398,0.448,0.212,0.377,0.277,0.32,0.674,0.801,0.394,0.43,0.423,0.322,0.385,0.353,0.489]}}

for method in results:
    plt.xlabel("Deepfake Generation Method")
    plt.ylabel("Accuracy")
    
    
    vit_results = results[method]["ViT"]
    plt.xticks(np.arange(0, 9, step=1))
    length = len(vit_results)
    ranges = [[min(vit_results) for i in range(length)], [max(vit_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='royalblue')
    plt.plot(vit_results, label = "ViT-Base", color='royalblue')
    plt.plot(ranges[1], linestyle='dashed', color='royalblue')
    
    
    efficientnet_results = results[method]["EfficientNet"]
    ranges = [[min(efficientnet_results) for i in range(length)], [max(efficientnet_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='firebrick')
    plt.plot(efficientnet_results, label="EfficientNetV2-M", color='firebrick')
    plt.plot(ranges[1], linestyle='dashed', color='firebrick')

    '''
    hybrid_mean_results = results[method]["Hybrid (Mean)"]
    ranges = [[min(hybrid_mean_results) for i in range(length)], [max(hybrid_mean_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='green')
    plt.plot(hybrid_mean_results, label="Hybrid (Mean)", color='green')
    plt.plot(ranges[1], linestyle='dashed', color='green')
    '''
    


    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.title("Training Set: " + method)
    plt.savefig(os.path.join("plots", method + ".png"), bbox_inches='tight')
    plt.clf()
