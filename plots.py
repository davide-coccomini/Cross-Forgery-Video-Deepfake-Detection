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

results_video_level_max_top =  {"1": {"ViT": [0.674,0.415,0.413,0.275,0.314,0.291,0.326,0.251,0.446] , 
                                    "EfficientNet": [0.193,0.623,0.755,0.615,0.572,0.473,0.667,0.324,0.974]},
                                "2": {"ViT": [0.742,0.236,0.621,0.186,0.236,0.218,0.242,0.191,0.392],
                                    "EfficientNet": [0.039,0.933,0.988,0.906,0.916,0.867,0.895,0.845,0.993]},
                                "3": {"ViT": [0.624,0.375,0.343,0.4,0.323,0.364,0.393,0.347,0.459],
                                    "EfficientNet": [0.18,0.672,0.775,0.625,0.593,0.473,0.684,0.349,0.958]},
                                "4": {"ViT": [0.536,0.402,0.582,0.345,0.449,0.406,0.404,0.305,0.45],
                                    "EfficientNet": [0.336,0.52,0.724,0.512,0.491,0.345,0.565,0.244,0.824]},
                                "6": {"ViT": [0.718,0.294,0.292,0.26,0.245,0.194,0.34,0.265,0.38],
                                    "EfficientNet": [0.264,0.563,0.684,0.541,0.46,0.382,0.53,0.238,0.923]},
                                "7": {"ViT": [0.711,0.286,0.364,0.268,0.236,0.218,0.281,0.379,0.628],
                                    "EfficientNet": [0.328,0.535,0.714,0.465,0.46,0.37,0.544,0.293,0.959]},
                                "8": {"ViT": [0.801,0.198,0.312,0.169,0.164,0.17,0.204,0.22,0.679],
                                    "EfficientNet": [0.433,0.429,0.588,0.377,0.356,0.285,0.449,0.215,0.956]}}


results_frame_level =  {"1": {"ViT": [0.654,0.55,0.502,0.624,0.507,0.056,0.239,0.692,0.287] , 
                            "EfficientNet": [0.859,0.502,0.322,0.341,0.555,0.806,0.494,0.459,0.382]},
                        "2": {"ViT": [0.767,0.608,0.707,0.341,0.555,0.487,0.258,0.599,0.428],
                            "EfficientNet": [0.935,0.558,0.59,0.46,0.617,0.806,0.506,0.646,0.285]},
                        "3": {"ViT": [0.584,0.505,0.352,0.341,0.616,0.487,0.513,0.665,0.571],
                            "EfficientNet": [0.946,0.558,0.147,0.385,0.678,0.806,0.506,0.646,0.238]},
                        "4": {"ViT": [0.465,0.611,0.616,0.385,0.63,0.625,0.258,0.662,0.38],
                            "EfficientNet": [0.805,0.766,0.381,0.221,0.815,0.625,0.735,0.569,0.143]},
                        "6": {"ViT": [0.687,0.6,0.352,0.58,0.801,0.487,0.506,0.635,0.285],
                            "EfficientNet": [0.962,0.558,0.088,0.46,0.678,0.806,0.01,0.599,0.048]},
                        "7": {"ViT": [0.897,0.716,0.15,0.46,0.617,0.806,0.506,0.401,0.524],
                            "EfficientNet": [0.324,0.474,0.701,0.423,0.405,0.3,0.472,0.268,0.992]},
                        "8": {"ViT": [0.859,0.558,0.264,0.504,0.63,0.806,0.265,0.505,0.477],
                            "EfficientNet": [0.805,0.766,0.381,0.221,0.815,0.625,0.735,0.569,0.143]}}


results = results_frame_level
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
