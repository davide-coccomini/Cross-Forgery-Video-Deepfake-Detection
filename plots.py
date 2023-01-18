import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import pandas as pd
from utils import approaches, groups
import numpy as np
RESULTS_PATH = "results/frame_level"

# CONSTRUCT THE DATAFRAMES
df_files = os.listdir(RESULTS_PATH)
df_files = sorted(df_files, key=lambda x: int(x.split("_")[1]))
model_names = ["efficientnet", "vit"]
dataframes = {}
for method in range(9):
    # Get the dataframes of the considered method
    method_df_files = [path for path in df_files if "_"+str(method)+"_" in path]

    # Init the keys for dataframes representing the training method
    if method not in dataframes:
        dataframes[method] = {}

    # Init the keys for dataframes[method] representing the selected evaluation metric
    if "Accuracy" not in dataframes[method]:
        dataframes[method]["accuracy"] = {}
        dataframes[method]["f1"] = {} 
        dataframes[method]["accuracy_ignored"] = {}
        dataframes[method]["f1_ignored"] = {} 

    # Store the metrics values for each metric and each model
    for method_df_file in method_df_files:
        metric = method_df_file.split("_")[2]
        if "ignored" in method_df_file:
            metric += "_ignored"
        
        model_name = method_df_file.split("_")[0]
        values = pd.read_csv(os.path.join(RESULTS_PATH, method_df_file)).loc[0, :10].values.flatten().tolist()
        dataframes[method][metric][model_name] = values

    
'''
dataframes = {0: 
                {"accuracy": 
                            { "efficientnet": [...]
                               "vit": [...]
                            }

                 ...
                 "f1_ignored": {...}
                }
            }
            ...
            8: {{...}}
'''

for method in dataframes:
    for metric in dataframes[method]:
        plt.xlabel("Deepfake Generation Method")
        plt.ylabel(metric)
        colors = {'vit': 'royalblue', 'efficientnet': 'firebrick'}
        for model in dataframes[method][metric]:

            model_results = dataframes[method][model]
            plt.xticks(np.arange(0, 9, step=1))
            length = 10
            ranges = [[min(model_results) for i in range(length)], [max(model_results) for i in range(length)]]
            plt.plot(ranges[0], linestyle='dashed', color=colors[model])
            plt.plot(model_results, label = model, color=colors[model])
            plt.plot(ranges[1], linestyle='dashed', color=colors[model])
            


        plt.legend(bbox_to_anchor = (1.05, 0.6))
        plt.title("Training Set: " + method)
        out_path = os.path.join("plots", metric)
        os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, str(method) + ".png"), bbox_inches='tight')
        plt.clf()
