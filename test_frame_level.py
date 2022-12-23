
import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import shutil
import cv2
import numpy as np
import math
import tensorflow as tf
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from utils import custom_round_single_value
import collections
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from sklearn.model_selection import train_test_split
from deepfakes_dataset import DeepFakesDataset

from timm.models.efficientnet import tf_efficientnet_b7_ns
from progress.bar import ChargingBar

from sklearn.metrics import accuracy_score
from statistics import mean

import timm
from transformers import ViTForImageClassification, ViTConfig

IMAGE_SIZE = 224

def read_images(paths, dataset, opt):
    for path in paths:
        #video_path_tmp = os.path.dirname(path[0]).split("release")[1]  
        image_path = os.path.join(opt.data_path, path[0])
        label = path[1]
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        row = (image, label)
        dataset.append(row)



# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=100, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--gpu_id', default=1, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--data_path', default='../../datasets/deepfakes/ForgeryNet', type=str,
                        help='Videos directory')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for validation (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--list_file', default="../../datasets/deepfakes/ForgeryNet/validation_video_image_list.csv", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: ViT, 1: EfficientNet B7, 2: Hybrid)")
    parser.add_argument('--forgery_method', type=int, default=1, 
                        help="Forgery method used for training")
    parser.add_argument('--save_errors', type=bool, default=False, 
                        help="Save errors in directory?")
    opt = parser.parse_args()
    print(opt)

    
    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    
    batch_size = 256

    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    if opt.model == 0: 
        if opt.use_pretrained:
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=1)
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
    elif opt.model == 1: 
        if opt.use_pretrained:
            model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=1)
        else:
            model = tf_efficientnet_b7_ns(num_classes=1, pretrained=False)
            #model = effnetv2_m()

    elif opt.model == 2: 
        if opt.use_pretrained:
            print("Pretrained network not found for this model, working from scratch.")

        #model = coatnet_2()
    elif opt.model == 3: 
        if opt.use_pretrained:
            print("Pretrained network not found for this model, working from scratch.")

    model.load_state_dict(torch.load(opt.model_path))

    model.eval()
    df = pd.read_csv(opt.list_file, sep=' ')
     
    results_df = pd.DataFrame(columns=[i for i in range(14)], index=[0])
    fake_accuracy = 0
    real_accuracy = 0
    values = []

    os.makedirs(os.path.join("results/frame_level",), exist_ok=True)
    print("FORGERY METHOD", opt.forgery_method, "MODEL", opt.model_path)
    for forgery_method in range(0, 9):
        f = open(os.path.join("results/frame_level", opt.model_name + "_" + str(opt.forgery_method) + "_" + str(forgery_method) + "_labels.txt"), "w+")
        df_tmp = df.loc[(df["16cls_label"] == forgery_method)]
        if opt.max_videos > -1:
            df = df.head(opt.max_videos)
        df_tmp = df_tmp.sort_values(by=['image_name'])
        paths = df_tmp.to_numpy()
        paths = np.array_split(paths, opt.workers) # Split the paths in chunks for processing
        mgr = Manager()
        dataset = mgr.list()

        with Pool(processes=opt.workers) as p:
            with tqdm(total=len(paths)) as pbar:
                for v in p.imap_unordered(partial(read_images, dataset=dataset, opt=opt), paths):
                    pbar.update()

        dataset = sorted(dataset, key=lambda tup: tup[1])
        correct_labels = [float(row[1]) for row in dataset]
        dataset = [row[0] for row in dataset]
        model = model.to(opt.gpu_id)


        print("__VALIDATION STATS__")
        counters = collections.Counter(correct_labels)
        print(counters)
        print("___________________")

        
        dataset = DeepFakesDataset(np.asarray(dataset), np.asarray(correct_labels), IMAGE_SIZE, mode='validation')
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                        pin_memory=False, drop_last=False, timeout=0,
                                        worker_init_fn=None, prefetch_factor=16,
                                        persistent_workers=False)

        total_correct = 0
        preds = []
        
        bar = ChargingBar('Predicting', max=(len(dl)*batch_size))
        for index, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images = np.transpose(images, (0, 3, 1, 2))
                labels = labels.unsqueeze(1) 
                images = images.to(opt.gpu_id).float()
                
                y_pred = model(images)
                if opt.model == 0:
                    y_pred = y_pred.logits

                y_pred = y_pred.cpu()
                preds.extend([custom_round_single_value(torch.sigmoid(y_pred).numpy()[0]) for pred in y_pred])

                for i in range(batch_size):
                    bar.next()

    
        current_accuracy = accuracy_score(correct_labels, preds)
        if forgery_method == 0:
            real_accuracy = current_accuracy
        else:
            fake_accuracy += current_accuracy

        string = "ACCURACY: " + str(current_accuracy)
        print("Method", forgery_method, string)
        f.write(string)
        results_df[forgery_method] = round(current_accuracy, 3)
        values.append(round(current_accuracy, 3))

    fake_accuracy /= 9
    global_accuracy = mean([fake_accuracy, real_accuracy])

    results_df.insert(10, "real_accuracy", real_accuracy)
    results_df.insert(11, "fake_accuracy", fake_accuracy)
    results_df.insert(12, "global_accuracy", global_accuracy)
    results_df.insert(13, "variance", np.var(values))
    print(results_df)
    f.close()
    results_df.to_csv(os.path.join("results/frame_level", opt.model_name + "_" + str(opt.forgery_method) + "_metrics.csv"))
