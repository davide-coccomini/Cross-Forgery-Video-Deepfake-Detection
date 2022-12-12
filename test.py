
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
from utils import custom_round

from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from sklearn.model_selection import train_test_split

from timm.models.efficientnet import tf_efficientnet_b7_ns

from sklearn.metrics import accuracy_score
from statistics import mean

import timm
from transformers import ViTForImageClassification, ViTConfig


def save_confusion_matrix(confusion_matrix):
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, cmap="Blues")

  threshold = im.norm(confusion_matrix.max())/2.
  textcolors=("black", "white")

  ax.set_xticks(np.arange(2))
  ax.set_yticks(np.arange(2))
  ax.set_xticklabels(["original", "fake"])
  ax.set_yticklabels(["original", "fake"])
  
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(2):
      for j in range(2):
          text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                         fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, "confusion.jpg"))


  
def read_frames(paths, dataset, opt, image_size, aggregation=None):
    transform = create_base_transform(image_size)
    for path in paths:
        video_path = os.path.join(opt.data_path, path[0])
        #video_path_tmp = os.path.dirname(path[0]).split("release")[1]
        #video_path = opt.data_path + os.path.sep + "crops" + video_path_tmp
        if not os.path.exists(video_path):
            return
        

        identity_names = os.listdir(video_path)
        row = []
        identity_images = []
        identity_paths = []
        for identity in identity_names:
            identity_path = video_path + os.sep + str(identity) + os.sep
            if not os.path.isdir(identity_path):
                continue

            frames_names = os.listdir(identity_path)
            label = path[1]
            images = []
            images_paths = []
            for frame_name in frames_names:
                image_path = os.path.join(identity_path, frame_name)
                image = transform(image=cv2.imread(image_path))['image']
                images.append(image)
                images_paths.append(image_path)
            images = np.asarray(images)
            identity_images.append(images)
            identity_paths.append(images_paths)

        if len(identity_images) == 0:
            continue

        row = (identity_images, label, identity_paths)
        dataset.append(row)


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def top_aggregation(values, top=5):
    values = sorted(values, reverse=True)
    return mean(values[:top])



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
    parser.add_argument('--data_path', default='../../MINTIME/datasets/ForgeryNet/faces', type=str,
                        help='Videos directory')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for validation (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--list_file', default="../../MINTIME/datasets/ForgeryNet/faces/val.csv", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: ViT, 1: EfficientNet B7, 2: Hybrid)")
    parser.add_argument('--forgery_method', type=int, default=1, 
                        help="Forgery method used for training")
    parser.add_argument('--aggregation_method', type=str, default="max_max", 
                        help="(max_max, avg_max, avg_avg, max_top)")
    parser.add_argument('--save_errors', type=bool, default=False, 
                        help="Save errors in directory?")
    opt = parser.parse_args()
    print(opt)

    
    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    
    batch_size = 1

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


    df = pd.read_csv(opt.list_file, sep=' ', names=["image_name", "label", "16cls_label"])
     
    results_df = pd.DataFrame(columns=[i for i in range(14)], index=[0])
    fake_accuracy = 0
    real_accuracy = 0
    values = []
    if opt.aggregation_method == "max_max":
        first_aggregation = max
        second_aggregation = max
    elif opt.aggregation_method == "avg_max":
        first_aggregation = mean
        second_aggregation = max
    elif opt.aggregation_method == "avg_avg":
        first_aggregation = mean
        second_aggregation = mean
    else:
        first_aggregation = top_aggregation
        second_aggregation = max

    os.makedirs(os.path.join("results", opt.aggregation_method), exist_ok=True)
    print("FORGERY METHOD", opt.forgery_method, "MODEL", opt.model_path)
    for forgery_method in range(0, 9):
        f = open(os.path.join("results", opt.aggregation_method, opt.model_name + "_" + str(opt.forgery_method) + "_" + str(forgery_method) + "_labels.txt"), "w+")
        df_tmp = df.loc[(df["16cls_label"] == forgery_method)]
        if opt.max_videos > -1:
            df = df.head(opt.max_videos)
        df_tmp = df_tmp.sort_values(by=['image_name'])
        paths = df_tmp.to_numpy()
        paths = np.array_split(paths, opt.workers) # Split the paths in chunks for processing
        
        mgr = Manager()
        dataset = mgr.list()

        with Pool(processes=cpu_count()) as p:
            with tqdm(total=len(paths)) as pbar:
                for v in p.imap_unordered(partial(read_frames, dataset=dataset, opt=opt, image_size = config['model']['image-size']), paths):
                    pbar.update()

        labels = [float(row[1]) for row in dataset]
        face_paths = [row[2] for row in dataset]
        dataset = [row[0] for row in dataset]
        model = model.to(opt.gpu_id)
        videos_preds = []
        bar = Bar('Predicting', max=int(len(dataset)/batch_size))

        for i in range(0, len(dataset)):
            identities = dataset[i]
            identities_preds = []
            for faces in identities:     
                if faces.shape[0] == 0:
                    continue
                   
                faces = np.transpose(faces, (0, 3, 1, 2))    
                faces = torch.tensor(np.asarray(faces))
                
                faces = faces.to(opt.gpu_id).float()
                pred = model(faces)
                if opt.model == 0:
                    pred = pred.logits
                pred = pred.cpu().detach()
                faces = faces.cpu().detach()
                identity_preds = [] 
                for idx, p in enumerate(pred):
                    identity_preds.append(torch.sigmoid(p).numpy()[0])
            

                identities_preds.append(identity_preds)

            #print("IDENTITIES PREDS", identities_preds)

            # Get the max identity
            for identity in range(len(identities_preds)):
                identities_preds[identity] = first_aggregation(identities_preds[identity])

            
            #print("After max identities preds", identities_preds)
            #print("VIDEO PRED", max(identities_preds))
            #print("EVALUATION", custom_round(max(identities_preds)))
            #print()
            #print()
            videos_preds.extend([second_aggregation(identities_preds)])
            bar.next()

        final_preds = []
        correct_labels = []

        for i in range(len(labels)):
            current_label = labels[i]
            video_pred = custom_round(videos_preds[i])
            final_preds.append(video_pred)
            correct_labels.append(current_label)
            f.write(" --> " + str(video_pred) + "(CORRECT: " + str(current_label) + ")" +"\n")

        current_accuracy = accuracy_score(correct_labels, final_preds)
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
    results_df.to_csv(os.path.join("results", opt.aggregation_method, opt.model_name + "_" + str(opt.forgery_method) + "_metrics.csv"))
    bar.finish()

    