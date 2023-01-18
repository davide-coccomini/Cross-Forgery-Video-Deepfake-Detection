import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import cv2
import numpy as np
import math
import tensorflow as tf
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything

import timm
from timm.models.efficientnet import tf_efficientnet_b7_ns
from timm.scheduler.cosine_lr import CosineLRScheduler


from sklearn.model_selection import train_test_split
import collections
from deepfakes_dataset import DeepFakesDataset
#from torch.optim import lr_scheduler
from progress.bar import ChargingBar
from utils import check_correct, resize, get_n_params, custom_round_single_value
from transformers import ViTForImageClassification, ViTConfig

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score
IMAGE_SIZE = 224

def read_images(paths, dataset, opt):
    fail = 0
    for path in paths:
        #video_path_tmp = os.path.dirname(path[0]).split("release")[1]  
        image_path = os.path.join(opt.data_path, path[1])
        label = path[2]
        if not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        row = (image, label)
        dataset.append(row)
    if fail > 0:
        print(fail)


# Main body
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    seed_everything(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=50, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--data_path', default='../../datasets/deepfakes', type=str,
                        help='Videos directory')
    parser.add_argument('--list_file', default="../../datasets/deepfakes/ForgeryNet/training_video_image_list_2.csv", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--model_path', type=str, default='models',
                        help='Path to save checkpoints.')
    parser.add_argument('--gpu_id', default=1, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--config', type=str, default='',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: ViT, 1: EfficientNet B7, 2: Hybrid)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--forgery_methods', '--list', nargs='+', 
                        help="Which forgery method for training?")
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--show_stats', type=bool, default=True, 
                        help="Show stats")
    parser.add_argument('--logger_name', default='runs/train',
                        help='Path to save the model and Tensorboard log.')
                        
    opt = parser.parse_args()
    print(opt)

    opt.forgery_methods = [int(method) for method in opt.forgery_methods]

    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    if opt.model == 0: 
        if opt.use_pretrained:
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=1)
            for index, (name, param) in enumerate(model.named_parameters()):
                if "layer.11" in name or "layer.10" in name or index > len(list(model.parameters()))-10:
                    param.requires_grad = True
                else:                    
                    param.requires_grad = False
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
    elif opt.model == 1: 
        if opt.use_pretrained:
            model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=1)
            for index, (name, param) in enumerate(model.named_parameters()):
                if "blocks.6" in name or "blocks.5" in name or index > len(list(model.parameters()))-10:
                    param.requires_grad = True
                else:                    
                    param.requires_grad = False
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


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters:", params)
    # Read dataset
    df = pd.read_csv(opt.list_file, sep=' ')
    df = shuffle(df, random_state = 42)
    if opt.show_stats:
        print("**** STATS ****")
        string = ""
        for i in range(0, 9):
            string += "Class " + str(i) + ": " + str(len(df.loc[df["16cls_label"] == i])) + " | "
        print(string)
    
    dataframes = []
    for forgery_method in opt.forgery_methods:
        forgery_method = int(forgery_method)
        df1 = df.loc[(df["16cls_label"] == forgery_method)]
        dataframes.append(df1)

    df2 = df.loc[(df["16cls_label"] == 0)]
    #df = df.drop(df[(df['16cls_label'] == 0) & (df.index % 2 == 0)].index)

    dataframes.append(df2)

    df = pd.concat(dataframes)
    df = df.reset_index()
    # For class 0, sample two frames per video
    already_seen_videos = {}
    indexes_to_drop = []
    for index, row in df.iterrows():
        path = row['image_name']
        label = int(row['16cls_label'])
        if label != 0:
            continue
        video_id = path.split(os.sep)[-1].split("_video")[0]
        if video_id in already_seen_videos:
            if already_seen_videos[video_id] < 3:
                already_seen_videos[video_id] += 1
                continue
            else:
                already_seen_videos[video_id] += 1
        else:
            already_seen_videos[video_id] = 1
        
        indexes_to_drop.append(index)
    df.drop(df.index[indexes_to_drop], inplace=True)
    if opt.show_stats:  
        print("**** REDUCED DATASET STATS ****")
        string = ""
        for i in range(0, 9):
            string += "Class " + str(i) + ": " + str(len(df.loc[df["16cls_label"] == i])) + " | "
        print(string)
    

    df = df.drop(['16cls_label'], axis=1)
    df = df.sort_values(by=['image_name'])
    paths = df.to_numpy()
    
    paths = np.array_split(paths, opt.workers) # Split the paths in chunks for processing
    mgr = Manager()
    dataset = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=dataset, opt=opt), paths):
                pbar.update()
    
    #dataset = sorted(dataset, key=lambda tup: tup[1])
    random.shuffle(dataset)
    print(len(dataset))
    labels = [float(row[1]) for row in dataset]
    dataset = [row[0] for row in dataset]
    
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(dataset, labels, test_size=0.10, random_state=42)

    train_samples = len(train_dataset)
    validation_samples = len(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    
    class_weights = train_counters[1] / train_counters[0]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(validation_labels)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create the data loaders
    if opt.config != '':
        batch_size = config['training']['bs']
    else:
        batch_size = 8
    print(class_weights)
    class_weights = [class_weights, 1-class_weights]
    print(class_weights)
    sample_weights = [0] * len(dataset)
    
    for index, sample in enumerate(train_dataset):
        label = int(train_labels[index])
        sample_weights[index] = class_weights[label]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=train_counters[1]*2, replacement=False)
    train_dataset = DeepFakesDataset(np.asarray(train_dataset), np.asarray(train_labels), IMAGE_SIZE)
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray(validation_dataset), np.asarray(validation_labels), IMAGE_SIZE, mode='validation')
    val_dataset = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    
    
    # TRAINING
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()
    
    model.train()   
    #if opt.model == 0:
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    num_steps = int(opt.num_epochs * len(dl))

    lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-2,
                cycle_limit=1,
                t_in_epochs=False,
    )
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("_")[1]) + 1
    else:
        print("No checkpoint loaded.")
        
    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        save_model = False
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*batch_size)+len(val_dataset))
        train_correct = 0
        positive = 0
        negative = 0
        
        train_batches = len(dl)
        val_batches = len(val_dataset)
        total_batches = train_batches + val_batches

        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            if opt.model == 0:
                y_pred = y_pred.logits

            y_pred = y_pred.cpu()
            
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step_update((t * (train_batches) + index))
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(batch_size):
                bar.next()

            if index%100 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*batch_size), "Train 0s: ", negative, "Train 1s:", positive)  


        
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
       
        train_correct /= counter*batch_size
        total_loss /= counter
        val_preds = []
        for index, (val_images, val_labels) in enumerate(val_dataset):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            
            if opt.model == 0:
                val_pred = val_pred.logits

            val_pred = val_pred.detach().cpu()
            val_preds.extend([custom_round_single_value(torch.sigmoid(pred).numpy()[0]) for pred in val_pred])
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            bar.next()
            
        #scheduler.step()
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            save_model = True
            not_improved_loss = 0
        
        tb_logger.add_scalar("Training/Accuracy", train_correct, t)
        tb_logger.add_scalar("Training/Loss", total_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", total_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_correct, t)

        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_f1_macro:" + str(f1_score(validation_labels, val_preds, average='macro')) + " val_f1_micro:" + str(f1_score(validation_labels, val_preds, average='micro')) + " val_f1_weighted:" + str(f1_score(validation_labels, val_preds, average='weighted')) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))
    
        
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)

        forgery_methods_string = 'm'.join([str(method) for method in opt.forgery_methods])
        
        if save_model and t > 10:
            torch.save(model.state_dict(), os.path.join(opt.model_path, opt.model_name + "_" + str(t) + "_" + forgery_methods_string))

    #training_set = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[0].split(" "))) for row in training_set]))
  
