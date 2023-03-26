from tqdm import tqdm
import os
import time
from datetime import datetime
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import re
import warnings


warnings.simplefilter("ignore")


from utils.Meter import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes


def get_losses(filename):

  # open the text file
  with open(filename, 'r') as f:
      contents = f.read()

  # extract the train losses and val losses using regular expressions
  train_losses_str = re.search(
      r"losses:{'train': (.+?), 'val'", contents).group(1)
  val_losses_str = re.search(r"'val': \[(.+?)\}", contents).group(1)

  train_losses_str = train_losses_str.replace('[', '').replace(']', '')
  val_losses_str = val_losses_str.replace('[', '').replace(']', '')

  # convert the strings to lists of floating-point numbers
  train_losses = [float(x.strip()) for x in train_losses_str.split(', ')]
  val_losses = [float(x.strip()) for x in val_losses_str.split(', ')]

  print(f"{len(train_losses)} train and {len(val_losses)} val losses found in {filename}")

  # print the lists
  return train_losses, val_losses


def get_param_count(filename):

  # open the text file
  with open(filename, 'r') as f:
      contents = f.read()

  # extract the param count using regular expressions
  param_str = re.search(
      r"(?<=parameter_count:).*", contents).group(0)
  param_count = int(param_str)
  print(f"Parameter count = {param_count} in {filename}")

  return param_count


def get_dice_scores(filename):

  # open the text file
  with open(filename, 'r') as f:
      lines = f.readlines()

  for line in lines:
    if "dice_scores" in line:

      # extract the dice scores and iou scores using regular expressions
      train_dice_str = re.search(
          r"'train': (.+?), 'val'", line).group(1)
      val_dice_str = re.search(r"'val': \[(.+?)\}", line).group(1)

      train_dice_str = train_dice_str.replace('[', '').replace(']', '')
      val_dice_str = val_dice_str.replace('[', '').replace(']', '')

      # convert the strings to lists of floating-point numbers
      train_dice = [float(x.strip()) for x in train_dice_str.split(', ')]
      val_dice = [float(x.strip()) for x in val_dice_str.split(', ')]
      break
  print(f"{len(train_dice)} train and {len(val_dice)} val dice score found in {filename}")
  return train_dice, val_dice


def get_jaccard_scores(filename):

  # open the text file
  with open(filename, 'r') as f:
      lines = f.readlines()

  for line in lines:
    if "jaccard_scores" in line:

      # extract the dice scores and iou scores using regular expressions
      train_jaccard_str = re.search(
          r"'train': (.+?), 'val'", line).group(1)
      val_jaccard_str = re.search(r"'val': \[(.+?)\}", line).group(1)

      train_jaccard_str = train_jaccard_str.replace('[', '').replace(']', '')
      val_jaccard_str = val_jaccard_str.replace('[', '').replace(']', '')

      # convert the strings to lists of floating-point numbers
      train_jaccard = [float(x.strip()) for x in train_jaccard_str.split(', ')]
      val_jaccard = [float(x.strip()) for x in val_jaccard_str.split(', ')]
      break
  print(f"{len(train_jaccard)} train and {len(val_jaccard)} val jaccard score found in {filename}")
  return train_jaccard, val_jaccard


def get_train_run_time(filename):

  # open the text file
  with open(filename, 'r') as f:
      contents = f.read()

  # extract the param count using regular expressions
  time_str = re.search(
      r"(?<=last_completed_run_time:).*(?=\n)", contents).group(0)
  time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
  print(f"Trainer runtime = {time_obj} in {filename}")
  return time_obj


def plot_param_count(results_dict, paletteCols):
  x = []
  y = []
  for model_name in results_dict:
    x.append(model_name)
    y.append(results_dict[model_name]["parameter_count"])

  # sort based on ascending order of parameter count
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Parameter Count")

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.set_xlabel("Models")
  ax.set_ylabel("Parameter Count")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_train_losses(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["train_losses"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  train_loss_data = {'Epochs': epoch_num_xlist}
  train_loss_df = pd.DataFrame(train_loss_data)

  for model_name in results_dict:
    train_loss_df[model_name] = results_dict[model_name]["train_losses"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Train Losses")
  sns.lineplot(data=train_loss_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Train Losses")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_val_losses(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["val_losses"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  val_loss_data = {'Epochs': epoch_num_xlist}
  val_loss_df = pd.DataFrame(val_loss_data)

  for model_name in results_dict:
    val_loss_df[model_name] = results_dict[model_name]["val_losses"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Validation Losses")
  sns.lineplot(data=val_loss_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Validation Losses")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_train_dices(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["train_dices"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  train_dices_data = {'Epochs': epoch_num_xlist}
  train_dices_df = pd.DataFrame(train_dices_data)

  for model_name in results_dict:
    train_dices_df[model_name] = results_dict[model_name]["train_dices"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Train Dice Scores")
  sns.lineplot(data=train_dices_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Train Dice Scores")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_val_dices(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["val_dices"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  val_dices_data = {'Epochs': epoch_num_xlist}
  val_dices_df = pd.DataFrame(val_dices_data)

  for model_name in results_dict:
    val_dices_df[model_name] = results_dict[model_name]["val_dices"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Validation Dice Scores")
  sns.lineplot(data=val_dices_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Validation Dice Scores")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_train_jaccards(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["train_jaccards"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  train_jaccards_data = {'Epochs': epoch_num_xlist}
  train_jaccards_df = pd.DataFrame(train_jaccards_data)

  for model_name in results_dict:
    train_jaccards_df[model_name] = results_dict[model_name]["train_jaccards"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Train Jaccard Scores")
  sns.lineplot(data=train_jaccards_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Train Jaccard Scores")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_val_jacards(results_dict, paletteCols):

  epoch_num = len(results_dict["3DOnet_DoubleConv_Kernel1"]["val_jacards"])
  epoch_num_xlist = list(range(1, epoch_num + 1))
  val_jacards_data = {'Epochs': epoch_num_xlist}
  val_jacards_df = pd.DataFrame(val_jacards_data)

  for model_name in results_dict:
    val_jacards_df[model_name] = results_dict[model_name]["val_jacards"]

  models_ls = []
  for model_name in results_dict:
    models_ls.append(model_name)

  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Validation Jaccard Scores")
  sns.lineplot(data=val_jacards_df[models_ls], linewidth=3, palette=paletteCols)
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Validation Jaccard Scores")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  plt.show()


def plot_trainer_runtime(results_dict, paletteCols):
  x = []
  y = []
  for model_name in results_dict:
    x.append(model_name)
    y.append(results_dict[model_name]["trainer_runtime"])

  # sort based on ascending order of parameter count
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  y = [round(i.hour+(i.minute/60), 2) for i in y]
  # today = datetime.datetime.now()
  # y = [datetime.datetime.combine(today, t) for t in y]
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("Model Trainer Runtime (hours)")

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])
  # plt.bar(x, y)
  ax.set_xlabel("Models")
  ax.set_ylabel("Trainer runtime in hours")
  ax.legend()
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/trainer_runtime_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()
  


def plot_WT_dice(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["WT dice"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("WT Dice Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("WT Dice")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/WT_dice_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_WT_jaccard(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["WT jaccard"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("WT Jaccard Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("WT Jaccard")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/WT_jaccard_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_TC_dice(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["TC dice"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("TC Dice Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("TC Dice")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/TC_dice_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_TC_jaccard(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["TC jaccard"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("TC Jaccard Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("TC Jaccard")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/TC_jaccard_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_ET_dice(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["ET dice"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("ET Dice Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("ET Dice")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/ET_dice_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_ET_jaccard(results_dict, paletteCols):
  mean_scores_dict = {
      k: results_dict[k]["ET jaccard"] for k in results_dict}
  x = []
  y = []
  for model_name in mean_scores_dict:
    x.append(model_name)
    y.append(round(mean_scores_dict[model_name], 3))
  x = [val for _, val in sorted(zip(y, x))]
  y = sorted(y)
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_title("ET Jaccard Scores")
  # plt.bar(x, y)

  sns.barplot(x=x, y=y, palette=paletteCols)
  ax.bar_label(ax.containers[0])

  ax.set_xlabel("Models")
  ax.set_ylabel("ET Jaccard")
  ax.set_ylim(0.5, 0.9)
  plt.xticks(rotation=45, ha='right')
  fig.savefig("results/ET_jaccard_all.png", format="png",
              pad_inches=0.2, transparent=False, bbox_inches='tight')
  plt.show()


def plot_inference_time(results_dict, paletteCols):
    inferenceDict = {
        k: results_dict[k]["Inference time"] for k in results_dict}
    # print(inferenceDict)
    x = []
    y = []
    for model_name in inferenceDict:
        x.append(model_name)
        y.append(inferenceDict[model_name])

    # sort based on ascending order of parameter count
    x = [val for _, val in sorted(zip(y, x))]
    y = sorted(y)
    y = [round(i.seconds + i.microseconds*1e-6, 3) for i in y]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Inference time for 53 samples (seconds)")

    sns.barplot(x=x, y=y, palette=paletteCols)
    ax.bar_label(ax.containers[0])
    ax.set_xlabel("Models")
    ax.set_ylabel("Inference time in seconds")
    ax.legend()
    # ax.set_ylim(15, 40)
    plt.xticks(rotation=45, ha='right')
    fig.savefig("results/inference_time_all.png", format="png",
                pad_inches=0.2, transparent=False, bbox_inches='tight')
    plt.show()



