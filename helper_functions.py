import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import datetime
import os
import random
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score
import time

def plot_loss_curves(history, fine_tune_history=None, fine_initial_epoch=None, figsize=(10, 6)):
  # Plots training and validation curves
  plot_number = int(len(history.history.keys()) / 2)
  fig, ax = plt.subplots(nrows=plot_number, figsize=figsize)
  i = 0
  for x in history.history.keys():
    if fine_tune_history is not None:
      if 'val' not in x:
        ax[i].plot(history.history[str(x)]+fine_tune_history.history[str(x)], label= str(x))
        ax[i].plot(history.history['val'+'_'+str(x)]+fine_tune_history.history['val'+'_'+ str(x)], label= 'val' + '_' + str(x))
        if fine_tune_history is not None:
          ax[i].plot([fine_initial_epoch-1, fine_initial_epoch-1], plt.ylim(), label='Fine Tuning')
        ax[i].set(title=x)
        ax[i].legend()
        i += 1
    else:
      if 'val' not in x:
        ax[i].plot(history.history[str(x)], label= str(x))
        ax[i].plot(history.history['val'+'_'+str(x)], label= 'val' + '_' + str(x))
        ax[i].set(title=x)
        ax[i].legend()
        i += 1
  fig.tight_layout()



  
def loadprep_image(filepath, shape=(224, 224), scaling=True):
  image = plt.imread(filepath)
  if scaling:
    image = tf.expand_dims(tf.image.resize(image, size=shape)/255, axis=0)
  else:
    image = tf.expand_dims(tf.image.resize(image, size=shape), axis=0)
  return image


def loadprep_img(filepath, shape=(224, 224), scaling=True):
  img = tf.io.decode_image(tf.io.read_file(filepath), channels=3)
  reshaped_img = tf.image.resize(img, size=shape)
  if scaling:
    return reshaped_img/255
  else:
    return reshaped_img

  
def plot_random_image(filepath, label):
  # Takes filepath and label of an image and plots it
  folder_path = filepath + '/' + label
  listdir = os.listdir(folder_path)
  image_path = folder_path + '/' + random.sample(listdir, 1)[0]
  image = plt.imread(image_path)
  plt.imshow(image)
  plt.title(f'Label = {label}, shape = {image.shape}')
  plt.axis('off')

def plot_prediction(filepath, classnames, model, shape=(224, 224), scaling=True):
  # Plots an image with predicted label and prediction probability
  img = loadprep_image(filepath, shape=shape, scaling=scaling)
  prediction = model.predict(img)

  if len(prediction[0]) > 1:
    pred_class = classnames[tf.argmax(prediction).numpy()[0]]
    prob = tf.math.reduce_max(prediction).numpy()
  else:
    pred_class = classnames[int(np.round(prediction)[0][0])]
    if pred_class == classnames[0]:
      prob = 1-prediction[0][0]
    else:
      prob = prediction[0][0]

  plt.imshow(plt.imread(filepath))
  plt.title(f'{pred_class}, {100 * prob:.2f}%')



def create_tb_callback(name, dir):
  log_dir = dir + '/' + name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%')
  tensorboard_callback = TensorBoard(log_dir=log_dir)
  return tensorboard_callback


def walk_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}.')

def plot_confusion_matrix(y_pred, y_true, class_names=None, figsize=(10, 10), text_size=10):
  conf_mat = confusion_matrix(y_true, y_pred)
  num_classes=conf_mat.shape[0]
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
  fig.colorbar(cax)
  if class_names:
    labels = class_names
  else:
    labels = np.arange(conf_mat.shape[0])
  
  ax.set(title='Confusion Matrix',
         xlabel='Predicted Label',
         ylabel='True Label',
         xticks=np.arange(num_classes),
         yticks=np.arange(num_classes),
         xticklabels=labels,
         yticklabels=labels)
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)
  for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
    plt.text(j, i, f"{conf_mat[i, j]}",
      horizontalalignment="center",
      size=text_size)

  
def plot_f1_scores(y_true, y_preds, class_names, figsize=(10, 10), text_size=10):  
  f1_scores = {}
  f1 = f1_score(y_true, y_preds, average=None)

  for i, class_name in enumerate(class_names):
    f1_scores[class_name] = f1[i]
  f1_scores_df = pd.DataFrame({'class_names':f1_scores.keys(), 'f1_score':f1_scores.values()}).sort_values('f1_score')

  fig, ax = plt.subplots(figsize=figsize)
  bar = ax.bar(f1_scores_df['class_names'], f1_scores_df['f1_score'])

  mean = [f1_scores_df['f1_score'].mean()] * len(class_names)
  plt.plot(f1_scores_df['class_names'], mean, linewidth=figsize[0]/5, color='red', linestyle='--', label='mean')

  fig.suptitle('F1 Scores', fontsize = figsize[0])

  ax.set(yticks=np.arange(0, 1, 0.1))

  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  legend = ax.legend(loc='upper left', prop={'size': figsize[0]})

  plt.show();

def time_pred(model, data):
  start = time.perf_counter()
  model.predict(data)
  end = time.perf_counter()
  return end-start
