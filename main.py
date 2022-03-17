def plot_loss_curves(histor):
  lis = list(histor.history.keys())
  list_name = lis.copy()
  fig, ax = plt.subplots(nrows=len(lis), figsize=(10, 10))
  for i in range(len(lis)):
    list_name[i] = ax[i].plot(histor.history[lis[i]])
    ax[i].set(title=f'{lis[i]}')
  fig.tight_layout()

  
  def loadprep_image(filepath, shape=(224, 224)):
    "Returns a normalized and reshaped image as size 1 batch"
    image = plt.imread(filepath)
    image = tf.expand_dims(tf.image.resize(image, size=shape)/255, axis=0)
    return image

  
import random, os
def plot_random_image(filepath, label):
  # Takes filepath and label of an image and plots it
  folder_path = filepath + '/' + label
  listdir = os.listdir(folder_path)
  image_path = folder_path + '/' + random.sample(listdir, 1)[0]
  image = plt.imread(image_path)
  plt.imshow(image)
  plt.title(f'Label = {label}, shape = {image.shape}')
  plt.axis('off')

def plot_prediction(filepath, classnames, model):
  # Plots an image with predicted label and prediction probability
  img = loadprep_image(filepath)
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

  from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

def create_tb_callback(name, dir):
  log_dir = dir + '/' + name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%')
  tensorboard_callback = TensorBoard(log_dir=log_dir)
  return tensorboard_callback
