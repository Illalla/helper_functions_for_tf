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
