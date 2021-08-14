# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 12:12:11 2021

@author: Chandramouli
"""

import tensorflow as tf
from IPython.display import Image, display

import matplotlib.pyplot as plt
import tensorflow_hub as hub

import numpy as np

import warnings
warnings.filterwarnings('ignore')
import wget
print(tf.__version__)
#!nvidia-smi for colab
url="https://storage.googleapis.com/ibeans/train.zip"
filename = wget.download(url)
url="https://storage.googleapis.com/ibeans/validation.zip"
filename = wget.download(url)
url="https://storage.googleapis.com/ibeans/test.zip"
filename = wget.download(url)


"""#!mkdir beans

!unzip train.zip -d beans/
!unzip test.zip -d beans/
!unzip validation.zip -d beans/

!ls beans

!find beans -type f | wc -l

!find beans/test -type f | wc -l

!find beans/validation -type f | wc -l"""


display(Image('beans/train/healthy/healthy_train.0.jpg'))

display(Image('beans/train/angular_leaf_spot/angular_leaf_spot_train.124.jpg'))

display(Image('beans/train/bean_rust/bean_rust_train.162.jpg'))

batch_size = 128
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory('beans/train',
  seed=111,
  image_size=(img_height, img_width),
  batch_size=batch_size)


test_ds = tf.keras.preprocessing.image_dataset_from_directory('beans/test',
  seed=111,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory('beans/validation',
  seed=111,
  image_size=(img_height, img_width),
  batch_size=batch_size)


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


classes=train_ds.class_names
print(classes)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):#taking first batch of images
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")
    
##for gpu to keep data in cache   
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"


feature_extractor_layer = hub.KerasLayer(feature_extractor, input_shape=(img_height,img_width,3))

feature_extractor_layer.trainable = False

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)#from tensorflow 2.4 version on if lower use seperate fn

tf.random.set_seed(111)

model = tf.keras.Sequential([
  normalization_layer,
  feature_extractor_layer,
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(3,activation='softmax')
])
    
    
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


history = model.fit(train_ds, epochs=20, validation_data=val_ds)

model.summary()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='best')
plt.show()

result=model.evaluate(test_ds)


plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)

    plt.tight_layout()
    
    img = tf.keras.preprocessing.image.img_to_array(images[i])                    
    img = np.expand_dims(img, axis=0)  

    pred=model.predict(img)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title("Actual Label: %s" % classes[labels[i]])
    plt.text(1, 240, "Predicted Label: %s" % classes[np.argmax(pred)], fontsize=12)

    plt.axis("off")
    
    
model.save('./models', save_format='tf')
#tf latest format older format h5

#!ls -alrt models

##checking the loaded model
model_loaded = tf.keras.models.load_model('./models/')

model_loaded.summary()


from PIL import Image
import numpy as np
from skimage import transform
def process(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


pred_label=model_loaded.predict(process('beans/train/healthy/healthy_train.0.jpg'))
print(classes[np.argmax(pred_label)])

#!zip -r models.zip models/
pred_label

print(tf.__version__)
