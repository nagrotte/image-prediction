##iMAGE REC MODEL USING Transfer Learning Inception V3
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config= ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input,Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

import numpy as np
from glob import glob

IMAGE_SIZE= [224,224]
train_path= 'Datasets/train'
valid_path='Datasets/test'

import tensorflow
resnet152V2= tensorflow.keras.applications.ResNet152V2(input_shape=IMAGE_SIZE + [3],weights='imagenet', include_top=False)

for layer in resnet152V2.layers:
  layer.trainable=False

folders = glob('Datasets/train/*')

print ("folders {}: and Folders {} :", folders, len(folders))

X= Flatten()(resnet152V2.output)

prediction = Dense(len(folders), activation ='softmax')(X)

model = Model(inputs=resnet152V2.input, outputs=prediction)

model.summary()


model.compile (
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics = ['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)