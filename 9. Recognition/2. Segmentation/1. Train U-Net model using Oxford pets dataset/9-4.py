from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import os
import random
import cv2 as cv

input_dir = './datasets/oxford_pets/images/images/'
target_dir = './datasets/oxford_pets/annotations/annotations/trimaps/'
img_siz = (160, 160)    # Size of input image that enters the model
n_class = 3             # Three segmentation labels (1:object, 2:background, 3:edge)
batch_siz = 32          # Size of mini-batch

img_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])
label_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(target_dir) if f.endswith('.png') and not f.startswith('.')])

# Prepare data for learning and predicting
class OxfordPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, img_paths, label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.label_paths = label_paths
        
    def __len__(self):
        return len(self.label_paths) // self.batch_size
    
    def __getitem__(self, idx): # Provide mini-batch when fit function needs it
        i = idx*self.batch_size
        batch_img_paths = self.img_paths[i:i+self.batch_size]
        batch_label_paths = self.label_paths[i:i+self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_img_paths):      # Get num(mini-batch) input images
            img = load_img(path, target_size = self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_label_paths):    # Get num(mini-batch) label images
            img = load_img(path, target_size = self.img_size, color_mode = 'grayscale')
            y[j] = np.expand_dims(img, 2)
            y[j] -= 1
        return x, y
    
def make_model(img_size, num_classes):
    inputs = keras.Input(shape = img_size + (3,))   # Input Layer
    
    # First convolution layer
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relue')(x)

    previous_block_activation = x # for shortcut connection
    
    # Down Sampling of U-Net
    for filters in [64, 128, 256]:  # Increase the depth
        # Two convolutions and one max pooling
        x = layers.Activation('relue')(x)
        x = layers.SeperableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relue')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)    
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])   # Connect shortcut
        previous_block_activation = x   # for shortcut connection
    
    # Up Sampling of U-Net
    for filters in [256, 128, 64, 32]:  # Decrease the depth
        # Two convolutions and one up sampling
        x = layers.Activation('relue')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relue')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])   # Connect shortcut
        previous_block_activation = x   # for shortcut connection
        
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    model = keras.Model(inputs, outputs)    # Create model
    return model

model = make_model(img_siz, n_class)

random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)
test_samples = int(len(img_paths)*0.1)  # Use 10% as test samples
train_img_paths = img_paths[:-test_samples] # exclude test samples
train_label_paths = label_paths[:-test_samples]
test_img_paths = img_paths[-test_samples:]
test_label_paths = label_paths[-test_samples:]

# Training set
train_gen = OxfordPets(batch_siz, img_siz, train_img_paths, train_label_paths)
# Testing set
test_gen = OxfordPets(batch_siz, img_siz, test_img_paths, test_label_paths)

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
cb = [keras.callbacks.ModelCheckpoint('oxford_seg.h5', save_best_only = True)]  # Automatically save best weights

model.fit(train_gen, epochs = 30, validation_data = test_gen, callbacks = cb)

preds = model.predict(test_gen) # Save all the predictions

cv.imshow('Sample image', cv.imread(test_img_paths[0])) # Input image
cv.imshow('Segmentation label', cv.imread(test_label_paths[0])*64)  # True result
cv.imshow('Segmentation prediction', preds[0])  # Prediction made by model

cv.waitKey()
cv.destroyAllWindows()
