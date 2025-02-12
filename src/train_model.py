
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization, Conv2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import seaborn as sns
import pandas as pd
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

# image parameters
image_size = 128
channel = 3         #RGB
batch_size = 32

# path to datasets
train_dir = "path_to_train_dataset"
val_dir = "path_to_validation_dataset"

# preprocessing, convert all images (128x128), normalize them between [0, 1]
# data augmentation 
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    horizontal_flip = True,
    zoom_range = 0.2,
    fill_mode = 'reflect',
    brightness_range = [0.8, 1.2],
    channel_shift_range = 50,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

val_datagen = ImageDataGenerator(rescale = 1./255)

# load datasets from the directory
train_dataset = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (128, 128),
    batch_size = 32,
    class_mode = "categorical"
)

val_dataset = val_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (128, 128),
    batch_size = 32,
    class_mode = "categorical"
)

# CNN(Convolutional Neural Network)
# layers
model = Sequential([
    Input(shape = (image_size, image_size, channel)),
    
    Conv2D(filters = 32, kernel_size = (3, 3), kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "valid"),
    Dropout(0.1),
    
    Conv2D(filters = 64, kernel_size = (3, 3), kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "valid"),
    Dropout(0.2),
    
    Conv2D(filters = 128, kernel_size = (3, 3), kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "valid"),
    Dropout(0.3),
    
    Conv2D(filters = 256, kernel_size = (3, 3), kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "valid"),
    Dropout(0.4),
    
    Flatten(),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),
    # last and the output layer
    Dense(2, activation = "softmax")
    ])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(" Model Summary\n", model.summary())

os.makedirs('models', exist_ok=True)

# save the model in every 2 epochs
checkpoint = ModelCheckpoint(
    'models/{epoch:02d}.h5',
    save_freq = 1250,
    save_best_only = False,
    save_weights_only = False,
)

# callbacks
early_stopping = EarlyStopping(
    monitor = "val_loss",           # value to be monitored
    min_delta = 0.00001,            # when the change in the monitored value is less than min_delta value stop training
    restore_best_weights = True,    # when this callback took action restore the model with the best value of the monitored value
    start_from_epoch = 8,
    verbose = 0
)

lr_scheduler = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,    # when active reduce learning rate by half
    patience = 2,   # if val_loss doesn't improve for 3 epochs, reduce lr
    min_lr = 1e-6,      # minimum learning rate
    verbose = 1      # print message to the log screen whenever lr is reduced 
)

# model training part
cat_and_dog_result = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 20,
    callbacks = [checkpoint, early_stopping, lr_scheduler]
)

# write the model history to a file
history_df = pd.DataFrame(cat_and_dog_result.history)
history_df.to_csv("history.txt", index = False)

plt.figure(figsize=(18, 5), dpi = 200)
sns.set_style('darkgrid')

plt.subplot(1, 2, 1)
plt.title('Cross Entropy Loss', fontsize = 15)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.plot(history_df['loss'])
plt.plot(history_df['val_loss'])

plt.subplot(1, 2, 2)
plt.title('Classification Accuracy', fontsize = 15)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('Accuracy', fontsize = 12)
plt.plot(history_df['accuracy'])
plt.plot(history_df['val_accuracy'])

plt.show()

# Evaluate for train generator
train_loss, train_acc = model.evaluate(train_dataset, batch_size = batch_size, verbose = 0)

print('The accuracy of the model for training data is:', train_acc * 100)
print('The Loss of the model for training data is:', train_loss)

# Evaluate for validation generator
val_loss, val_acc = model.evaluate(val_dataset, batch_size = batch_size, verbose = 0)

print('The accuracy of the model for validation data is:', val_acc * 100)
print('The Loss of the model for validation data is:', val_loss)

# save the model
os.makedirs('models', exist_ok=True)
model.save("models/model_name.h5")

# Load and display sample images for verification
sample_dir = os.path.join(train_dir, os.listdir(train_dir)[0])
sample_images = os.listdir(sample_dir)[:10]

fig, axes = plt.subplots(1, 10, figsize=(20, 4))
for idx, img_name in enumerate(sample_images):
    img_path = os.path.join(sample_dir, img_name)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(os.path.basename(sample_dir))
    axes[idx].axis('off')
plt.show()
