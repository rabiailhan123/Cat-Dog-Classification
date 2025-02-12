
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
addr = sys.argv[1]
model = load_model(addr)
batch_size = 32
image_size = 128 

test_datagen = ImageDataGenerator(rescale = 1./255)

# creating data frame objects for test set
test_image_dir = "/home/rabia/Documents/catndog_images/test"
all_test_images = [image for image in os.listdir(test_image_dir) if os.path.isfile(os.path.join(test_image_dir, image))]

test_image_paths = [os.path.join(test_image_dir, image) for image in os.listdir(test_image_dir)]
test_labels = []

for image in all_test_images:
    if "dog" in image.lower():
        test_labels.append("DOG")
    elif "cat" in image.lower():
        test_labels.append("CAT")
    else:
        test_labels.append("UNKNOWN")
        print(f"Unknown image is detected: {image}")

test_df = pd.DataFrame({
    "image_path" : test_image_paths,
    "label": test_labels
    })

test_generator = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    x_col = 'image_path',
    y_col = 'label',
    batch_size = batch_size,
    target_size = (image_size, image_size),
    shuffle = False
)

print("indices are ", test_generator.class_indices)
test_loss, test_acc = model.evaluate(test_generator)

# Predict on test images
predictions = model.predict(test_generator)
