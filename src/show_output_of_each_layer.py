import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

model = load_model("path_to_trained_model")

img_path = "path_to_the_image"
img = Image.open(img_path).resize(size = (128, 128))                           # Adjust size to match the model input
img = img.convert('RGB')                                                       # convert the channel format to RGB
img_array = image.img_to_array(img)                                            # convert the image into an array
img_array = np.expand_dims(img_array, axis=0)   
img_array /= 255.0                              
        
layer_outputs = [layer.output for layer in model.layers]                       # output of each layer
activation_model = Model(inputs = model.inputs, outputs = layer_outputs)       # another model to keep the input and outputs of the original model
activations = activation_model(img_array)

for layer, activation in zip(model.layers, activations):
    print(f"Activations of layer: {layer.name}")
    print(activation.shape)                                                    # Print the shape of the activation
    if len(activation.shape) == 4:                                             # If the output is 4D (batch_size, height, width, channels)
        # Visualization of the feature maps
        num_filters = activation.shape[-1]
        size = activation.shape[1]
        plt.title(layer.name)
        # arranging number of grids based on num_filters
        x = 8
        y = int(np.ceil(num_filters / x))
        for i in range(num_filters):                                            # print all the filters
            plt.subplot(x, y, i + 1)
            plt.imshow(activation[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.show()
    else:
        # for fully connected layers or other 2D outputs, print the shape
        print(activation)
