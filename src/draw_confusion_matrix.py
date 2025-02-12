from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

test_image_dir = "/home/rabia/Documents/catndog_images/test"

# Load the trained model
model = load_model("models/cat_dog_prediction_with_basic_cnn_epoch_11.h5")
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

test_generator = test_datagen.flow_from_dataframe(dataframe = test_df,
                                                x_col = 'image_path',
                                                y_col = 'label',
                                                batch_size = batch_size,
                                                target_size = (image_size, image_size),
                                                shuffle = False
                                                )

# Get true labels and predictions from the train dataset
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
true_classes = test_generator.classes  # True class labels
class_labels = list(test_generator.class_indices.keys())  # ['CAT', 'DOG']

# Create the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Display the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print a classification report for precision, recall, f1-score
print("Classification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))
