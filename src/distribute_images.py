import os
import random
import shutil

image_dir = "path_to_the_root"
train_dir = "path_to_the_train_dir"
val_dir = "path_to_the_validation_dir"
test_dir = "path_to_the_test_dir"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
num_of_images = len(all_images)

# choose test images
num_of_test_images = int(len(all_images) * 0.1)                        # 10% of all the images will be used as test set
test_images = set(random.sample(all_images, num_of_test_images))

# move test images to it's directory
for image in test_images:
    old_path = os.path.join(image_dir, image)
    new_path = os.path.join(test_dir, image)
    shutil.move(old_path, new_path)

all_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
num_of_train_images = int(num_of_images * 0.8)                         # 10% of all the images will be used as train set
train_images = set(random.sample(all_images, num_of_train_images))

# move train images to the new directory
for image in train_images:
    old_path = os.path.join(image_dir, image)
    new_path = os.path.join(train_dir, image)
    shutil.move(old_path, new_path)

# move the rest to the validation directory
for image in os.listdir(image_dir):                                     # Ramaining images (10% of all the images) will be used as validation set
    old_path = os.path.join(image_dir, image)
    new_path = os.path.join(val_dir, image)
    shutil.move(old_path, new_path)
