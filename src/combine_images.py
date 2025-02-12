import os
image_dir = "path_to_the_downloaded_dataset"
new_dir = "path_to_the_new_directory"

os.makedirs(new_dir, exist_ok=True)

for image in os.listdir(image_dir + "Cat"):
    old_path = image_dir + "Cat/" + image
    new_path = new_dir + "CAT_" + image       
    os.rename(old_path, new_path)
        
for image in os.listdir(image_dir + "Dog"):
    old_path = image_dir + "Dog/" + image
    new_name =new_dir + "DOG_" + image  
    os.rename(old_path, new_name)
