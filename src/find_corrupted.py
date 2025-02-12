from PIL import Image
import os

dataset_path = "path_to_image_directory"
log_file = "log_file_name"
file = open(log_file, "w")

for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image is not corrupted
                    file.write(f"{file_path} is not corrupted.")
            except Exception as e:
                # print information about corrupted files to the log file
                file.write(f"⚠️Corrupted image found: {file_path} - {e}")
