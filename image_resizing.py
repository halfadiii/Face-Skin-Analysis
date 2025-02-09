import os
from PIL import Image

def get_image_dimensions(directory):
    # Dictionary to store image filenames and their dimensions
    image_dimensions = {}
    
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Construct the full path to the file
            file_path = os.path.join(directory, filename)
            # Open the image file
            with Image.open(file_path) as img:
                # Get dimensions
                width, height = img.size
                # Store dimensions in dictionary
                image_dimensions[filename] = (width, height)
    
    return image_dimensions

# Specify the directory containing the images
directory_path = r'C:\Adi\GitHub\LOREAL\images\archive\DATA\testing\Acne'

# Get the dimensions of all images in the directory
dimensions = get_image_dimensions(directory_path)

# Print the dimensions
for filename, size in dimensions.items():
    print(f'{filename}: {size[0]} x {size[1]} pixels')
