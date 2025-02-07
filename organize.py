import os
import pandas as pd
import shutil

def organize_images(base_path, csv_filename, output_base):
    # Construct the full path to the CSV file
    csv_path = os.path.join(base_path, csv_filename)
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the filename of the image
        filename = row['filename']
        # Loop through each condition column
        for condition in ['acne', 'eksim', 'herpes', 'panu', 'rosacea']:
            if row[condition] == 1:
                # Construct the folder path for this condition in the output directory
                folder_path = os.path.join(output_base, condition)
                # Create the folder if it does not exist
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # Construct the full path to the source image
                src_image_path = os.path.join(base_path, filename)
                # Construct the full path to the destination
                dest_image_path = os.path.join(folder_path, filename)
                # Copy the image to the corresponding folder
                shutil.copy(src_image_path, dest_image_path)
                break  # Since each image belongs to one folder based on the dataset description

# Set the paths for the test and train datasets
test_path = r'C:\Adi\GitHub\LOREAL\images\Face Skin Diseases\test'
train_path = r'C:\Adi\GitHub\LOREAL\images\Face Skin Diseases\train'
output_base = r'C:\Adi\GitHub\LOREAL'

organize_images(test_path, r'C:\Adi\GitHub\LOREAL\images\Face Skin Diseases\test\_classes.csv', output_base)  # Assuming the name of the CSV file
organize_images(train_path, r'C:\Adi\GitHub\LOREAL\images\Face Skin Diseases\train\_classes.csv', output_base)  # Assuming the name of the CSV file
