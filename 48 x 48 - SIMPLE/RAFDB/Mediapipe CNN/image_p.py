import os
import cv2
import numpy as np
import pickle

# Define the path to your data folder
data_folder = "W:/RAFDB/RAFDB names/"
# data_folder = "C:/Users/darks/Desktop/Emotion AI/Emotion Recognition/DATASETS/Batch_Ready 7/"

# Define the list of classes
classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define the size of the images
img_size = 48

# Initialize the list of image data and labels
data = []
labels = []

# Loop over each class folder
for class_idx, class_name in enumerate(classes):
    # Define the path to the class folder
    class_folder = os.path.join(data_folder, class_name)
    # Loop over each image in the class folder
    for img_name in os.listdir(class_folder):
        # Read the image using OpenCV
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)
        # Resize the image to the desired size
        img = cv2.resize(img, (img_size, img_size))
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add the image data and label to the lists
        data.append(img)
        labels.append(class_idx)

# Convert the lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save the data and labels as a .p file
with open("image_data48.p", "wb") as f:
    pickle.dump((data, labels), f)

