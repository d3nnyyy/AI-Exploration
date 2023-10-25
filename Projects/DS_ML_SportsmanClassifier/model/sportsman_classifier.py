import os
import shutil
import cv2 as cv
import numpy as np
import pywt
from matplotlib import pyplot as plt

# Load the image
img = cv.imread("test_images/MESSI.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Load Haar cascades for face and eyes detection
face_cascade = cv.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

# Print the detected faces
print("Detected faces:", faces)

# Draw rectangles around the detected faces and eyes
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Region of Interest (ROI) for the detected face
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect eyes within the ROI
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # Draw rectangles around the eyes
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Display the image with rectangles drawn
plt.imshow(img, cmap='gray')

# Define a function for wavelet transformation
def w2d(img, mode='haar', level=1):
    """
    Perform wavelet transformation on an image.

    Args:
        img (numpy.ndarray): Input image.
        mode (str): Wavelet transformation mode.
        level (int): Transformation level.

    Returns:
        numpy.ndarray: Transformed image.
    """
    img_array = img
    # Datatype conversions
    # convert to grayscale
    img_array = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
    # convert to float
    img_array = np.float32(img_array)
    img_array /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(img_array, mode, level=level)

    # Process Coefficients
    coeffs_h = list(coeffs)
    coeffs_h[0] *= 0

    # Reconstruction
    img_array_h = pywt.waverec2(coeffs_h, mode)
    img_array_h *= 255
    img_array_h = np.uint8(img_array_h)

    return img_array_h

# Define a function to get a cropped image if it contains at least 2 eyes
def get_cropped_image_if_2_eyes(image_path):
    """
    Load an image, detect and crop the face if it contains at least 2 eyes.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Cropped face image if conditions are met, else None.
    """
    img = cv.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

# Get and display the cropped image
cropped_image = get_cropped_image_if_2_eyes("test_images/MESSI.jpg")
plt.imshow(cropped_image)

# Directory paths
path_to_imgs = 'dataset/'
path_to_cropped_imgs = 'dataset/cropped/'

# List image directories
img_dirs = [entry.path for entry in os.scandir(path_to_imgs) if entry.is_dir()]

# Create the cropped image directory if it doesn't exist
if os.path.exists(path_to_cropped_imgs):
    shutil.rmtree(path_to_cropped_imgs)
os.mkdir(path_to_cropped_imgs)

# Store cropped image directories and file names
cropped_image_dirs = []
celebrity_file_names_dict = {}

# Iterate through image directories
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            # Create a directory for each sportsman
            cropped_folder = path_to_cropped_imgs + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)

            # Generate a unique filename for each cropped image
            cropped_file_name = f"{celebrity_name}{count}.png"
            cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
            cv.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1

# Create a class dictionary
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count += 1

print(class_dict)

# Prepare data for machine learning
x = []
y = []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv.imread(training_image)
        scaled_raw_img = cv.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_har_img = cv.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_har_img.reshape(32 * 32, 1)))
        x.append(combined_img)
        y.append(class_dict[celebrity_name])

print(len(x))
print(len(y))

# Wait for a key press to close the OpenCV window
cv.waitKey(0)
