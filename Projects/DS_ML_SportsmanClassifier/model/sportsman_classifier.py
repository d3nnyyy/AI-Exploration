import os
import shutil
import cv2 as cv
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
plt.show()


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
plt.show()

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

# Wait for a key press to close the OpenCV window
cv.waitKey(0)
