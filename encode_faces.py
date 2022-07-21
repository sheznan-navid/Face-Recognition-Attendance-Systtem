# USAGE
# When encoding on laptop, desktop, or GPU (slower, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# When encoding on Raspberry Pi (faster, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

# import the necessary packages
import imutils
from imutils import paths
from PIL import Image
import face_recognition
import numpy as np
# import argparse
import pickle
import cv2
import os
import sys

path = os.path.abspath(os.path.split(sys.argv[0])[0])

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(path + "\\dataset\\"))
print("[INFO] loading face detector...")
protoPath = path + "\\face_detection_model\\" + "deploy.prototxt"
modelPath = path + "\\face_detection_model\\" + "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    if not os.path.isdir(path + "\\trained_dataset\\" + name):
        os.mkdir(path + "\\trained_dataset\\" + name)
    im = Image.open(imagePath)
    im.save(imagePath, optimize=True, quality=50)
    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    print(imagePath)
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    rgb = image[:, :, ::-1]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # boxes = face_recognition.face_locations(rgb, model="hog")
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) == 1:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        im = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, im, 2]
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, im, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            intbox = [(startY, endX, endY, startX)]
            print(intbox)
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, intbox)
            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            cv2.imwrite(path + "\\trained_dataset\\" + name + "\\" + name + " " + str(i + 1) + ".jpg", face)
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            knownEncodings.append(encodings[0])
            knownNames.append(name)
        else:
            os.remove(imagePath)
            print("No face found, Removed.")
    else:
        os.remove(imagePath)
        print("No face or multiple found, Removed.")


# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(path + "/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
