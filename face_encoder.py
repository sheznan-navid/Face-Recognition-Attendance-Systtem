import logging
import os, sys
import pickle
import imutils
from imutils import paths
import cv2
import numpy as np
import face_recognition
from PIL import Image

path = os.path.abspath(os.path.split(sys.argv[0])[0])
storage_count = 60
prototxt_path = path + 'model_data/deploy.prototxt'
caffemodel_path = path + 'model_data/weights.caffemodel'


def encode_faces():
    # print("[INFO] started encoding faces...")
    logging.info('started encoding faces...')
    # print("[INFO] quantifying faces...")
    logging.debug('quantifying faces...')
    _, dirs, __ = os.walk(path + "\\dataset\\").__next__()
    image_paths = list(paths.list_images(path + "\\dataset\\"))
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        logging.debug("processing image {}/{}".format(i + 1, len(image_paths)))
        name = imagePath.split(os.path.sep)[-2]
        if not os.path.isdir(path + "\\trained_dataset\\" + name):
            os.mkdir(path + "\\trained_dataset\\" + name)

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=800)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")
        for (x, y, h, w) in boxes:
            face_img = image[x*4: h*4, y*4: w*4]
            cv2.imwrite(path + "\\trained_dataset\\"+name+"\\"+str(i)+".jpg", face_img)
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            known_encodings.append(encoding)
            known_names.append(name)

    # dump the facial encodings + names to disk
    # print("[INFO] serializing encodings...")
    logging.info("serializing encodings...")
    info = {"encodings": known_encodings, "names": known_names}
    f = open(path + "/encodings.pickle", "wb")
    f.write(pickle.dumps(info))
    f.close()
    for d in dirs:
        directory_path = path + "\\dataset\\" + d
        _, __, files = os.walk(directory_path).__next__()
        file_count = len(files)
        os.chdir(directory_path)
        while file_count > storage_count:
            try:
                file_array = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
                # print(file_array)
                logging.debug(file_array)
                oldest = file_array[0]
                os.remove(directory_path + "\\" + oldest)
                _, __, files = os.walk(directory_path).__next__()
                file_count = len(files)
            except IndexError:
                logging.debug("No extra file found")
                break
    os.chdir(path)


if __name__ == '__main__':
    encode_faces()
