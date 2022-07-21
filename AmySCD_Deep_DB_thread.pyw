#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Vision libraries
import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import numpy as np
from SceneChangeDetect import sceneChangeDetect
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# Voice libraries
import pyttsx3
import winspeech
# import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr
# import wave
# Database libraries
import sqlite3
from sqlite3 import OperationalError
import mysql.connector
from mysql.connector import Error
# Time and date libraries
import time
import datetime
# Threading libraries
from threading import Thread, Lock, Timer
from contextlib import contextmanager
import locale
from multiprocessing import Process
from queue import Queue
from multiprocessing.pool import ThreadPool
# Online Checker libraries
import socket
# GUI
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# Miscellaneous libraries
from fuzzywuzzy import fuzz, process
# import random
import sys
import os
from os import _exit
import signal
import logging

FORMAT = '%(levelname)s:%(asctime)s:PID:%(thread)d:%(threadName)s:%(funcName)s:%(message)s'
logging.basicConfig(filename='logs\\AMY_log {}.log'.format(datetime.datetime.now().strftime(" %Y-%m-%d %H.%M.%S")),
                    level=logging.DEBUG, format=FORMAT)
storage_count = 60
xlarge_text_size = 48
large_text_size = 28
medium_text_size = 18
small_text_size = 10
font1 = QFont('Helvetica', small_text_size)
font2 = QFont('Helvetica', medium_text_size)
font3 = QFont('Helvetica', xlarge_text_size)
LOCALE_LOCK = Lock()
# find current directory path for loading the files
path = os.path.abspath(os.path.split(sys.argv[0])[0])

# load our serialized face detector from disk
# print("[INFO] loading face detector...")
logging.info('loading detector files and settings...')
protoPath = path + "\\face_detection_model\\" + "deploy.prototxt"
modelPath = path + "\\face_detection_model\\" + "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(path + "\\openface_nn4.small2.v1.t7")
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(path + "\\output\\recognizer.pickle", "rb").read())
le = pickle.loads(open(path + "\\output\\le.pickle", "rb").read())

# print("[INFO] loading voice settings...")
logging.info('loading voice settings...')
amy_voice = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_enUS_EvaM'
q = Queue()
show = Queue()

db = mysql.connector.connect(host="localhost", user="root", passwd="", database="amy-ai")
sql = db.cursor()

r = sr.Recognizer()
m = sr.Microphone()
face_no = 0
Scene = sceneChangeDetect()
pool1 = ThreadPool(processes=1)
pool2 = ThreadPool(processes=2)
pool3 = ThreadPool(processes=3)
pool4 = ThreadPool(processes=4)
pool5 = ThreadPool(processes=5)


@contextmanager
def setlocale(name):  # thread proof function to work with locale
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)


class FullscreenWindow(QMainWindow):
    def __init__(self, parent, *args, **kwargs):
        QMainWindow.__init__(self)
        self.setFixedSize(100, 100)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("Quit Amy?")
        self.setWindowIcon(QIcon(path + "\\vector-logout.png"))
        self.qbutton = QPushButton(self)
        self.qbutton.setIcon(QIcon(path + "\\vector-logout.png"))
        self.qbutton.clicked.connect(self.closeEvent)
        self.qbutton.setIconSize(QSize(100, 100))
        self.qbutton.resize(100, 100)
        self.showMinimized()

        self.parent = parent
        self.qt = QWidget()
        self.qt.showFullScreen()
        # self.qt.resize(window_width, window_height)
        # window icon & title
        self.qt.setWindowIcon(QIcon(path + "\\WS-Icons-03.png"))
        self.qt.setWindowTitle("AMY UI")
        # Hide mouse cursor
        self.qt.setCursor(Qt.BlankCursor)

        self.pal = QPalette()
        self.pal.setColor(QPalette.Background, Qt.black)
        self.pal.setColor(QPalette.Foreground, Qt.white)
        self.qt.setPalette(self.pal)

        self.qt.hbox1 = QHBoxLayout()
        self.qt.messageBox = Message(QWidget())
        self.qt.hbox1.addWidget(self.qt.messageBox)

        self.qt.vbox = QVBoxLayout()

        self.qt.vbox.addLayout(self.qt.hbox1)

        self.qt.setLayout(self.qt.vbox)

        # self.qt.setWindowState(Qt.WindowMaximized)
        # self.qt.showMaximized()

    def passed(self):
        pass

    def keyPressEvent(self, event):
        key = event.key()
        # print(key)
        logging.debug(key)

        if key == Qt.Key_Q:
            self.closeEvent(event)

    def closeEvent(self, event):
        _exit(0)


class Message(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(Message, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.msgLbl = QLabel('')
        self.msgLbl.setAlignment(Qt.AlignCenter)
        self.msgLbl.setFont(font3)
        self.hbox = QHBoxLayout()
        self.now = ''
        self.hbox.addWidget(self.msgLbl)
        self.setLayout(self.hbox)
        self.update_check()

    def update_check(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_message)
        self.timer.start(500)

    def update_message(self):
        global current
        if current != self.now:
            char_count = len(current)
            if char_count > 38:
                line_count = int(char_count / 38)
                for index in range(line_count):
                    trace = 38 * (index + 1)
                    while current[trace] != ' ':
                        trace -= 1
                    current = current[:trace] + '\n' + current[trace + 1:]
            self.msgLbl.setText(current)
            self.now = current


def start_qt(tmp):
    if QApplication.instance() is None:
        a = QApplication(sys.argv)
        w = FullscreenWindow(a)
        w.passed()
        sys.exit(a.exec_())
    else:
        _exit(0)


current = ''


def speak():
    global current
    amy = pyttsx3.init()
    amy.setProperty('rate', 140)
    amy.setProperty('volume', 1)
    amy.setProperty('voice', amy_voice)
    while not q.empty():
        word = q.get()
        current = show.get()
        # print(word)
        logging.debug(word)
        amy.say(word)
        try:
            amy.runAndWait()
        except RuntimeError as e:
            # print("Error: {}".format(e))
            logging.error("{}".format(e))
        q.task_done()
        show.task_done()
    current = ''


def greeting(name_str, ac_name_str, frame=None):
    moment = datetime.datetime.now().strftime("%I %M %p").split()
    hr = int(moment[0])
    # print("Voice Thread Started")
    logging.info("Voice thread started")
    if name_str == "Unknown":
        q.put('Hello There')
        show.put('Hello There')
    else:
        q.put("Hi " + name_str)
        show.put("Hi " + ac_name_str)
    if name_str != "Unknown":
        q.put("It's " + moment[0] + ":" + moment[1] + " " + moment[2])
        show.put("It's " + moment[0] + ":" + moment[1] + " " + moment[2])
    if 12 > hr >= 7 and moment[2] == "AM":
        greet = "Good Morning"
    elif moment[2] == "PM" and (12 == hr or hr <= 3):
        greet = "Good Afternoon"
    elif moment[2] == "PM" and 3 < hr <= 7:
        greet = "Good Evening"
    else:
        greet = "Wish you a wonderful day"
    if name_str == "Unknown":
        q.put('Welcome to Robot Company')
        show.put('Welcome to Robot Company')
        cv2.imwrite(path + "\\Unknown\\" + "Unknown" + datetime.datetime.now().strftime(" %Y-%m-%d %H.%M.%S") + '.jpg',
                    frame)
    q.put(greet)
    show.put(greet)
    speak()


'''
def internet_on(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


class STT:
    def __init__(self):
        self.word = ""

    def off_recognizer(self, report, listened):
        self.word = report
        listened.stop_listening()

    def main_recognizer(self):
        if internet_on():
            with m as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                # recognize speech using Wit.ai
                WIT_AI_KEY = "PSI7WZXZMEEUKTDXE22F5QOSQU6YFI46"  # Wit.ai keys are 32-character uppercase alphanumeric strings
                try:
                    value = r.recognize_wit(audio, key=WIT_AI_KEY)
                    if str is bytes:
                        self.word = u"{}".format(value).encode("utf-8")

                    else:
                        self.word = "{}".format(value)
                except sr.UnknownValueError:
                    print("")
                except sr.RequestError as e:
                    print("{0}".format(e))
            """
            while self.word == "":
                result = speech_recognizer.recognize_once()
                # Checks result.
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    self.word = result.text
                else:
                    self.word = ""
            """
        else:
            winspeech.initialize_recognizer(winspeech.INPROC_RECOGNIZER)
            listener = winspeech.listen_for_anything(self.off_recognizer)
            while listener.is_listening():
                continue
        if self.word:
            return self.word


def match_counter(question):
    dictionary = [line.rstrip('\n') for line in open('qa.txt', 'r')]
    match_percent = process.extractOne(question, dictionary)
    ans4match = dictionary.index(match_percent[0]) + 1
    match_info = list(match_percent)
    match_info[1] = (fuzz.ratio(question, match_percent[0]) + match_percent[1]) / 2
    match_info.append(dictionary)
    match_info.append(ans4match)
    return match_info



def conversation():
    hotword_amy = ["amy", "me", "ami", "ammi", "any", "baby", "army"]
    punc = str.maketrans(dict.fromkeys("₲⧸–❧¥؋’₼‹₹₵﷼$‰‴₭±₮¢“”元¶៛)⁀―™ฯ'◊、"":₦℗৳~›₡»₯«#ƒ₨،฿_/‽.=⋯圓"
                                       "₫\"₢₱-+!֏‖©§¿£₧​″(※—%|¦¤₳⟨∓₩№−🄯₿‘,᠁₥;‐☞℠‒〃₪⟩?⁂⸮₠ℳª÷[′¡₰₸円₺}×{º®…₣°₾₽⁄]€₴圆"))
    wakeup = False
    last_cap = time.time()
    find_hw = {}
    try:
        while True:
            s = STT()
            if s != "":
                try:
                    interval = round(time.time() - last_cap) / 60
                    question = s.main_recognizer().lower()
                    print(question)
                    hotword = question.translate(punc).split()
                    find_hw = set(hotword).intersection(set(hotword_amy))
                    logging.debug(question)
                except Exception as e:
                    print(e)
            if find_hw or wakeup:
                if find_hw:
                    last_cap = time.time()
                    wakeup = True
                logging.debug("Hotword Found")
                match, match_percent, dictionary, ans4match = match_counter(question)
                # if match_percent >= 70 and answer not in ambiguous:
                if match_percent >= 60:
                    # speak(dictionary[ans4match])
                    ans_by_line = dictionary[ans4match].split(". ")
                    for x in ans_by_line:
                        q.put(x)
                        show.put(x)
                    speak()
                elif question.find('time'):
                    moment = datetime.datetime.now().strftime("%I %M %p").split()
                    q.put("It's " + moment[0] + ":" + moment[1] + " " + moment[2])
                    show.put("It's " + moment[0] + ":" + moment[1] + " " + moment[2])
                    speak()
                elif question.find('date'):
                    dt = datetime.datetime.now().date()
                    day = dt.strftime("%A")
                    mon = dt.strftime("%B")
                    date = dt.strftime("%d")
                    x = "Today is "+day+" "+mon+" "+date
                    q.put(x)
                    show.put(x)
                    speak()
                else:
                    q.put("I don't know that one. But, I will be learning that any sooner.")
                    show.put("I don't know that one. But, I will be learning that any sooner.")
                    speak()
                    learn_recorder = open("Learner.txt", "a")
                    learn_recorder.write(question + "\n")
                    learn_recorder.close()
            if interval > 1.0:
                wakeup = False
    except Exception as e:
        logging.warning(e)
'''


def encode_faces():
    # print("[INFO] started encoding faces...")
    logging.info('started encoding faces...')
    # print("[INFO] quantifying faces...")
    logging.debug('quantifying faces...')
    _, dirs, __ = os.walk(path + "\\dataset\\").__next__()
    image_paths = list(paths.list_images(path + "\\dataset\\"))

    # initialize the list of known encodings and known names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        logging.debug("processing image {}/{}".format(i + 1, len(image_paths)))
        name = imagePath.split(os.path.sep)[-2]
        if not os.path.isdir(path + "\\trained_dataset\\" + name):
            os.mkdir(path + "\\trained_dataset\\" + name)

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) == 1:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            im = np.argmax(detections[0, 0, :, 2])
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, im, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, im, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]
                cv2.imwrite(path + "\\trained_dataset\\" + name + "\\" + name + " " + str(i + 1) + ".jpg", face)
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                    # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial encodings + names to disk
    # print("[INFO] serializing {} encodings...".format(total))
    logging.info("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(path + "/encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    model_trainer()
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


def model_trainer():
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(path + "\\output\\encodings.pickle", "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(path + "\\output\\recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(path + "\\output\\le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()


# Launcher for encode_faces() module using Timer
def encode_launcher():
    encode = Process(target=encode_faces())
    encode.start()
    encode.join()
    # run every 12 hr
    Timer(12 * 3600.0, encode_launcher).start()


# Imaging Device counter
def devices():
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            index += 1
            cap.release()
            cv2.destroyAllWindows()
        else:
            break
    return index


def net_devices():
    ip_list = []
    ip_limit = 255
    for ip_id in range(ip_limit):
        ip = '192.168.0.' + str(ip_id)
        url = 'rtsp://admin:Ad2729$G@' + ip + ':554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
        cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)
        if cap.isOpened():
            ip_list.append(ip)
            cap.release()
            cv2.destroyAllWindows()
    return ip_list


# Multiple camera thread launcher
class CamThread(Thread):
    def __init__(self, preview_name, cam_id):
        Thread.__init__(self)
        self.previewName = preview_name
        self.camID = cam_id

    def run(self):
        # print("Starting " + self.previewName)
        logging.info("Starting " + self.previewName)
        face_stream(self.previewName, self.camID)


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        # self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def isOpened(self):
        return self.stream.isOpened()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def get_data(ids):
    dict_names = {}
    list_pro = []
    infos = []
    for i in ids:
        try:
            sql.execute("SELECT * FROM CLIENTS where ID='" + i + "'")
            info = sql.fetchone()
            info = [x.decode('utf-8') if x is not None else x for x in info]
            if info[2] is not None:
                list_pro.append(info[2])
            else:
                list_pro.append(info[1])
            dict_names[i] = info[1]
            infos.append(info)
        except (KeyError, TypeError, OperationalError) as e:
            logging.debug(e)
            pass
    return list_pro, dict_names, infos


def save_data(eid, name, first_entry, frame, person_no):
    # db = mysql.connector.connect(host="localhost", user="root", passwd="", database="amy-ai")
    # sql = db.cursor()
    current_time = datetime.datetime.now()
    date = current_time.strftime("%Y-%m-%d")
    t = current_time.strftime("%H:%M:%S")
    # millis = current_time.strftime(":%f")
    timestamp = date + " " + t
    if first_entry:
        fe = 1
    else:
        fe = 0
    try:
        sql.execute("INSERT INTO LOG (ID, NAME, DATE, TIME, FIRST_ENTRY) "
                    'VALUES (' + repr(
            eid) + ',\'' + name + '\',\'' + date + '\',\'' + t + '\',\'' +
                    repr(fe) + "')")
        # print("Data Saved Successfully")
        logging.info("Data Saved Successfully")
    except (KeyError, OperationalError) as e:
        logging.debug(e)
        pass
    if first_entry and person_no == 1:
        cv2.imwrite(path + "\\dataset\\" + eid + "\\" + eid + datetime.datetime.now().strftime(" %Y-%m-%d") + '.jpg',
                    frame)
    elif person_no > 1:
        cv2.imwrite(path + "\\employees\\" + eid + datetime.datetime.now().strftime(" %Y-%m-%d") + '.jpg', frame)
    # print("Client: " + name + "\nCaptured at:" + timestamp)
    logging.info("Client: " + name + "\nCaptured at:" + timestamp)
    db.commit()
    return


def time_interval(eid):
    # db = mysql.connector.connect(host="localhost", user="root", passwd="", database="amy-ai")
    # sql = db.cursor()
    time_diff = 0.0
    try:
        sql.execute("SELECT MAX(TIME) FROM LOG WHERE ID='" + eid + "' and DATE='" + datetime.datetime.now().strftime(
            "%Y-%m-%d") + "'")
        last_entry_time = sql.fetchone()
        if last_entry_time[0] is not None:
            time_diff = (datetime.datetime.strptime(datetime.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S') -
                         datetime.datetime.strptime(str(last_entry_time[0]), '%H:%M:%S')).seconds / 60
    except(KeyError, TypeError, OperationalError) as e:
        logging.debug(e)
        pass
    return time_diff


def entry_monitor(ids):
    first_entries = {}
    # db = mysql.connector.connect(host="localhost", user="root", passwd="", database="amy-ai")
    # sql = db.cursor()
    for i in ids:
        try:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            sql.execute("SELECT TIME FROM LOG WHERE ID='" + i + "' and DATE='" + date + "'")
            entry_time = sql.fetchall()
            # rstate = open("state.txt", "r")
            # if not entry_time and rstate.read() == '0':
            if not entry_time:
                # rstate.close()
                first_entries[i] = True  # open("state.txt", "w").write('1')
            else:
                # rstate.close()
                first_entries[i] = False  # open("state.txt", "w").write('0')  # open("state.txt", "w").close()
        except (KeyError, OperationalError) as e:
            logging.debug(e)
            pass
    if "Unknown" in first_entries:
        del first_entries["Unknown"]
    # db.close()
    return first_entries


def face_stream(camera, src):
    # cv2.namedWindow(camera)
    # print("[INFO] starting imaging device...")
    logging.info("starting imaging device...")
    vs = WebcamVideoStream(src=src).start()
    # vs = VideoStream(src).start()
    # frame = vs.read()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    # print("[INFO] Finding faces..")
    logging.info("Finding faces..")
    # start the FPS counter
    last_cap = time.time()
    pre_ids = {}
    count = {}
    id_strength = 0
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 600px (to speedup processing)
        ids = {}
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if Scene.detectChange(gray):
            # detect faces in the gray scale frame
            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()
            if len(detections) > 0:
                pronounce = ''
                names = ''
                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections
                    if confidence > 0.5:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue

                            # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()
                        # perform classification to recognize the face
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        probability = preds[j]
                        name = le.classes_[j]
                        if probability > 30:
                            ids[name] = probability
                if not pre_ids:
                    pre_ids = ids.copy()
                    continue
                else:
                    for item in set(pre_ids).intersection(set(ids)):
                        if item in count:
                            count[item] += 1
                        else:
                            count[item] = 1

                # logging.debug(count)
                print(count)
                '''
                interval = round(time.time() - last_cap) / 60
                if len(ids) == 1 and ids[0] == "Unknown" and interval > 1:
                    # greet_unknown(frame)
                    last_cap = time.time()
                else:
                    first_entries = pool5.apply_async(entry_monitor, (ids,)).get()
                    if not LOCALE_LOCK.locked():
                        # LOCALE_LOCK.acquire()
                        pronounce, names, infos = pool4.apply_async(get_data, (ids,)).get()
                    if LOCALE_LOCK.locked():
                        LOCALE_LOCK.release()
                    # print(pronounce)
                    logging.debug(pronounce)
                    if pronounce:
                        name_str = ''
                        ac_name_str = ''
                        if not LOCALE_LOCK.locked():
                            # LOCALE_LOCK.acquire()
                            if len(pronounce) > 1:
                                temp_id = ids.copy()
                                last_name = pronounce.pop(-1)
                                last_name_id = temp_id.pop(-1)
                                for name in pronounce:
                                    name_str += ', ' + name
                                name_str += ' and ' + last_name
                                try:
                                    for i in temp_id:
                                        ac_name_str += ', ' + names[i]
                                    ac_name_str += ' and ' + names[last_name_id]
                                except KeyError as e:
                                    logging.debug(e)
                                    pass
                            else:
                                try:
                                    name_str = pronounce[0]
                                    ac_name_str = names[ids[0]]
                                except KeyError as e:
                                    logging.debug(e)
                                    pass
                            # print("Actual Name: ")
                            logging.debug("First Entry Condition: {}".format(first_entries))
                            logging.debug("Actual Name: {} ".format(ac_name_str))
                        if True in first_entries.values():
                            greet = Thread(target=greeting, args=(name_str, ac_name_str,), daemon=True)
                            greet.start()
                            # greet.join()
                        for i in ids:
                            time_inter = pool3.apply_async(time_interval, (i,)).get()
                            names_no = len(pronounce)
                            # print(i + " interval: {}".format(interval))
                            logging.debug(i + " interval: {}".format(interval))
                            if i != "Unknown" and (time_inter > 2.0 or first_entries[i] is True):
                                save_data(i, names[i], first_entries[i], frame, names_no)
                                if not first_entries[i]:
                                    global current
                                    current = 'Hey there, {}'.format(names[i])
                                    time.sleep(1)
                                    current = ''
                            elif i == "Unknown" and interval > 1:
                                greet = Thread(target=greeting, args=("Unknown", "Unknown", frame), daemon=True)
                                greet.start()
                                # greet.join()
                                last_cap = time.time()
                            else:
                                pass
                        if LOCALE_LOCK.locked():
                            LOCALE_LOCK.release()
                        '''
                '''
                # initiate a frame multiplyer for resizing the frame back to core
                f_mul = 1
                # loop over the recognized faces
                for ((top, side_right, bottom, side_left), name) in zip(boxes, names):
                    # draw the predicted face name on the image
                    cv2.rectangle(frame, (side_left * f_mul, top * f_mul), (side_right * f_mul, bottom * f_mul),
                                  (0, 255, 0), 2)
                    y = top * 4 - 15 if top * f_mul - 15 > 15 else top * f_mul + 15
                    cv2.putText(frame, name, (side_left * f_mul, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)
                '''

        # display the image to our screen
        cv2.imshow(camera, frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, lambda *argc: QApplication.quit())
        camera_number = devices()
        # net_dev = net_devices()
        net_dev = ['192.168.0.109', '192.168.0.111']
        # Start GUI
        # Thread(target=start_qt, args=(1,)).start()
        # Thread(target=conversation).start()
        '''
        while True:
            current = input()
        '''

        for device in range(camera_number):
            # Create threads as follows
            thread = CamThread("Camera {}".format(device + 1), device)
            # Start Camera threads
            thread.start()

        for device in net_dev:
            url = 'rtsp://admin:Ad2729$G@' + device + ':554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
            thread = CamThread("Net Camera {}".format(net_dev.index(device) + 1), url)
            thread.start()
        '''
        thread = CamThread("Camera 1", 'rtsp://admin:Ad2729$G@192.168.0.107:554/cam/realmonitor?channel=1&subtype=0&
        unicast=true&proto=Onvif')
        thread.start()
        '''

        # Launch face encoder with different PID
        # encode_launcher()

    except (KeyboardInterrupt, SystemExit):
        # print("[INFO] Program exited successfully.")
        logging.info("Program exited successfully.")
        _exit(0)
