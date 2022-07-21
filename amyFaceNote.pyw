#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Vision libraries
import imutils
from imutils import paths
# from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import cv2
import numpy as np
# Voice libraries
import pyttsx3
import winspeech
# import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr
# import wave
# Database libraries
import sqlite3
from sqlite3 import OperationalError
# Time and date libraries
import time
import datetime
# Threading libraries
from threading import Thread, Lock, Timer
from contextlib import contextmanager
import locale
from multiprocessing import Process
from queue import Queue
# Online Checker libraries
import socket
# GUI
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
'''
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
'''
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
# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
# print("[INFO] loading detector files and settings...")
logging.info('loading detector files and settings...')
data = pickle.loads(open(path + '/encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier(path + '/haarcascade_frontalface_default.xml')
# print("[INFO] loading voice settings...")
logging.info('loading voice settings...')
amy_voice = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_enUS_EvaM'
q = Queue()
show = Queue()
# Creates an instance of a speech config with specified subscription key and service region.
# Replace with your own subscription key and service region (e.g., "westus").
"""
speech_key, service_region = "3099615656794b91b2440aa951639419", "CentralUS"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Creates a recognizer with the given settings
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
"""
r = sr.Recognizer()
m = sr.Microphone()
face_no = 0
net_dev = [
    '192.168.0.102',
    '192.168.0.124'
]


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
        self.qt.setWindowIcon(QIcon(path+"\\WS-Icons-03.png"))
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
                line_count = int(char_count/38)
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
    ans4match = dictionary.index(match_percent[0])+1
    match_info = list(match_percent)
    match_info[1] = (fuzz.ratio(question, match_percent[0])+match_percent[1])/2
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

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=800)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

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
        ip = '192.168.0.'+str(ip_id)
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
        #self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
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


def face_identifier(encodings):
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
        identity = "Unknown"
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        best_match_index = np.argmin(face_distances)
        # check to see if we have found a match
        if True in matches or matches[best_match_index]:
            # finding the list of all recognized faces then adding them to a
            # dictionary for counting the total number, when each face
            # was matched
            matched_ids = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matched_ids:
                identity = data["names"][i]
                counts[identity] = counts.get(identity, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            identity = max(counts, key=counts.get)

        # entry the recognized faces into the list
        names.append(identity)
    return names


def get_data(ids):
    db = sqlite3.connect('amy.sqlite')
    sql = db.cursor()
    dict_names = {}
    list_pro = []
    infos = []
    for i in ids:
        try:
            sql.execute("SELECT * FROM CLIENTS where ID='" + i + "'")
            info = sql.fetchone()
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
    db = sqlite3.connect('amy.sqlite')
    sql = db.cursor()
    current_time = datetime.datetime.now()
    date = current_time.strftime("%d-%m-%Y")
    t = current_time.strftime("%H:%M:%S")
    millis = current_time.strftime(":%f")
    timestamp = date + " " + t + millis
    try:
        sql.execute('INSERT INTO LOG (ID, NAME, DATE, TIME, TIMESTAMP, FIRST_ENTRY) '
                    'VALUES (' + repr(eid) + ',\'' + name + '\',\'' + date + '\',\'' + t + '\',\'' + timestamp + '\',\''
                    + str(first_entry) + "')")
        # print("Data Saved Successfully")
        logging.info("Data Saved Successfully")
    except (KeyError, OperationalError) as e:
        logging.debug(e)
        pass
    if first_entry and person_no == 1:
        cv2.imwrite(path + "\\dataset\\" + eid + "\\" + eid +
                    datetime.datetime.now().strftime(" %Y-%m-%d") + '.jpg', frame)
    elif person_no > 1:
        cv2.imwrite(path + "\\employees\\" + eid + datetime.datetime.now().strftime(" %Y-%m-%d") + '.jpg',
                    frame)
    # print("Client: " + name + "\nCaptured at:" + timestamp)
    logging.info("Client: " + name + "\nCaptured at:" + timestamp)
    db.commit()
    return


def time_interval(eid):
    db = sqlite3.connect('amy.sqlite')
    sql = db.cursor()
    time_diff = 0.0
    try:
        sql.execute("SELECT MAX(TIME) FROM LOG WHERE ID='" + eid +
                    "' and DATE='" + datetime.datetime.now().strftime("%d-%m-%Y") + "'")
        last_entry_time = sql.fetchone()
        time_diff = (datetime.datetime.strptime(datetime.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S') -
                     datetime.datetime.strptime(last_entry_time[0], '%H:%M:%S')).seconds / 60
    except(KeyError, TypeError, OperationalError) as e:
        logging.debug(e)
        pass
    return time_diff


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
    db = sqlite3.connect('amy.sqlite')
    sql = db.cursor()
    # start the FPS counter
    fps = FPS().start()
    last_cap = time.time()

    # loop over frames from the video file stream
    try:
        while True:
            first_entries = {}
            # grab the frame from the threaded video stream and resize it
            # to 400px (to speedup processing)
            frame = vs.read()
            '''
            try:
                frame = imutils.resize(frame, width=400)
            except Exception as e:
                logging.debug(e)
            '''
            # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # convert the input frame from (1) BGR to gray scale (for face
            # detection) and (2) from BGR to RGB (for face recognition)
            # gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = frame[:, :, ::-1]

            # detect faces in the gray scale frame
            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(10, 10),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
            global face_no
            face_no = len(rects)
            # update the FPS counter
            fps.update()
            # OpenCV returns bounding box coordinates in (x, y, w, h) order
            # but we need them in (top, right, bottom, left) order, so we
            # need to do a bit of reordering
            if face_no != 0:
                boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

                # compute the facial embeddings for each face bounding box
                encodings = face_recognition.face_encodings(rgb, boxes)
                ids = face_identifier(encodings)
                ids = list(dict.fromkeys(ids))
                # print(ids)
                logging.debug(ids)
                moment = datetime.datetime.now().strftime("%I %M %p").split()
                hr = int(moment[0])
                for i in ids:
                    try:
                        date = datetime.datetime.now().strftime("%d-%m-%Y")
                        sql.execute("SELECT TIME FROM LOG WHERE ID='" + i + "' and DATE='" + date + "'")
                        entry_time = sql.fetchall()
                        # rstate = open("state.txt", "r")
                        # if not entry_time and rstate.read() == '0':
                        if not entry_time:
                            # rstate.close()
                            first_entries[i] = True
                            # open("state.txt", "w").write('1')
                        else:
                            # rstate.close()
                            first_entries[i] = False
                            # open("state.txt", "w").write('0')
                        # open("state.txt", "w").close()
                    except (KeyError, OperationalError) as e:
                        logging.debug(e)
                        pass
                if "Unknown" in first_entries:
                    del first_entries["Unknown"]
                if not LOCALE_LOCK.locked():
                    LOCALE_LOCK.acquire()
                    pronounce, names, infos = get_data(ids)
                if LOCALE_LOCK.locked():
                    LOCALE_LOCK.release()
                name_str = ''
                ac_name_str = ''
                # print(pronounce)
                logging.debug(pronounce)
                if pronounce:
                    if not LOCALE_LOCK.locked():
                        LOCALE_LOCK.acquire()
                        names_no = len(pronounce)
                        if names_no > 1:
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
                        # print("Voice Thread Started")
                        logging.info("Voice thread started")
                        q.put("Hi " + name_str)
                        show.put("Hi " + ac_name_str)
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
                        q.put(greet)
                        show.put(greet)
                        speak()
                    for i in ids:
                        time_inter = time_interval(i)
                        names_no = len(pronounce)
                        interval = round(time.time() - last_cap) / 60
                        # print(i + " interval: {}".format(interval))
                        logging.debug(i + " interval: {}".format(interval))
                        if i != "Unknown" and (time_inter > 5.0 or first_entries[i] is True):
                            save_data(i, names[i], first_entries[i], frame, names_no)
                            if not first_entries[i]:
                                global current
                                current = 'Hey there, {}'.format(names[i])
                                time.sleep(0.5)
                                current = ''
                        elif i == "Unknown" and interval > 1:
                            q.put('Hello There')
                            show.put('Hello There')
                            if 12 > hr >= 7 and moment[2] == "AM":
                                greet = "Good Morning"
                            elif moment[2] == "PM" and (12 == hr or hr <= 3):
                                greet = "Good Afternoon"
                            elif moment[2] == "PM" and 3 < hr <= 7:
                                greet = "Good Evening"
                            else:
                                greet = "Wish you a wonderful day"
                            q.put(greet)
                            show.put(greet)
                            q.put('Welcome to Robot Company')
                            show.put('Welcome to Robot Company')
                            speak()
                            cv2.imwrite(path + "\\Unknown\\" + i +
                                        datetime.datetime.now().strftime(" %Y-%m-%d %H.%M.%S") + '.jpg', frame)
                            last_cap = time.time()
                        else:
                            pass
                    if LOCALE_LOCK.locked():
                        LOCALE_LOCK.release()

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
                # update the FPS counter
                fps.update()

            # display the image to our screen
            cv2.imshow(camera, frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    except Exception:
        connect_cam(net_dev)

    # stop the timer and display FPS information
    fps.stop()
    # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    logging.debug("Elapsed time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    logging.debug("Approx. FPS: {:.2f}".format(fps.fps()))
    # # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    db.close()


def connect_cam(dev):
    for device in dev:
        url = 'rtsp://admin:Ad2729$G@' + device + ':554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
        thread = CamThread("Net Camera {}".format(net_dev.index(device) + 1), url)
        thread.start()


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, lambda *argc: QApplication.quit())
        # camera_number = devices()
        # net_dev = net_devices()
        # Start GUI
        Thread(target=start_qt, args=(1,)).start()
        # Thread(target=conversation).start()
        '''
        while True:
            current = input()
        '''
        '''
        for device in range(camera_number):
            # Create threads as follows
            thread = CamThread("Camera {}".format(device+1), device)
            # Start Camera threads
            thread.start()
        '''
        connect_cam(net_dev)
        '''
        thread = CamThread("Camera 1", 'rtsp://admin:Ad2729$G@192.168.0.107:554/cam/realmonitor?channel=1&subtype=0&
        unicast=true&proto=Onvif')
        thread.start()
        '''

        # Launch face encoder with different PID
        encode_launcher()

    except (KeyboardInterrupt, SystemExit):
        # print("[INFO] Program exited successfully.")
        logging.info("Program exited successfully.")
        _exit(0)
