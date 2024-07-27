import cv2
import sys
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi


import dlib
from imutils import face_utils
import winsound
import win32com.client as wincl
import serial
import serial.tools.list_ports
import warnings
import time


class SerialPortDataProcessThread(QThread):

    # system_status = pyqtSignal(bool)

    def __init__(self, serialPort):
        super(SerialPortDataProcessThread, self).__init__()
        self.serialPort = serialPort

    def run(self):
        self.exec_()

    def Close_Serial(self):
        self.serialPort.close()

    @pyqtSlot(object)
    def writeData(self, data):
        data_encoded = data.encode("utf-8")
        self.serialPort.write(data_encoded)


def find_Arduino_Com_Port():
    for p in serial.tools.list_ports.comports():
        print(p.description)

    arduino_ports = [
        p.device
        for p in serial.tools.list_ports.comports()
        if 'USB-SERIAL CH340' in p.description
    ]
    if not arduino_ports:
        raise IOError("No Arduino found")
    if len(arduino_ports) > 1:
        warnings.warn('Multiple Arduino found - using the first')

    port_str = arduino_ports[0]
    return port_str


class DriverAlertSystem_Main(QMainWindow):
    serialWrite = QtCore.pyqtSignal(object)

    def __init__(self):
        super(DriverAlertSystem_Main, self).__init__()

        loadUi('MainWindow_gui.ui', self)

        self.comm_port = find_Arduino_Com_Port()
        self.serialArduino = serial.Serial(port=self.comm_port, baudrate=9600)      # RS232 - asynch

        self.SerialThread = SerialPortDataProcessThread(self.serialArduino)
        # self.SerialThread.system_status.connect(self.updateSystemStatus)

        self.serialWrite.connect(self.SerialThread.writeData)

        self.openCameraButton.clicked.connect(self.openCameraClicked)
        self.stopCameraButton.clicked.connect(self.stopCameraClicked)
        self.startDetectionButton.clicked.connect(self.startAllDetection)
        self.stopDetectionButton.clicked.connect(self.stopAllDetection)
        self.exitButton.clicked.connect(self.exitClicked)

        print("[INFO] loading facial landmark predictor...")
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 16
        self.COUNTER = 0
        self.YAWN_COUNTER = 0
        self.FACE_COUNTER = 0
        self.ALARM_ON = False
        self.start_detection_Flag = False
        self.frequency = 2500
        self.duration = 500
        self.speak = wincl.Dispatch("SAPI.SpVoice")
        self.systemStatusFlag = False

    @pyqtSlot(bool)
    def updateSystemStatus(self, status):
        self.systemStatusFlag = status

    @pyqtSlot()
    def startAllDetection(self):
        self.start_detection_Flag = True
        self.SerialThread.start()

    @pyqtSlot()
    def stopAllDetection(self):
        self.start_detection_Flag = False
        if self.SerialThread.isRunning():
            self.SerialThread.terminate()

    def sendData(self, buff):
        self.serialWrite.emit(buff)

    def blinkDetector(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # print(rects)

        if len(rects) != 0:
            self.FACE_COUNTER = 0

            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                mouthshape = shape[self.mStart:self.mEnd]
                mouthOpenDistance = self.euclidean_dist(mouthshape[18], mouthshape[14])
                # print(mouthOpenDistance)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    # print('Counter:',self.COUNTER)

                    # if the eyes were closed for a sufficient number of
                    # frames, then sound the alarm
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            # print("ALARM_ON")
                            # winsound.Beep(self.frequency, self.duration)
                            self.sendData('a')

                        # draw an alarm on the frame
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        self.DisplayImage(img, 1)

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold, so reset the counter and alarm
                else:
                    self.COUNTER = 0
                    self.ALARM_ON = False
                    self.sendData('x')

                if mouthOpenDistance > 4:
                    self.YAWN_COUNTER += 1

                    if self.YAWN_COUNTER >= 15:
                        print('Driver is Yawning !')

                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            # print("ALARM_ON")
                            # winsound.Beep(self.frequency, self.duration)

                            # self.speak.Speak("you are feeling sleepy, please refrain from driving and refresh yourself")
                            self.sendData('b')

                            # draw an alarm on the frame
                            cv2.putText(img, "Yawning ALERT!", (100, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            self.DisplayImage(img, 1)

                else:
                    self.YAWN_COUNTER = 0
                    self.ALARM_ON = False
                    self.sendData('x')

                    # draw the computed eye aspect ratio on the frame to help
                    # with debugging and setting the correct eye aspect ratio
                    # thresholds and frame counters
                cv2.putText(img, "EAR: {:.3f}".format(ear), (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                self.DisplayImage(img, 1)

        else:
            self.FACE_COUNTER += 1

            if self.FACE_COUNTER >= 15:
                print('Driver Not AWAKE !')
                # self.speak.Speak("Wake UP")
                self.sendData('c')

                cv2.putText(img, "Driver Not AWAKE !", (100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                self.DisplayImage(img, 1)

    def parse_Serial_data(self):
        pass

    def euclidean_dist(self, ptA, ptB):
        # compute and return the euclidean distance between the two
        # points
        return np.linalg.norm(ptA - ptB)

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates

        A = self.euclidean_dist(eye[1], eye[5])
        B = self.euclidean_dist(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = self.euclidean_dist(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    @pyqtSlot()
    def openCameraClicked(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.start_detection_Flag:
            self.blinkDetector(self.image)
        else:
            self.DisplayImage(self.image, 1)

    @pyqtSlot()
    def stopCameraClicked(self):
        self.timer.stop()
        self.capture.release()

    def DisplayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

    @pyqtSlot()
    def exitClicked(self):
        self.SerialThread.Close_Serial()
        QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DriverAlertSystem_Main()
    window.show()
    app.exec_()
