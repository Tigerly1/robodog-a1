import cv2
from retinaface import RetinaFace
import numpy as np
import time
import screeninfo


np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        ##################################
        self.cap = cv2.VideoCapture(self.videoPath)
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        self.windowName = "Camera"
        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        print(self.classesList)

    def onVideo(self):
        end = 0
        if (self.cap.isOpened()==False):
            print("Error opening file..")
            return

        (success, image) = self.cap.read()
        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.6, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelId = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelId]
                    classColor = [int(c) for c in self.colorList[classLabelId]]

                    displayText = "{}".format(classLabel)

                    x,y,w,h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x+5,y+15), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    ################################################################################

                    lineWidth = 30
                    cv2.line(image, (x,y), (x + lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=5)

                    cv2.line(image, (x+w, y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x+w, y), (x+w, y + lineWidth), classColor, thickness=5)

                    cv2.line(image, (x, y+h), (x + lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x, y+h), (x, y + h - lineWidth), classColor, thickness=5)

                    cv2.line(image, (x + w, y+h), (x + w - lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x + w, y+h), (x + w, y + h - lineWidth), classColor, thickness=5)

            screen = screeninfo.get_monitors()[0]
            cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(self.windowName, screen.x - 1, screen.y - 1)
            cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            image = cv2.resize(image, (1920, 1080))
            cv2.imshow(self.windowName, image)

            end = self.changeView()
            if end > 0.5:
                break

            (success, image) = self.cap.read()


        if end == 1 :
                return
        elif end == 2:
            self.VideoFaceDetection()

    def VideoFaceDetectionLow(self):

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        end = 0
        _, img = self.cap.read()
        while True:
            # Read the frame


            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display
            img = cv2.resize(img, (1920, 1080))
            cv2.putText(img, "Number of face detected: " + str(len(faces)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 4,
                        (0, 255, 0), 4)
            cv2.imshow(self.windowName, img)

            end = self.changeView()
            if end > 0:
                break

            _, img = self.cap.read()

        if end == 1:
            return
        elif end == 2:
            self.onVideo()

    def VideoFaceDetection(self):
        while True:
            _, img = self.cap.read()
            resp = RetinaFace.detect_faces(img)
            if len(resp) > 0:
                for face, details in resp.items():
                    # print(f"face: {face}, {details['facial_area']}")
                    x1, y1, x2, y2 = tuple(details['facial_area'])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # if count == 3:
                #    break
                # print("Person counted: {}")
            img = cv2.resize(img, (1920, 1080))
            cv2.putText(img, "Number of face detected: " + str(len(resp)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 4,
                        (0, 255, 0), 4)

            cv2.imshow(self.windowName, img)

            end = self.changeView()
            if end > 0:
                break

            _, img = self.cap.read()

        if end == 1:
            return
        elif end == 2:
            self.onVideo()

    def changeView(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e"):
            return 2
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 1

        return 0