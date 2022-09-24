from retinaface import RetinaFace
import cv2
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.123.12:8080/?action=stream")
#img = "Picture2.jpg"
while True:
    _, img = cap.read()
    resp = RetinaFace.detect_faces(img)
    if len(resp) > 0:
        for face, details in resp.items():
            # print(f"face: {face}, {details['facial_area']}")
            x1, y1, x2, y2 = tuple(details['facial_area'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #if count == 3:
        #    break
        #print("Person counted: {}")
    img = cv2.resize(img, (1920, 1080))
    cv2.putText(img, "Number of face detected: " + str(len(resp)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)

    cv2.imshow("video", img)
    cv2.waitKey(30)
