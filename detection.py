import cv2
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret , frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray)
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(frame , (x,y) ,(x+w , y+h)  , (255,255,255) , 4)
        cv2.imshow("video",frame)
    key = cv2.waitKey(1)
    if key & 0xff == ord("q"):
         break


cap.release()
cv2.destroyAllWindows()