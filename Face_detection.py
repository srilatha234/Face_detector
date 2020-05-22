import cv2

###########

Face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


img = cv2.imread("img1520089173862 (2).jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = Face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for x, y, w, h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,x+h),(0,255,0),3)

print(type(faces))
print(faces)

resized = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/3)))
cv2.imshow("Face Detected",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
