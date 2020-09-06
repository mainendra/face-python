import cv2

# Use processed data for lookup
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Choose image to test
img = cv2.imread('data/images/ak.png')
# Convert to gray scale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Draw reactagle on matched coordinates on colored image
for (x, y, w, h) in face_cascade.detectMultiScale(grayImg):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
# Show colored image
cv2.imshow('Face detect', img)
# Wait for any keypress before exit
cv2.waitKey(0)
