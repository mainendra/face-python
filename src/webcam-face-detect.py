import cv2

# Use preprocessed data for lookup
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# Default webcam is at index 0
webcam = cv2.VideoCapture(0)
# Keep reading fram from webcam
while True:
    success, frame = webcam.read()
    img_flip = cv2.flip(frame, 1)
    # Convert frame to gray scale
    grayImg = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
    # Draw rectangle on matched face
    for (x, y, w, h) in face_cascade.detectMultiScale(grayImg):
        cv2.rectangle(img_flip, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # Show frame
    cv2.imshow("Video", img_flip)
    # Wait for 1ms or `q` key press
    if cv2.waitKey(1) == ord('q'):
        break
# Release webacam stream
webcam.release()

