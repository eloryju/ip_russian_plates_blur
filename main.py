import cv2
import matplotlib.pyplot as plt

plate = cv2.imread('C:\\Users\\memil\\OneDrive\\Pulpit\\py4e\\licence_blur\\sample\\Russian.jpg')
plate = plate.astype('uint8')

def display(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


plate_cascade = cv2.CascadeClassifier('C:\\Users\\memil\\OneDrive\\Pulpit\\py4e\\licence_blur\\sample\\haarcascade_russian_plate_number.xml')

def detect_and_blur_plate(img):
    
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in plate_rects:
        x_end = x + w
        y_end = y + h
        plate_img[y:y_end, x:x_end] = cv2.medianBlur(plate_img[y:y_end, x:x_end],9)
    return plate_img

display(detect_and_blur_plate(plate))

