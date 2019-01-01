from human_detector import DetectorAPI
import cv2

detector = DetectorAPI("mobilenet_v2lite.pb") # pass in your path here

img = cv2.imread('sample_img2.jpg') # img provided. please use for sample purpose only :)

# Switch to processFrame to get the boxes and scores programmatically, and filter the classes = 1 (person class)
processedImg = detector.processImage(img)

# write results to a sample file, do whatever you need here
cv2.imwrite('sample_result.png',processedImg)

detector.close()
