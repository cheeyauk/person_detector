# Person Detector

This is a minimal wrapper around tensorflow Zoo object detection model to detect person(s) in an image. This repo exposes a class *DetectorAPI* which can be easily integrated for most usages. 

__Configurations__

This repo uses ssdlite_mobilenet_v2_coco for person detection due to its support for lightweight applications and reasonable accuracy. If you can afford heavier processing (GPU) and need higher accuracy, please refer to the tensorflows object detection repo and download the relevant pb (RCNN, SSD ResNet) files and replace the path in the code

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

__Results__

*sample.py* is a sample file on how to call the packaged library and return the processed image. For usage that requires output of bounding boxes instead of annotated image, please use *processFrame()* instead.

Single person in image
<img src="/sample_result.png?" width="500"/>

Multiple persons in image
<img src="/sample_result_mult.png?" width="500"/>
