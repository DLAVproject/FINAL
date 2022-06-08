# yolov4-deepsort

__TO TA__

- We uploaded the whole repo as the detector.py has dependencies all over it. Milestone 1 can be found in milestone1.ipynb or milestone1.py, milestone 2 in milestone2.ipynb, and milestone 3 in the detector.py file. The report is found as report.pdf. Please, let us know if you have any questions!
- The notebooks for the milestones use OpenCV for webcam capturing. So you should run them locally, not in Colab.
- necessary packages can be installed by running `pip install -r requirements.txt`

Below, you see two example images from our milestone 1 initialization with pose estimation:

![Alt text](/images/1.jpg?raw=true "Title")
![Alt text](/images/2.jpg?raw=true "Title")


This repo is forked from: https://github.com/theAIGuysCode/yolov4-deepsort to customize for an own application. We use the DeepSort with Yolo algorithm to create a tracker which is partcipating in the Tandem Race at EPFL as part of the course CIVIL-459: Deep Learning for autonomous vehicles. The relevant additions can be found in the file detector.py, which is the file that runs the tracker on the Loomo. 

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * https://github.com/theAIGuysCode/yolov4-deepsort
