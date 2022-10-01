Folder/file name and its content:

1. dataset
- a modified dataset used for FYP. Consists of cropped images of me
and certain classes from the original deep funneled LFW dataset.

2. lfw.tgz
- original deep funneled LFW dataset from the website:
http://vis-www.cs.umass.edu/lfw/

3. haarcascade_frontalface_default.xml
- haar cascades used for cropping images (face detection) and
 live face recognition, specifically in Python files 4, 5 and 6.

4. FaceRecognitionSystem.py
- main codes for face recognition model, including image preprocessing, CNN 
model training and testing, and live face recognition.
- when the system runs, figures are expected to pop out. Close them to continue.

5. faceimage_capture.py
- codes to create a personal dataset with cropped images of yourself via webcam.
Make sure to rename the new folder (newfoldername) and its samples (username).

6. face_extractor.py
- codes to modify existing dataset by cropping image samples of a single
class/folder. Make sure the path of folder is correct.

7. Model_B (Will appear after running Python file(4))
- a new folder for CNN model will be created IF the model path in the file(4)
changes.



Instructions:
a. execute file(5) to create a folder of images of yourself. If you have your 
own folder of images available in local drive, execute file(6) instead.

b.execute file(4) to run the face recognition system. 



NOTE:
Make sure the directories in all python files are correct. The current parent
path is C:\Users\User\Desktop\GINA_FYP2_FormalSubmission\FaceRec_Materials   



