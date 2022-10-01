#To extract face images from available dataset
import cv2
import os
import glob
  
# Read the input image
face_cascade=cv2.CascadeClassifier('C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//haarcascade_frontalface_default.xml')

#change path for different classes
filename = "C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//dataset//user_fullname"

# Detect faces
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #face_cascade = cv2.CascadeClassifier('opencvfiles/lbpcascade_frontalface.xmlv')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

for img in glob.glob(filename+'/*.*'):
    var_img = cv2.imread(img)
    face = detect_face(var_img)
    print(face)
    if (len(face) == 0):
        continue
    for(ex, ey, ew, eh) in face:
        crop_image = var_img[ey:ey+eh, ex:ex+ew]
        cv2.imshow("Cropping...", crop_image)
    cv2.imwrite(os.path.join("outputs/",str(img)),crop_image)