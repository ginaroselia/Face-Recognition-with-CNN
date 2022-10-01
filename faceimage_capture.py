# Opens the inbuilt camera of laptop to capture video.
import cv2
import os
import glob

cap = cv2.VideoCapture(0)
i = 0

#change to full name here to create a new class, eg. Gina_Roselia.
newfoldername="user_fullname"
#change to rename created image samples with the short name, eg. Gina.
username = "user_shortname"

parentPath="C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//dataset"
userPath=os.path.join(parentPath, newfoldername)
os.mkdir(userPath)

dimension = (640,480) 
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    else:
        cv2.imshow('Capture Camera', frame)
        cv2.imwrite((os.path.join(userPath,username+str(i)+'.jpg')), cv2.resize(frame, dimension,interpolation=cv2.INTER_AREA))
        i += 1
        
        #capture 90 images, the more the better
        if (i+1)%90==0:
            break
        elif cv2.waitKey(1) == ord('q'):
            break
print(frame.size)

cap.release()
cv2.destroyAllWindows()

face_cascade=cv2.CascadeClassifier('C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//haarcascade_frontalface_default.xml')
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

for img in glob.glob(userPath+'/*.*'):
    var_img = cv2.imread(img)
    face = detect_face(var_img)
    print(face)
    if (len(face) == 0):
        continue
    for(ex, ey, ew, eh) in face:
        crop_image = var_img[ey:ey+eh, ex:ex+ew]
        cv2.imshow("Cropping...", crop_image)
    cv2.imwrite(os.path.join("outputs/",str(img)),crop_image)