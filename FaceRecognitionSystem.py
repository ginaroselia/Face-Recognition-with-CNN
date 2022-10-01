import cv2
import os
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

##1. IMPORT LFW DATASET AND FACE DETECTION
face_cascade=cv2.CascadeClassifier('C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//haarcascade_frontalface_default.xml')

#to show progress of reading data
def print_progress(val, val_len, folder, bar_size=20):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] (%d samples)\t label : %s \t\t" % (progr, val+1, folder), end="\r")

dataset_folder = "C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//dataset"

names = []
images = []
for files in os.listdir(dataset_folder):
    samples = os.listdir(os.path.join(dataset_folder, files))[:150]
    if len(samples)<50 :
        #only classes with 50 samples above are selected for training
       continue
       
    for i, name in enumerate(samples): 
        if name.find(".jpg") > -1 :
            img = cv2.imread(os.path.join(dataset_folder, files, name))
            face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.resize(face_img, (50, 50))
            if face_img is not None :
              images.append(face_img)
              names.append(files)

              print_progress(i, len(samples), files)

print("number of samples :", len(names))
print("shape of img :", face_img.shape)
plt.imshow(images[0], cmap="gray")

##2. AUGMENTATION / PREPROCESSING

def img_augmentation(face_img):
    h, w = face_img.shape
    center = (w // 2, h // 2)

    #cv2.getrotationMatrix2D(center, angle, scale)==
    M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
    M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])
    
    imgs = []
    imgs.append(cv2.warpAffine(face_img, M_rot_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_rot_neg_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_rot_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_rot_neg_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_neg_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_neg_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_neg_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(face_img, M_trans_neg_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.add(face_img, 10))
    imgs.append(cv2.add(face_img, 30))
    imgs.append(cv2.add(face_img, -10))
    imgs.append(cv2.add(face_img, -30)) 
    imgs.append(cv2.add(face_img, 15))
    imgs.append(cv2.add(face_img, 45))
    imgs.append(cv2.add(face_img, -15))
    imgs.append(cv2.add(face_img, -45))
    
    return imgs

img_test = images[0]

augmented_image_test = img_augmentation(img_test)

plt.figure(figsize=(15,10))
for i, img in enumerate(augmented_image_test):
    plt.subplot(4,5,i+1)
    plt.imshow(img, cmap="gray")
plt.show()

augmented_images = []
augmented_names = []
for i, face_img in enumerate(images):
    try :
        augmented_images.extend(img_augmentation(face_img))
        augmented_names.extend([names[i]] * 20)
    except :
        print(i)

#total raw and augmented images
len(augmented_images), len(augmented_names)
images.extend(augmented_images)
names.extend(augmented_names)
len(images), len(names)

unique, counts = np.unique(names, return_counts = True)
for item in zip(unique, counts):
    print(item)

##3. DATA NORMALIZATION
def print_data(label_distr, label_name):
    plt.figure(figsize=(12,6))
    my_circle = plt.Circle( (0,0), 0.7, color='white')
    plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')
    plt.gcf().gca().add_artist(my_circle)
    plt.show()
    
unique = np.unique(names)
label_distr = {i:names.count(i) for i in names}.values()
print_data(label_distr, unique)

n=1000
def randc(labels, l):
    return np.random.choice(np.where(np.array(labels) == l)[0], n, replace=False)

img_mask = np.hstack([randc(names, l) for l in np.unique(names)])
names = [names[m] for m in img_mask]
images = [images[m] for m in img_mask]
label_distr = {i:names.count(i) for i in names}.values()
print_data(label_distr, unique)
len(names)

##4 ONE HOT ENCODING

lab_enc = LabelEncoder()
lab_enc.fit(names)
labels = lab_enc.classes_
name_vec = lab_enc.transform(names)
categorical_name_vec = to_categorical(name_vec)

print("number of class :", len(labels))
print(labels)
print(name_vec)
print(categorical_name_vec)

## 5 SPLIT DATASET
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images, dtype=np.float64),   
    np.array(categorical_name_vec),
    test_size=0.25, 
    random_state=42)
print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train.shape, x_test.shape

## 6 CNN MODEL (Model B)
## Model B is used instead of A and C cuz its training time is the shortest (approx 10mins)
## If wanna change parameters, be sure to change the path of model as well.
def cnn_model(input_shape):  
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size = (3,3),
                    padding="valid",
                    activation="relu",
                    input_shape=input_shape
                     ))
    
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64,
                    (3,3),
                    padding="valid",
                    activation="relu",
                    input_shape=input_shape
                     ))
    
    model.add(Conv2D(64,
                    (3,3),
                    padding="valid",
                    activation="relu"))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128,
                    (3,3),
                    padding="valid",
                    activation="relu"))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(labels)))  
    model.add(Activation("softmax"))
    
    model.summary()  
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])

    return model

## 7 TRAINING MODEL
input_shape= x_train[0].shape
EPOCHS = 15
BATCH_SIZE = 32
model = cnn_model(input_shape)

history = model.fit(x_train, 
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.25  
                    )

def evaluate_model(history):
    names = [['accuracy', 'val_accuracy'], 
             ['loss', 'val_loss']]
    for name in names :
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()
evaluate_model(history)

modelPath="C://Users//User//Desktop//GINA_FYP2_FormalSubmission//FaceRec_Materials//Model_B"
model.save(modelPath)

# 8 PREDICT TEST DATA
y_pred=model.predict(x_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 9 COMPUTE CONFUSION MATRIX
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=labels,normalize=False,
                      title='Confusion matrix')

print(classification_report(y_test.argmax(axis=1), 
                            y_pred.argmax(axis=1), 
                            target_names=labels))

# 10 FACE RECOGNITION WITH VIDE0 FRAME
from keras.models import load_model
def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

# Load CNN Model
model = load_model(modelPath)

cap = cv2.VideoCapture(0)
while cap.isOpened() :
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)
            
            result = model.predict(face_img)
            idx = result.argmax(axis=1)
            confidence = result.max(axis=1)*100

            if confidence > 90:
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)

            else :
                label_text = "N/A"
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
       
        cv2.imshow('Detect Face', frame)
    else :
        break
    #press 'q' key to quit
    if cv2.waitKey(10) == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
