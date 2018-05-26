import cv2
import glob
import random
import numpy as np
emotions = ["anger","happy", "sadness", "surprise"] #Emotion list
color = [(0,0,255) , (0,255,0) , (255,0,) , (0,255,255)]
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset2/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_video():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
   
run_video()  

def tell(file,x,y,w,h):
    pred, conf = fishface.predict(file)
    cv2.putText(img,emotions[pred],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color[pred] ,1)
    cv2.rectangle(img,(x,y),(x+w,y+h),color[pred],2)
    print("The sentiment analysis shows that you are showing a " + emotions[pred] +" emotion")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.fastNlMeansDenoising(gray,None,3,7,21)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y-20:y+h, x-20:x+w]
        try:
            out = cv2.resize(roi_gray, (350, 350)) 
            tell(out,x,y,w,h)    
        except:
           pass #If error, pass file
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()


