import cv2
import glob
import random
import numpy as np
import time
#from pygame import mixer
emotions = ["anger","happy", "sadness", "surprise"] #Emotion list
color = [(0,0,255) , (0,255,0) , (255,0,) , (0,255,255)]
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifierl
data = {}
filters = ['Filters/ang.png','Filters/happy_filter.png','Filters/sad_filter.png','Filters/surprise_filter.png']
quotes = ['There is only one happiness in this life, to love and be loved. ' ,'The word happy would lose its meaning if it were not balanced by sadness.','For every minute you remain angry you give up sixty seconds of peace of mind.','Expect the best, plan for the worst, and prepare to be surprised.']
# music = ['Sounds/anger.mp3','Sounds/happy.mp3','Sounds/sad.mp3','Sounds/surprised.mp3']

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset2/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] 
    prediction = files[-int(len(files)*0.2):]
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

def tell(img,file,x,y,w,h):
    pred, conf = fishface.predict(file)
    ro = str(emotions[pred] + ' ' + str(round( (100.000 - conf/ 100),2 )) + '%')
    print(ro)
    # print(str(round(100.000 - conf/ 100,2 ) ) )
    cv2.putText(img,ro,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color[pred] ,1)
    cv2.putText(img,quotes[pred],(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color[pred] ,1)
    cv2.rectangle(img,(x,y),(x+w,y+h),color[pred],2)
    print("The sentiment analysis shows that you are showing a " + emotions[pred] +" emotion")
    return pred

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y-20:y+h, x-20:x+w]
        try:
            out = cv2.resize(roi_gray, (350, 350)) 
            pred = tell(img,out,x,y,w,h)    
            img2 = cv2.imread(filters[pred])
            img2 = cv2.resize(img2, (w,h))
            rows,cols,channels = img2.shape
            roi = img[y:y+rows ,x:x+cols]
            img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
            dst = cv2.add(img1_bg,img2_fg)
            img[ y:y+rows, x:x+cols] = dst
            # mixer.music.load(music[pred])
            # mixer.music.play()
            # time.sleep(2)
        except:
           pass #If error, pass file
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

