import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from os import walk
from os.path import join,splitext

characterRecognition = tf.keras.models.load_model('/home/ff5v/git-file/car_template_recognition/CNN/weights/char_binary_model.h5')

def opencvReadPlate(img):
    charList=[]
	
    # set black thresh
    lower_black=np.array([0,0,0])
    upper_black=np.array([180,255,46])

    lower_white = np.array([0, 0, 65])
    upper_white = np.array([180, 80, 255])

    # change to hsv model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get mask
    mask = cv2.inRange(hsv, lower_black, upper_black)

    #draw rect
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#只检测外轮廓
	
    #cv2.boundingRect回傳值: x、y是矩阵左上点的坐标，w、h是矩阵的宽和高
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0]) #左到右排序
    img_area = img.shape[0]*img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        non_max_sup = roi_area/img_area #框中面積/原圖面
        
        if((non_max_sup >= 0.011) and (non_max_sup < 0.1)):
            if ((h>1.2*w) and (10.6*w>=h) and (x!=0) and (y!=0)):
                char = mask[y:y+h,x:x+w]
                char = char.reshape(char.shape[0],char.shape[1],1)
                #print(char.shape)
                char=np.concatenate([char,char,char],2)
                #print(char.shape)
                cv2.imshow('char', char)
                cv2.waitKey(0)
                charList.append(cnnCharRecognition(char))
                cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
        
    licensePlate="".join(charList)
    return licensePlate

def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K',
    20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T', 28:'U',
    29:'V', 30:'W', 31:'X', 32:'Y', 33:'Z'}

    test_img = cv2.resize(img,(224,224))
    test_img = np.asarray(test_img.astype('float32'))
    test_img = test_img/255.
    test_img = test_img.reshape((1,224,224,3))

    new_predictions = characterRecognition.predict(test_img)
    char = np.argmax(new_predictions)
    return dictionary[char]
	

img = cv2.imread('/home/ff5v/train_1/default/1.jpg')
print(type(img))
print(img.shape)
licensePlate = opencvReadPlate(img)
print("OpenCV+CNN : " + licensePlate)
cv2.putText(img, licensePlate, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow('result', img)
cv2.waitKey(0)