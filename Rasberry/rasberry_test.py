import cv2
import numpy as np
import os
import torch
from models.CDCNs import CDCNpp
from torchvision import transforms


#os.chdir("C:\Python39\Lib\site-packages\cv2\data")
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read('/home/pi/FaceRecognition/trainer/trainer.yml')
cascadePath = "C:\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

'''
這邊是訓練完的資料
epoch:30, Val:  val_threshold= 0.5164, val_ACC= 0.9950, val_ACER= 0.0063 
epoch:30, Test:  ACC= 0.9987, APCER= 0.0000, BPCER= 0.0033, ACER= 0.0017 
'''

model = CDCNpp()
model.to('cpu')
model.load_state_dict(torch.load('CDCNpp_train_total.pkl',map_location='cpu'))
model.eval()

with torch.no_grad():
    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite('crop_img.jpg', crop_img)
            input = np.zeros((1,256,256,3))
            input[0,:,:,:] = cv2.resize(crop_img, (256,256))
            input = transforms.Compose(input)

            input = transform(input)

            map_score = 0.0

            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(input[0,:,:,:])

            map_score = torch.sum(map_x) / 32*32

            if map_score < 0.5164:
                spoofing_label = 'fake'
            else:
                spoofing_label = 'live'

            # puttext把資訊實時更新
            cv2.putText(img, spoofing_label, (x+5,y-5), font, 1, (255,255,255), 2)

        cv2.imshow('camera',img) # 把圖顯示出來

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
