import os
import pandas as pd
import numpy as np

train_list = '/home/fas1/SiWTRAIN/protocol/Train_I.txt'
train_image_dir = '/home/fas1/SiWTRAIN/dataset/train'

landmarks_frame = pd.read_csv(train_list, delimiter=',', header=None)
videoname = str(landmarks_frame.iloc[100, 0])
videoname = '147-2-3-2-2'
image_path = os.path.join(train_image_dir, videoname)

for name in os.listdir(image_path):
    img = os.path.join(image_path,name)
    print(img)
    if os.path.exists(img):
        print('find image \n')
    else:
        print('fuck')

frames_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
print(frames_total)
for a in range(frames_total):
    print(videoname)
    if a == (frames_total - 1):
        print('wow fuck you\n')
    image_id = np.random.randint(1, frames_total - 1)
    s = "-%04d" % image_id
    print(type(image_id))
    if type(image_id) is int:
        print('type is int')
    else:
        print('wrong')
    # s = "_%03d_scene" % image_id

    # s = "_%03d_scene" % 1
    image_name = videoname + s + '.jpg'
    image_path = os.path.join(image_path, image_name)

    if os.path.exists(image_path):
        print('find image \n')
        break
    print('haven`t find image\n')
print('out of for loop \n')

path = '/home/fas1/SiWTRAIN/dataset/train/099-2-3-3-1/099-2-3-3-1-0002.jpg/'
print('////////////////////////////////////////////TEST\n')
print(os.path.exists(path))
