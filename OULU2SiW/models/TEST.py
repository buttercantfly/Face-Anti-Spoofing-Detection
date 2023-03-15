import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

info_list = '/1_1_01_1/1_1_01_1_000_scene.jpg'

# for name in os.listdir(info_list):
#     image_path = os.path.join(info_list, name)
#     image = cv2.imread(image_path)
image = cv2.imread(info_list)

plt.subplots(1,2,1)
plt.imshow(image)
plt.show()

image_aug = seq.augment_image(image)

plt.subplots(1,2,1)
plt.imshow(image)

plt.subplots(1,2,2)
plt.imshow(image_aug)
plt.show()
print(image)
print('////////////////////////')
print(image_aug)