import math

import cv2
import random

import numpy as np
from mtcnn.mtcnn import MTCNN

image=cv2.imread('mr_bean.jpeg',cv2.IMREAD_GRAYSCALE)

def my_noise():
    rows, col= image.shape
    number = 1
    for i in range(rows):
        for j in range(col):
            rnd = random.randint(0,2)
            if rnd < number:
                image[i, j] = 0
            elif rnd > number:
                image[i, j] = 255
    return image
def my_rotate(image):
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    detector=MTCNN()
    result=detector.detect_faces(image)
    detection= result[0]
    keypoints=detection['keypoints']
    left_eye=keypoints['left_eye']
    right_eye = keypoints['right_eye']
    left_eye_x ,left_eye_y=left_eye
    right_eye_x , right_eye_y = right_eye
    if left_eye_y > right_eye_y:
        point3=(right_eye_x ,left_eye_y)
        direction=-1
    else:
        point3 = (left_eye_x, right_eye_y)
        direction = 1
    a=math.sqrt(math.pow(right_eye_x-point3[0],2)+math.pow(right_eye_y-point3[1],2))
    b = math.sqrt(math.pow(left_eye_x - right_eye_x, 2) + math.pow(left_eye_y - right_eye_y, 2))
    cos=a/b
    angle=np.arccos(cos)
    angle=(angle*180)/math.pi
    if direction == -1:
        angle=90-angle
    img=image.fromarray(image)
    result=np.array(img.rotate(direction*angle))
    result=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    return result


noise_img=my_noise()
clean_noise=cv2.medianBlur(noise_img,3)
rotate_img=my_rotate(image)
cv2.imshow('out',rotate_img)
cv2.waitKey()