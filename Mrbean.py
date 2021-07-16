import math
import cv2
import random

import numpy as np
from mtcnn.mtcnn import MTCNN

image=cv2.imread('mr_bean.jpeg')

def my_noise(img):
    rows, col ,chn= img.shape

    number_of_pixels_white = 0.1 * rows*col//2
    for i in range(int(number_of_pixels_white)):
        y_coord = random.randint(0, rows - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels_black = 0.1 * rows*col//2
    for i in range(int(number_of_pixels_black)):
        y_coord = random.randint(0, rows - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def my_rotate(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(image)
    detection = result[0]
    keypoints = detection['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    if left_eye_y > right_eye_y:
        point3 = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point3 = (left_eye_x, right_eye_y)
        direction = 1
    a = math.sqrt(math.pow(right_eye_x - point3[0], 2) + math.pow(right_eye_y - point3[1], 2))
    b = math.sqrt(math.pow(left_eye_x - right_eye_x, 2) + math.pow(left_eye_y - right_eye_y, 2))
    cos = a / b
    angle = np.arccos(cos)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle

    result = rotate_image(image, int(direction * angle))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


noise_img=my_noise(image)
clean_noise=cv2.medianBlur(noise_img,3)
rotate_img=my_rotate(clean_noise)
cv2.imshow('show output',rotate_img)
cv2.waitKey()