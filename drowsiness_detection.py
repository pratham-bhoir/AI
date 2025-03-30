# Import packages
from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import pygame
image = cv2.imread('Images/drowsiness.png')  # Load input (coffee break) image
image = cv2.resize(image, (100, 100))          # Resize image to 100x100

pygame.mixer.init()
pygame.mixer.music.load('siren.wav') #music loading


#Eye Aspect Ratio (E.A.R.)
#Function to calculate eye aspect ratio as in paper :

#Landmarks |   0  1  2  3  4  5
 #Left Eye : [36,37,38,39,40,41]
#Right Eye : [42,43,44,45,46,47]

def eye_aspect_ratio(eye):
    # Vertical distances
    dist1 = dist.euclidean(eye[1], eye[5])  # P2-P6
    dist2 = dist.euclidean(eye[2], eye[4])  # P3-P5
    # Horiontal distance
    dist3 = dist.euclidean(eye[0], eye[3])  # P1-P4

    # Eye Aspect Ratio (E.A.R.)
    ear = (dist1 + dist2) / (2.0 * dist3)

    return ear


# Lips Aspect Ratio (L.A.R.)
# Function to calculate lips aspect ratio in the same way as in E.A.R.
# Landmarks |   0  1  2  3  4  5  6  7
#      Lips : [60,61,62,63,64,65,66,67]

def lips_aspect_ratio(lips):
    # Vertical distance
    dist1 = dist.euclidean(lips[2], lips[6])  # L3-L7
    # Horiontal distance
    dist2 = dist.euclidean(lips[0], lips[4])  # L1-L5

    # Lips Aspect Ratio (L.A.R.)
    lar = float(dist1/dist2)

    return lar


# Facial Landmarks for any face part
# Function to calculate facial landmark point coordinates (x,y),
# draw them on frame and return a numpy array with the corresponding points



def draw_landmarks(face_part, landmarks):
    landmarks_list = []
    for point in face_part:
        x, y = landmarks.part(point).x, landmarks.part(point).y
        landmarks_list.append([x, y])
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return np.array(landmarks_list)