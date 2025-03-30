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