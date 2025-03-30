# Import packages
from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import pygame
image = cv2.imread('Images/drowsiness.png')  # Load input (coffee break) image
image = cv2.resize(image, (100, 100))          # Resize image to 100x100

pygame.mixer.init()
pygame.mixer.music.load('siren.wav')
