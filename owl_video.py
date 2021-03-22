# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:42:05 2021
A program that will create the folder dataset with each frame of the video
@author: Joanna Makary
"""

import cv2
import os
import pytesseract
import pandas as pd
import numpy as np
from itertools import groupby

# Tells pytesseract where the tesseract environment is installed on local computer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize+1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # converts the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def create_frames():
    hashes = []
    # create the frames in folder "dataset" from the video
    # create directory for frames
    if not os.path.exists('owl_dataset'):
        os.makedirs('owl_dataset')

    # video path
    video_file = cv2.VideoCapture('owl_video.mp4')

    # start index or count for the frames
    index = 0
    while video_file.isOpened():
        ret,frame = video_file.read()
        if not ret:
            break

        # assign name for files
        name = './owl_dataset/frame' + str(index) + '.tiff'
        # print('Extracting frames...' + name)

        # if there is a cursor in the frame,
        testing_frame = frame[350:750, 675:1250]
        img_gray = cv2.cvtColor(testing_frame, cv2.COLOR_BGR2GRAY)
        # Read the template
        template = cv2.imread('template.png', 0)
        # Store width and height of template in w and h
        w, h = template.shape[::-1]
        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        # Specify a threshold
        threshold = 0.6
        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)
        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(testing_frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
            print("")
        try:
            cursor_coord_x = pt[0]
            cursor_coord_y = pt[1]
            frame = frame[350:1200, 675:1500]
        except:
            frame = frame[350:750, 675:1250]

        # save the hash of the CROPPED test image in the array
        current_frame = dhash(testing_frame)

        # if the hash exists in the hashes array, do not save file
        # otherwise, save to hashes and save file and increase index
        if current_frame not in hashes:
            hashes.append(current_frame)
            cv2.imwrite(name, frame)
            index = index + 1
        else:
            print("This frame already exists, skipping the save")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video_file.release()
    cv2.destroyAllWindows()

create_frames()
