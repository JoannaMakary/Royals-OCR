# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:42:05 2021
A program that will process the text of the extracted frames
@author: Joanna Makary
"""

import cv2
import os
import numpy as np
import pytesseract
import re
import pandas as pd

# Tells pytesseract where the tesseract environment is installed on local computer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# This is the "start" function that will process all images in the dataset
def process_all_images():
    directory = "./dataset"
    for filename in os.listdir(directory):
        if filename.endswith(".tiff"):
            filepath = os.path.join(directory, filename).replace("\\", "/")
            print(filepath)
            img = cv2.imread(filepath)
            detect_mouse(img)

# This function will detect where the cursor position is in the screenshot and return the x and y coordinates
def detect_mouse(img):
    # Convert it to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Read the template
    template = cv2.imread('template.png', 0)
    # Store width and height of template in w and h
    w, h = template.shape[::-1]
    # Perform match operations.
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # Specify a threshold
    threshold = 0.8
    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)
    # Draw a rectangle around the matched region.
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        print("")
    # Show the final image with the matched area.
    #cv2.imshow('Detected Cursor', img)
    #cv2.waitKey(0)
    cursor_coord_x = pt[0]
    cursor_coord_y = pt[1]
    crop_hover_list(img, cursor_coord_x, cursor_coord_y)

# This function will crop based on where the mouse is located, the item LISTED
def crop_hover_list(img, cursor_coord_x, cursor_coord_y):
    # Remove unneccessary data
    # Y:LENGTH, X:WIDTH
    img_hover_list = img[cursor_coord_y-50:cursor_coord_y+50, cursor_coord_x-250:cursor_coord_x+60]
    grayimage = cv2.cvtColor(img_hover_list, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grayimage, 0, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilated = cv2.dilate(canny, kernel)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img_hover_list, contours, -1, (0,255,0), 2)
    # cv2.imshow("Hover List: Canny", canny)
    # cv2.imshow("Hover List: Contours", img_hover_list)
    # cv2.imshow("Hover List: Dilated", dilated)
    # cv2.waitKey(0)

    for contour in contours:
        # Contour area determines what it will contour/crop
        if cv2.contourArea(contour) < 3000 or cv2.contourArea(contour) > 10000:
             continue
        print(cv2.contourArea(contour))
        # # Show bounding box
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(img_hover_list, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Hover List: Crop", img_hover_list)
        # cv2.waitKey(0)
        # Grab coordinates for the box to crop
        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        # Grabbing points to crop the image
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        roi_corners = np.array([box], dtype=np.int32)
        cv2.polylines(img_hover_list, roi_corners, 1, (255, 0, 0), 3)
        # cv2.imshow('Hover List: Crop', img_hover_list)
        # cv2.waitKey(0)
        cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        cv2.imwrite('hover_list.tiff', cropped_image)

    crop_hover_popup(img, cursor_coord_x, cursor_coord_y)

# This function will crop the corresponding HOVER POP-UP of the item based on cursor coordinate
def crop_hover_popup(img, cursor_coord_x, cursor_coord_y):
    # Remove unneccessary data
    # Y:LENGTH, X:WIDTH
    img3 = img[cursor_coord_y-5:cursor_coord_y+450, cursor_coord_x-5:cursor_coord_x+450]
    grayimage = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grayimage, 400, 500)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated = cv2.dilate(canny, kernel)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img3, contours, -1, (0,255,0), 2)
    # cv2.imshow("Hover Pop-up: Canny", canny)
    # cv2.imshow("Hover Pop-up: Dilated", dilated)
    cv2.waitKey(0)

    for contour in contours:
        # Contour area determines what it will contour/crop
        if cv2.contourArea(contour) < 10000 or cv2.contourArea(contour) > 90000:
            continue
        # Show bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Hover Pop-up: Contours", img3)
        # cv2.waitKey(0)
        # Grab coordinates for the box to crop
        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        if cv2.contourArea(contour) < 40000:
            # save as hover_popup_stats
            # Grabbing points to crop the image
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            roi_corners = np.array([box], dtype=np.int32)
            cv2.polylines(img3, roi_corners, 1, (255, 0, 0), 3)
            # cv2.imshow('Hover Pop-up: Stats', img3)
            # cv2.waitKey(0)
            cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
            cv2.imwrite('hover_popup_stats.tiff', cropped_image)
        else:
            # save as hover_popup_name
            # Grabbing points to crop the image
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            roi_corners = np.array([box], dtype=np.int32)
            cv2.polylines(img3, roi_corners, 1, (255, 0, 0), 3)
            # cv2.imshow('Hover Pop-up: Name', img3)
            # cv2.waitKey(0)
            cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
            cv2.imwrite('hover_popup_name.tiff', cropped_image)
    process_hover_list()

# This function will process the text from the cropped screenshot of the item LIST
def process_hover_list():
    image = cv2.imread("hover_list.tiff")
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    def adjust_gamma(crop_img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(crop_img, table)

    # adjusted gamma
    adjusted = adjust_gamma(image, gamma=0.4)
    # grayscale the image to get rid of some color issues
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    # denoising image
    dst = cv2.fastNlMeansDenoising(gray, None, 15, 15, 15)
    # process to make it easier to read text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 31)

    # OCR configurations (3 is default)
    config = "--psm 6"

    # # Just show the image
    # cv2.imshow("Hover List: Original", image)
    # cv2.imshow("Hover List: Gamma", adjusted)
    # cv2.imshow("Hover List: Denoised", dst)
    # cv2.imshow("Hover List: Thresh", thresh)
    # cv2.waitKey(0)

    text = pytesseract.image_to_string(thresh, lang='eng', config=config)
    # remove double lines
    # text = text.replace('\n\n', '\n')
    # removing empty lines
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    # replace anything that is not properly formatted
    string_without_empty_lines = string_without_empty_lines.replace('[^A-Za-z|\d| |.|,|-]', '')
    # replace . with ,
    string_without_empty_lines = string_without_empty_lines.replace('.', ',')

    # grabs first string and regex to split capital letters
    item_list_name = (re.sub(r"(\w)([A-Z])", r"\1 \2", string_without_empty_lines.partition('\n')[0]))
    # grabs second string and regex to remove extra letters
    item_list_price = string_without_empty_lines.splitlines()[1]
    item_list_price =  item_list_price[:item_list_price.index("mesos")+5]
    #print(item_list_name)
    #print(item_list_price)
    process_hover_popup(item_list_name, item_list_price)

# This function will process the text from the cropped screenshot of the item HOVER POPUP
def process_hover_popup(item_list_name, item_list_price):
    # Grabs both hover popup name and stats and resizes to be large
    image_name = cv2.imread("hover_popup_name.tiff")
    image_name = image_name[5: 30, 13: 1000]
    image_name = cv2.resize(image_name, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    image_stats = cv2.imread("hover_popup_stats.tiff")
    image_stats = image_stats[0: 1000, 8: 1000]
    image_stats = cv2.resize(image_stats, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    def adjust_gamma(crop_img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(crop_img, table)

    # pre-processing image NAME
    adjusted_name = adjust_gamma(image_name, gamma=0.5)
    # grayscale the image to get rid of some color issues
    gray_name = cv2.cvtColor(adjusted_name, cv2.COLOR_BGR2GRAY)
    # denoising image
    dst_name = cv2.fastNlMeansDenoising(gray_name, None, 10, 10, 10)
    # process to make it easier to read text
    thresh_name = cv2.adaptiveThreshold(dst_name, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 4)

    # pre-processing image STATS
    adjusted_stats = adjust_gamma(image_stats, gamma=0.9)
    # grayscale the image to get rid of some color issues
    gray_stats = cv2.cvtColor(adjusted_stats, cv2.COLOR_BGR2GRAY)
    # denoising image
    dst_stats = cv2.fastNlMeansDenoising(gray_stats, None, 10, 10, 10)

    # process to make it easier to read text
    # ret, thresh_stats = cv2.threshold(dst_stats,70, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(thresh_stats, kernel, iterations=1)
    thresh_stats = cv2.threshold(dst_stats, 90, 255, cv2.THRESH_BINARY_INV)[1]


    # OCR configurations (3 is default)
    config = "--psm 6"

    # # Just show the image
    # cv2.imshow("name", image_name)
    # cv2.imshow("dst_stats", dst_stats)
    # cv2.imshow("thresh_stats", thresh_stats)
    # cv2.waitKey(0)

    text_name = pytesseract.image_to_string(thresh_name, lang='eng', config=config)
    text_stats = pytesseract.image_to_string(thresh_stats, lang='eng', config=config)

    # processing NAME text
    # remove double lines
    # text = text.replace('\n\n', '\n')
    # removing empty lines
    lines_name = text_name.split("\n")
    non_empty_lines_name = [line for line in lines_name if line.strip() != ""]
    string_without_empty_lines_name = ""
    for line in non_empty_lines_name:
        string_without_empty_lines_name += line + "\n"
    # replace anything that is not properly formatted
    string_without_empty_lines_name = string_without_empty_lines_name.replace('[^A-Za-z|\d| |.|,|-]', '')
    # replace . with ,
    string_without_empty_lines_name = string_without_empty_lines_name.replace('.', ',')
    # grabs first string and regex to split capital letters
    item_hover_name = string_without_empty_lines_name.partition('\n')[0]
    item_hover_name = re.sub("[^a-zA-Z]+", "", item_hover_name)
    item_hover_name = (re.sub(r"(\w)([A-Z])", r"\1 \2", item_hover_name))
    #print(item_hover_name)

    # processing STATS text
    # remove double lines
    # text = text.replace('\n\n', '\n')
    # removing empty lines
    lines_stats = text_stats.split("\n")
    non_empty_lines_stats = [line for line in lines_stats if line.strip() != ""]
    string_without_empty_lines_stats = ""
    for line in non_empty_lines_stats:
        string_without_empty_lines_stats += line + "\n"
    # replace anything that is not properly formatted
    string_without_empty_lines_stats = string_without_empty_lines_stats.replace('[^A-Za-z|\d| |.|,|-]', '')
    # replace ? with 7
    string_without_empty_lines_stats = string_without_empty_lines_stats.replace('?', '7')
    # replace i with 1
    string_without_empty_lines_stats = string_without_empty_lines_stats.replace('i', '1')
    # grabs first string and regex to split capital letters
    item_hover_stats = string_without_empty_lines_stats.replace('\n', ', ')
    #print(item_hover_stats)
    save_item(item_list_name, item_list_price, item_hover_name, item_hover_stats)

# This function will save the item NAME, PRICE, AND STATS into a txt file AND csv
def save_item(item_list_name, item_list_price, item_hover_name, item_hover_stats):
    if item_list_name == item_hover_name:
        # Storing text into txt file
        with open('rawtext.txt', 'a') as f:
            print(item_hover_name + "|" + item_list_price + "|" + item_hover_stats, file=f)

        # combining into csv
        dataset = pd.read_csv('rawtext.txt', names=['Item Name', 'Price Read', 'Item Stats'], delimiter='|')
        # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
        print(dataset)
        dataset.to_csv(r'items.csv')

    else:
        print("NOTE: Names do not correspond")
        print(item_list_name + " " + item_hover_name)
        # Storing text into txt file
        with open('rawtext.txt', 'a') as f:
            print(item_hover_name + "|" + item_list_price + "|" + item_hover_stats, file=f)

        # combining into csv
        dataset = pd.read_csv('rawtext.txt', names=['Item Name', 'Price Read', 'Item Stats'], delimiter='|')
        # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
        print(dataset)
        dataset.to_csv(r'items.csv')


#detect_mouse()
process_all_images()