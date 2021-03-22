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
# # FROM jTessBoxEditor
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Gaming\Desktop\TesseractTraining\jTessBoxEditorFX\tesseract-ocr\tesseract.exe"
# TESSDATA_PREFIX = r"C:\Program Files\VietOCR\tessdata"

def adjust_gamma(crop_img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(crop_img, table)

# This is the "start" function that will process all images in the dataset
def process_all_images():
    directory = "./fm_dataset"

    # ## FOR TESTING PURPOSES: one image at a time
    # global img_name
    # img = cv2.imread("./fm_dataset/frame0.tiff")
    # # Saving frame name for future reference
    # img_name = "frame0"
    # detect_mouse(img)

    # For running: multiple images at a time
    for filename in os.listdir(directory):
        if filename.endswith(".tiff"):
            global img_name
            img_name = filename
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
        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        print("")
    print("Starting the process!")
    print("Step 1: Finding cursor in the frame")
    # Show the final image with the matched area.
    # cv2.imshow('Detected Cursor', img)
    # cv2.waitKey(0)
    cursor_coord_x = pt[0]
    cursor_coord_y = pt[1]
    crop_hover_list(img, cursor_coord_x, cursor_coord_y)

# This function will crop based on where the mouse is located, the item LISTED
def crop_hover_list(img, cursor_coord_x, cursor_coord_y):
    # Remove unneccessary data
    # Y:LENGTH, X:WIDTH
    print("Step 2: Cropping to show the item in the list based on cursor co-ordinates")
    img_hover_list = img[cursor_coord_y-35:cursor_coord_y+60, cursor_coord_x-250:cursor_coord_x+60]
    # img_hover_list = img[cursor_coord_y-60:cursor_coord_y+60, cursor_coord_x-250:cursor_coord_x+100]
    grayimage = cv2.cvtColor(img_hover_list, cv2.COLOR_BGR2GRAY)

    # CASE 1: If the item is sold out, the background is very white. Must threshold stronger.
    n_white_pix = np.sum(grayimage == 255)

    if(n_white_pix >= 200):
        # CASE 1: If the item is sold out, the background is very white. Must threshold stronger.
        print("- Note: item appears to be sold out - white pixels: ", n_white_pix)
        adjusted = adjust_gamma(grayimage, gamma=0.1)
        canny = cv2.Canny(adjusted, 0, 500)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(canny, kernel)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img_hover_list, contours, -1, (0,255,0), 2)

    else:
        # CASE 2: If the item is not sold out, the background is normal. Normal threshold.
        print("- Note: item appears to be in stock - white pixels: ", n_white_pix)
        canny = cv2.Canny(grayimage, 0, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(canny, kernel)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img_hover_list, contours, -1, (0,255,0), 2)

    print("Step 3: Filtering the loosely cropped list")
    # cv2.imshow("Cropped List", img_hover_list)
    # cv2.imshow("Cropped List", grayimage)
    # cv2.imshow("Hover List: Canny", canny)
    # cv2.imshow("Hover List: Contours", img_hover_list)
    # cv2.imshow("Hover List: Dilated", dilated)
    # cv2.waitKey(0)

    for contour in contours:
        # Contour area determines what it will contour/crop
        if cv2.contourArea(contour) < 1300 or cv2.contourArea(contour) > 10000:
              continue
        # print(cv2.contourArea(contour))
        # # Show bounding box
        (x,y,w,h) = cv2.boundingRect(contour)
        # cv2.rectangle(img_hover_list, (x, y), (x + w, y + h), (0, 0, 255), 3)
        print("Step 4: Contour area, will determine what to crop further")
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
        print("Step 5: Saving the crop to as 'hover_list'")
        # cv2.imshow('Hover List: Crop', img_hover_list)
        # cv2.waitKey(0)
        cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        cv2.imwrite('hover_list.tiff', cropped_image)

    crop_hover_popup(img, cursor_coord_x, cursor_coord_y)

# This function will crop the corresponding HOVER POP-UP of the item based on cursor coordinate
def crop_hover_popup(img, cursor_coord_x, cursor_coord_y):
    # Remove unneccessary data
    # Y:LENGTH, X:WIDTH
    print("Step 6: Cropping the hover pop-up based on cursor coordinate")
    # roughly cropping hover box
    img3 = img[cursor_coord_y-10:cursor_coord_y+450, cursor_coord_x-10:cursor_coord_x+450]
    # adjusted gamma
    adjusted = adjust_gamma(img3, gamma=0.4)
    grayimage = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    n_black_pix = np.sum(grayimage == 0)
    # CASE 1: If the hover list is small, crop a little further
    if(n_black_pix < 3000):
        grayimage = grayimage[0:220, 0:450]
        blur = cv2.GaussianBlur(grayimage, (5, 5), 0)
        thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=5)
        thresh = cv2.dilate(thresh, None, iterations=5)
    else:
        blur = cv2.GaussianBlur(grayimage, (5, 5), 0)
        thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(grayimage, contours, -1, (0,255,0), 2)
    print("Step 7: Showing the binary thresh, eroded and dilated")
    # cv2.imshow("Cropped hover popup", adjusted)
    # cv2.imshow("Gray image", grayimage)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Hover Pop-up: Thresh", thresh)
    # cv2.waitKey(0)

    for contour in contours:
        if cv2.contourArea(contour) < 50000:
             continue
        # print(cv2.contourArea(contour))
        # Show bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(grayimage, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Hover Pop-up: Contours", grayimage)
        # cv2.waitKey(0)
        # Grab coordinates for the box to crop
        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        # save as hover_popup_name
        # Grabbing points to crop the image
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        roi_corners = np.array([box], dtype=np.int32)
        cv2.polylines(img3, roi_corners, 1, (255, 0, 0), 3)
        # cv2.imshow('Hover Pop-up: Name', grayimage)
        # cv2.waitKey(0)
        cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        print("Step 8: Saving to hover_popup_box")
        cv2.imwrite('hover_popup_box.tiff', cropped_image)

    process_hover_list()

# This function will process the text from the cropped screenshot of the item LIST
def process_hover_list():
    print("Step 9: Processing the Hover List (name and price)")
    image = cv2.imread("hover_list.tiff")
    image = image[0:1000, 0:200]
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # adjusted gamma
    adjusted = adjust_gamma(image, gamma=0.4)
    # grayscale the image to get rid of some color issues
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    # denoising image
    dst = cv2.fastNlMeansDenoising(gray, None, 15, 15, 15)
    # process to make it easier to read text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 31)

    # OCR configurations (3 is default)
    config = "--psm 6"

    # Just show the image
    # cv2.imshow("Hover List: Original", image)
    # cv2.imshow("Hover List: Gamma", adjusted)
    # cv2.imshow("Hover List: Denoised", dst)
    # cv2.imshow("Hover List: Thresh", thresh)
    # cv2.waitKey(0)

    text = pytesseract.image_to_string(thresh, lang='maple', config=config)
    # Text before filtering it
    print(text)
    # remove double lines
    # text = text.replace('\n\n', '\n')

    # removing empty lines
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    # replace anything that is not properly formatted
    string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%]', '', string_without_empty_lines)
    # grabs first string and regex to split capital letters
    item_list_name = (re.sub(r"(\w)([A-Z])", r"\1 \2", string_without_specialchars_name.partition('\n')[0]))

    if(string_without_empty_lines.count('\n') >= 2 and string_without_empty_lines.find("mesos") != -1):
        # grabs second string and regex to remove extra letters
        item_list_price = string_without_empty_lines.splitlines()[1]
        item_list_price = item_list_price.replace('.', ',')
        item_list_price = item_list_price[:item_list_price.index("mesos") + 5]
    else:
        item_list_price = "N/A"
    print("Step 10: Saving the following as the item_list_name and item_list_price:")
    print(item_list_name)
    print(item_list_price)
    process_hover_popup(item_list_name, item_list_price)

# This function will process the text from the cropped screenshot of the item HOVER POPUP
def process_hover_popup(item_list_name, item_list_price):
    print("Step 11: Processing the hover popup image (splitting into name vs stats)")
    # Grabs box of hover popup and resizes
    image_hover = cv2.imread("hover_popup_box.tiff")
    img_height = np.size(image_hover, 0)
    img_width = np.size(image_hover, 1)
    # print(img_height, img_width)
    img_gray = cv2.cvtColor(image_hover, cv2.COLOR_BGR2GRAY)
    # grabs coordinates for NAME part
    image_crop = img_gray[5: 40, 20: 320]
    image_name = cv2.resize(image_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # pre-processing NAME image
    adjusted_name = adjust_gamma(image_name, gamma=3.3)
    # denoising NAME image
    dst_name = cv2.fastNlMeansDenoising(adjusted_name, None, 2, 2, 2)
    # process to make it easier to read text
    ret, thresh_name = cv2.threshold(dst_name,136,255,cv2.THRESH_BINARY)

    # canny = cv2.Canny(image_hover, 0, 150)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # dilated = cv2.dilate(canny, kernel)
    # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(dilated, contours, -1, (0,255,0), 2)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(thresh, contours, -1, (0,255,0), 2)

    # cv2.imshow("Hover: Thresh", thresh)
    # cv2.imshow("Hover: Canny", canny)
    # cv2.imshow("Hover: Dilated", dilated)
    cv2.waitKey(0)

    for contour in contours:
        if cv2.contourArea(contour) < 20000 or cv2.contourArea(contour) > 50000:
            global no_stats
            no_stats = "true"
            continue
        # Contour area determines what it will contour/crop
        print(cv2.contourArea(contour))
        # # Show bounding box
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(image_hover, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Hover List: Crop", image_hover)
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
        cv2.polylines(image_hover, roi_corners, 1, (255, 0, 0), 3)
        # cv2.imshow('Hover List: Crop', image_hover)
        # cv2.waitKey(0)
        cropped_image_stats = img_gray[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        cv2.imwrite('hover_popup_stats.tiff', cropped_image_stats)

    # Non-equips/weapons do not need stats - remove entirely.
    if(no_stats == "true"):
        # Grabs box of hover popup and resizes
        stats_hover = cv2.imread("hover_popup_stats.tiff")
        stats_gray = cv2.cvtColor(stats_hover, cv2.COLOR_BGR2GRAY)
        image_stats = cv2.resize(stats_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # pre-processing STATS image
        adjusted_stats = adjust_gamma(image_stats, gamma=3.3)
        # denoising NAME image
        dst_stats = cv2.fastNlMeansDenoising(adjusted_stats, None, 2, 2, 2)
        # process to make it easier to read text
        ret, thresh_stats = cv2.threshold(dst_stats,136,255,cv2.THRESH_BINARY)

        # OCR configurations (3 is default)
        config = "--psm 6"

        # # Just show the name image
        # cv2.imshow("name", image_name)
        # cv2.imshow("dst_name", dst_name)
        # cv2.imshow("thresh_name", thresh_name)
        # cv2.waitKey(0)
        # # Just show the stats image
        # cv2.imshow("stats", image_stats)
        # cv2.imshow("dst_stats", dst_stats)
        # cv2.imshow("thresh_stats", thresh_stats)
        # cv2.waitKey(0)

        text_name = pytesseract.image_to_string(thresh_name, lang='maple', config=config)
        text_stats = pytesseract.image_to_string(thresh_stats, lang='maple', config=config)
        # Text_name and text_stats before processing
        print("Detected text")
        print(text_name)
        print(text_stats)

        # processing NAME text
        # removing empty lines
        lines_name = text_name.split("\n")
        non_empty_lines_name = [line for line in lines_name if line.strip() != ""]
        string_without_empty_lines_name = ""
        for line in non_empty_lines_name:
            string_without_empty_lines_name += line + "\n"
        # replace anything that is not properly formatted
        string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%]', '', string_without_empty_lines_name)
        # grabs first string and regex to split capital letters
        item_hover_name = string_without_specialchars_name.partition('\n')[0].strip()

        # processing STATS text
        # removing empty lines
        lines_stats = text_stats.split("\n")
        non_empty_lines_stats = [line for line in lines_stats if line.strip() != ""]
        string_without_empty_lines_stats = ""
        for line in non_empty_lines_stats:
            string_without_empty_lines_stats += line + "\n"
        # replace anything that is not properly formatted
        string_without_specialchars_stats = re.sub('[^a-zA-Z0-9\s(\)+%:]', '', string_without_empty_lines_stats)
        # grabs first string and regex to split capital letters
        item_hover_stats = string_without_specialchars_stats.replace('\n', ', ')
        # print(item_hover_stats)
        save_item(item_list_name, item_list_price, item_hover_name, item_hover_stats)

    else:
        # OCR configurations (3 is default)
        config = "--psm 6"

        # # Just show the name image
        # cv2.imshow("name", image_name)
        # cv2.imshow("dst_name", dst_name)
        # cv2.imshow("thresh_name", thresh_name)
        # cv2.waitKey(0)

        text_name = pytesseract.image_to_string(thresh_name, lang='maple', config=config)
        text_stats = "NO STATS"
        # Text_name and text_stats before processing
        print("Detected text")
        print(text_name)
        print(text_stats)

        # processing NAME text
        # removing empty lines
        lines_name = text_name.split("\n")
        non_empty_lines_name = [line for line in lines_name if line.strip() != ""]
        string_without_empty_lines_name = ""
        for line in non_empty_lines_name:
            string_without_empty_lines_name += line + "\n"
        # replace anything that is not properly formatted
        string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%]', '', string_without_empty_lines_name)
        # grabs first string and regex to split capital letters
        item_hover_name = string_without_specialchars_name.partition('\n')[0].strip()

        item_hover_stats = text_stats
        # print(item_hover_stats)
        save_item(item_list_name, item_list_price, item_hover_name, item_hover_stats)

# This function will save the item NAME, PRICE, AND STATS into a txt file AND csv
def save_item(item_list_name, item_list_price, item_hover_name, item_hover_stats):
    global img_name
    print("FINAL: Saving item list name, item list price, item hover name, and item hover stats")
    if item_list_name == item_hover_name:
        # Storing text into txt file
        with open('rawtext.txt', 'a') as f:
            print(item_hover_name + "|" + item_list_price + "|" + item_hover_stats + "|" + img_name, file=f)

        # combining into csv
        dataset = pd.read_csv('rawtext.txt', names=['Item Name', 'Price Read', 'Item Stats', 'Frame Name'], delimiter='|')
        # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
        print(dataset)
        dataset.to_csv(r'items.csv')

    else:
        print("NOTE: Names do not correspond")
        print(item_list_name + " " + item_hover_name)
        # Storing text into txt file
        with open('rawtext.txt', 'a') as f:
            print(item_hover_name + "|" + item_list_price + "|" + item_hover_stats + "|" + img_name, file=f)

        # combining into csv
        dataset = pd.read_csv('rawtext.txt', names=['Item Name', 'Price Read', 'Item Stats', 'Frame Name'], delimiter='|')
        # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
        print(dataset)
        dataset.to_csv(r'items.csv')

process_all_images()