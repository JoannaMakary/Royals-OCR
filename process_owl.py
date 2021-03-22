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

global owl_item_name
global prices_array
global hoverable

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
    directory = "./owl_dataset"

    ## FOR TESTING PURPOSES: one image at a time
    global img_name
    img = cv2.imread("./owl_dataset/frame10.tiff")
    # Saving frame name for future reference
    img_name = "frame10"
    process_owl(img)
    if(hoverable == "false"):
        calculate_price()

    # # # For running: multiple images at a time
    # for filename in os.listdir(directory):
    #     if filename.endswith(".tiff"):
    #         global img_name
    #         img_name = filename
    #         filepath = os.path.join(directory, filename).replace("\\", "/")
    #         print(filepath)
    #         img = cv2.imread(filepath)
    #         process_owl(img)

# This function will check whether it is a regular item or needs further processing (hovered)
def process_owl(img):
    global owl_item_name
    global hoverable

    owl_item_name = img[0:28, 0:500]
    image = cv2.resize(owl_item_name, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    owl_item_name = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if it is an equip/needs hovering
    width = img.shape[1]
    if(width >= 825):
        owl_item_prices = img[130:700, 230:1000]
        hovered_item(owl_item_prices, owl_item_name)
        hoverable = "true"
    else:
        owl_item_prices = img[130:376, 230:365]
        regular_item(owl_item_prices, owl_item_name)
        hoverable = "false"

# If it is a regular item, this will process just the average of the item prices.
def regular_item(owl_item_prices, owl_item_name):
    image2 = cv2.resize(owl_item_prices, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    owl_item_prices = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # pre-processing NAME image
    adjusted_name = adjust_gamma(owl_item_name, gamma=0.5)
    # denoising NAME image
    blur_name = cv2.GaussianBlur(adjusted_name,(5,5),0)
    # process to make it easier to read text
    ret, thresh_name = cv2.threshold(blur_name, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # pre-processing PRICES image
    adjusted_prices = adjust_gamma(owl_item_prices, gamma=0.5)
    # denoising NAME image
    blur_prices = cv2.GaussianBlur(adjusted_prices, (5, 5), 0)
    # process to make it easier to read text
    ret, thresh_prices = cv2.threshold(blur_prices, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imshow("adjusted", adjusted_name)
    # cv2.imshow("Original Name", owl_item_name)
    # cv2.imshow("Thresh Name", thresh_name)
    # cv2.imshow("Original Prices", owl_item_prices)
    # cv2.imshow("Thresh Prices", thresh_prices)
    # cv2.waitKey(0)

    # OCR configurations (3 is default)
    config = "--psm 6"
    item_name = pytesseract.image_to_string(thresh_name, lang='eng', config=config)
    item_prices = pytesseract.image_to_string(thresh_prices, lang='eng', config=config)

    # processing NAME text
    # replace anything that is not properly formatted
    string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%]', '', item_name)
    # grabs first string and regex to split capital letters
    owl_item_name = string_without_specialchars_name.partition('\n')[0].strip()
    owl_item_name = re.search(r'Search results for (.*?) that you entered', owl_item_name).group(1)
    # print(owl_item_name)

    # processing PRICES text
    lines_name = item_prices.split("\n")
    non_empty_lines_name = [line for line in lines_name if line.strip() != ""]
    string_without_empty_lines_name = ""
    for line in non_empty_lines_name:
        string_without_empty_lines_name += line + "\n"
    string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%],', '', string_without_empty_lines_name)

    owl_item_prices = string_without_specialchars_name.replace(".", "")
    owl_item_prices = owl_item_prices.replace(",", "")
    # owl_item_prices = owl_item_prices.replace("\n", ",")

    global prices_array

    if ("prices_array" in globals()):
        temp_array = owl_item_prices.split("\n")
        prices_array = np.concatenate((prices_array, temp_array))
    else:
        prices_array = owl_item_prices.split("\n")

# If it is an item that requires hovering, needs much further processing
def hovered_item(owl_item_prices, owl_item_name):
    # OCR configurations (3 is default)
    config = "--psm 6"

    image2 = cv2.resize(owl_item_prices, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    owl_item_prices = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # pre-processing NAME image
    adjusted_name = adjust_gamma(owl_item_name, gamma=0.5)
    blur_name = cv2.GaussianBlur(adjusted_name, (5, 5), 0)
    ret, thresh_name = cv2.threshold(blur_name, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("adjusted", adjusted_name)
    # cv2.imshow("Original Name", owl_item_name)
    # cv2.imshow("Thresh Name", thresh_name)
    # cv2.waitKey(0)
    process_name = pytesseract.image_to_string(thresh_name, lang='eng', config=config)
    # processing NAME text
    # replace anything that is not properly formatted
    string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%]', '', process_name)
    # grabs first string and regex to split capital letters
    item_name = string_without_specialchars_name.partition('\n')[0].strip()
    item_name = re.search(r'Search results for (.*?) that you entered', item_name).group(1)
    # print(item_name)

    # crop PRICES based on mouse position
    # Read the template
    template = cv2.imread('template.png', 0)
    template = cv2.resize(template, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Store width and height of template in w and h
    w, h = template.shape[::-1]
    # Perform match operations.
    res = cv2.matchTemplate(owl_item_prices, template, cv2.TM_CCOEFF_NORMED)
    # Specify a threshold
    threshold = 0.7
    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)
    # Draw a rectangle around the matched region.
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(owl_item_prices, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        print("")
    # Show the final image with the matched area.
    # cv2.imshow('Detected Cursor', owl_item_prices)
    # cv2.waitKey(0)
    cursor_coord_x = pt[0]
    cursor_coord_y = pt[1]

    # crop to get the individual item price
    indiv_item_price = owl_item_prices[cursor_coord_y - 35:cursor_coord_y + 60, cursor_coord_x - 680:cursor_coord_x - 265]
    # pre-process item price
    adjusted_price = adjust_gamma(indiv_item_price, gamma=0.5)
    blur_price = cv2.GaussianBlur(adjusted_price, (5, 5), 0)
    ret, thresh_price = cv2.threshold(blur_price, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('Original Cropped Price', indiv_item_price)
    # cv2.imshow('Cropped Thresh', thresh_price)
    cv2.waitKey(0)
    # processing individual price text
    item_price = pytesseract.image_to_string(thresh_price, lang='eng', config=config)
    lines_name = item_price.split("\n")
    non_empty_lines_name = [line for line in lines_name if line.strip() != ""]
    string_without_empty_lines_name = ""
    for line in non_empty_lines_name:
        string_without_empty_lines_name += line + "\n"
    string_without_specialchars_name = re.sub('[^a-zA-Z0-9\s(\)+%],.', '', string_without_empty_lines_name)
    # save individual item price
    owl_item_price = string_without_specialchars_name.replace("\n", "")
    owl_item_price = owl_item_price.replace(".", ",")
    owl_item_price = owl_item_price.replace(",", "")
    # print(owl_item_price)

    # roughly crop hover box
    item_hover_box = owl_item_prices[cursor_coord_y + 80:cursor_coord_y + 1400, cursor_coord_x + 80:cursor_coord_x + 1100]
    adjusted = adjust_gamma(item_hover_box, gamma=0.4)
    blur = cv2.GaussianBlur(adjusted, (5, 5), 0)
    thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, None, iterations=3)
    thresh = cv2.dilate(thresh, None, iterations=3)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(item_hover_box, contours, -1, (0,255,0), 2)
    # cv2.imshow("Item hover box rough crop", adjusted)
    # cv2.imshow("Item hover box Thresh", thresh)
    # cv2.waitKey(0)
    for contour in contours:
        if cv2.contourArea(contour) < 100000:
             continue
        # print(cv2.contourArea(contour))
        # Show bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(item_hover_box, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Hover Pop-up: Contours", item_hover_box)
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
        cv2.polylines(item_hover_box, roi_corners, 1, (255, 0, 0), 3)
        # cv2.imshow('Hover Pop-up: Name', item_hover_box)
        # cv2.waitKey(0)
        cropped_image = item_hover_box[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        cv2.imwrite('hover_popup_box_owl.tiff', cropped_image)

    image_hover = cv2.imread('hover_popup_box_owl.tiff')
    image_stats = cv2.cvtColor(image_hover, cv2.COLOR_BGR2GRAY)
    # pre-processing STATS image
    adjusted_stats = adjust_gamma(image_stats, gamma=1.5)
    # denoising NAME image
    dst_stats = cv2.fastNlMeansDenoising(adjusted_stats, None, 5, 5, 5)
    # process to make it easier to read text
    ret, thresh_stats = cv2.threshold(dst_stats, 136, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Hover: Thresh", thresh_stats)
    # cv2.waitKey(0)
    text_stats = pytesseract.image_to_string(thresh_stats, lang='maple', config=config)
    lines_stats = text_stats.split("\n")
    non_empty_lines_stats = [line for line in lines_stats if line.strip() != ""]
    string_without_empty_lines_stats = ""
    for line in non_empty_lines_stats:
        string_without_empty_lines_stats += line + "\n"
    # replace anything that is not properly formatted
    string_without_specialchars_stats = re.sub('[^a-zA-Z0-9\s(\)+%:]', '', string_without_empty_lines_stats)
    # grabs first string and regex to split capital letters
    text_stats = string_without_specialchars_stats.replace('\n', ', ')
    save_prices(item_name, owl_item_price, text_stats)

def calculate_price():
    global prices_array
    filter_object = list(filter(lambda x: x != "", prices_array))
    remove_str = filter(str.isnumeric, filter_object)
    res = [int(i) for i in remove_str]

    list_sum = sum(res)
    list_len = len(res)
    price_average = (list_sum)/(list_len)

    print(filter_object)
    print(res)
    print(list_sum)
    print(list_len)
    print(price_average)

    save_item(price_average)

# This function will save the item NAME, PRICE, AND STATS into a txt file AND csv
def save_item(price_average):
    global img_name
    global owl_item_name
    # Storing text into txt file
    with open('raw_owl_text.txt', 'a') as f:
        print(owl_item_name + "|" + str(price_average) + "|" + img_name, file=f)

    # combining into csv
    dataset = pd.read_csv('raw_owl_text.txt', names=['Item Name', 'Average Price', 'Frame Name'], delimiter='|')
    # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
    print(dataset)
    dataset.to_csv(r'owl_items.csv')

def save_prices(item_name, owl_item_price, text_stats):
    global img_name
    # Storing text into txt file
    with open('raw_owl_text_equips.txt', 'a') as f:
        print(str(item_name) + "|" + str(owl_item_price) + "|" + str(text_stats) + "|" + img_name, file=f)

    # combining into csv
    dataset = pd.read_csv('raw_owl_text_equips.txt', names=['Item Name', 'Price', 'Stats', 'Frame Name'], delimiter='|')
    # dataset = dataset.groupby('Item Name').agg({'Price Read': lambda x: ' '.join(x)})
    print(dataset)
    dataset.to_csv(r'owl_equips.csv')


process_all_images()