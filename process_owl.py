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

    # ## FOR TESTING PURPOSES: one image at a time
    # global img_name
    # img = cv2.imread("./owl_dataset/frame2.tiff")
    # # Saving frame name for future reference
    # img_name = "frame2"
    # process_owl(img)

    # # For running: multiple images at a time
    for filename in os.listdir(directory):
        if filename.endswith(".tiff"):
            global img_name
            img_name = filename
            filepath = os.path.join(directory, filename).replace("\\", "/")
            print(filepath)
            img = cv2.imread(filepath)
            process_owl(img)

    calculate_price()

# This function will process the text from the cropped screenshot of the item HOVER POPUP
def process_owl(img):
    global owl_item_name
    # owl_box = img[350:750, 675:1250]
    owl_item_name = img[350:381, 675:1250]
    owl_item_name = cv2.cvtColor(owl_item_name, cv2.COLOR_BGR2GRAY)

    owl_item_prices = img[480:730, 905:1025]
    owl_item_prices = cv2.cvtColor(owl_item_prices, cv2.COLOR_BGR2GRAY)

    # pre-processing NAME image
    adjusted_name = adjust_gamma(owl_item_name, gamma=3.3)
    # denoising NAME image
    dst_name = cv2.fastNlMeansDenoising(adjusted_name, None, 2, 2, 2)
    # process to make it easier to read text
    ret, thresh_name = cv2.threshold(dst_name, 246, 255, cv2.THRESH_BINARY)

    # pre-processing PRICES image
    adjusted_prices = adjust_gamma(owl_item_prices, gamma=3.3)
    # denoising NAME image
    dst_prices = cv2.fastNlMeansDenoising(adjusted_prices, None, 5, 5, 5)
    # process to make it easier to read text
    ret, thresh_prices = cv2.threshold(dst_prices, 248, 255, cv2.THRESH_BINARY)

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

    if("prices_array" in globals()):
        temp_array = owl_item_prices.split("\n")
        prices_array = np.concatenate((prices_array, temp_array))
    else:
        prices_array = owl_item_prices.split("\n")

def calculate_price():
    global prices_array
    filter_object = list(filter(lambda x: x != "", prices_array))
    remove_str = filter(str.isdigit, filter_object)
    res = [int(i) for i in remove_str]

    list_sum = sum(res)
    list_len = len(filter_object)
    price_average = (list_sum)/(list_len)

    print(filter_object)
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

process_all_images()