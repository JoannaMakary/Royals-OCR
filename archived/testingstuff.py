
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

img = cv2.imread("hover_popup_name.tiff")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_name = img_gray[5: 100, 17: 320]
image_name = cv2.resize(image_name, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

def adjust_gamma(crop_img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(crop_img, table)

adjusted = adjust_gamma(image_name, gamma=1.5)
# denoising image
dst = cv2.fastNlMeansDenoising(adjusted, None, 2, 2, 2)

# global thresholding
ret1,thresh1 = cv2.threshold(dst,136,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,thresh2 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# OCR configurations (3 is default)
config = "--psm 6"
text_name1 = pytesseract.image_to_string(thresh1, lang='maple', config=config)
text_name2 = pytesseract.image_to_string(thresh2, lang='maple', config=config)

cv2.imshow("gray", image_name)
cv2.imshow("adjusted", adjusted)
cv2.imshow("thresh1", thresh1)
cv2.imshow("thresh2", thresh2)
cv2.waitKey(0)

print(text_name1)
print(text_name2)