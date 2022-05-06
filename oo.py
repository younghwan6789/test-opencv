from turtle import end_fill
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import datetime as pydatetime

print('start: ', pydatetime.datetime.now())

filelist = [f for f in listdir('.') if isfile(join('.', f))]

base_filename = '1.png'
base = cv.imread(base_filename)

hsv_base = cv.cvtColor(base, cv.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]

hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

compare_method = cv.HISTCMP_CORREL
max_diff = 0
max_file = ''

for filename in filelist:
    if (filename != base_filename and (filename.endswith('.png') or filename.endswith('.jpg'))):
        # print(filename)
        hsv_target = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2HSV)
        hist_target = cv.calcHist([hsv_target], channels, None, histSize, ranges, accumulate=False)
        cv.normalize(hsv_target, hsv_target, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        
        diff = cv.compareHist(hist_base, hist_target, compare_method)
        
        if (diff > 0.9999):
            print('99.99%')
            print('file : ', filename, ', diff : ', diff)
            break

        if (max_diff < diff):
            max_diff = diff
            max_file = filename
        
        # print('file : ', filename, ', diff : ', diff)

print('max_diff : ', max_diff, '(', max_file, ')')

print('finish: ', pydatetime.datetime.now())
