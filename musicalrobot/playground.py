import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import signal
from skimage.transform import rescale
from skimage import filters
from skimage import feature
from skimage.measure import label
from skimage.morphology import remove_small_objects, erosion, binary_erosion
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.measure import regionprops

from sklearn.preprocessing import normalize, binarize
import math
import cv2

# Importing the required modules
from musicalrobot import irtemp
from musicalrobot import edge_detection_MN as ed
from musicalrobot import edge_detection_MN_ver2 as ed2
from musicalrobot import pixel_analysis as pa

frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')  # default
# frames = ed.input_file('../musicalrobot/data_MN/PPA_Melting_6_14_19.tiff')
# frames = ed.input_file('../musicalrobot/data_MN/10_17_19_quinine_shallow_plate.tiff')

print(frames.shape)
# print(type(frames))

crop_frame = []
for frame in frames:
    # frame = normalize(frame, norm='max')
    # frame = frame/frame.max()
    crop_frame.append(frame[35:85, 40:120])
    # crop_frame.append(frame[20:100, 15:140])
    # crop_frame.append(frame)

# upsizing
# for i in range(len(crop_frame)):
#     crop_frame[i] = rescale(crop_frame[i], scale=(2, 2))


#  sharpening image
filter_blurred_f = ndimage.gaussian_filter(crop_frame, 0.1)
# alpha = 1
alpha = 0
result = crop_frame + alpha * (crop_frame - filter_blurred_f)

# plt.imshow(result[-1], cmap='Greys', vmin=32700, vmax=33000)
# plt.figure(1)
# plt.imshow(result[0])

# CV method to remove background
# file_path = "../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff"

# cap = cv2.VideoCapture(file_path)
# # cap = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
# # print(cap.shape)
# first_iter = True
# result = None
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         break
#
#     if first_iter:
#         avg = np.float32(frame)
#         first_iter = False
#
#     cv2.accumulateWeighted(frame, avg, 0.005)
#     result = cv2.convertScaleAbs(avg)
#
# plt.imshow(result)
# plt.show()


# plt.imshow(np.power(result[0],2), cmap='Greys')
# plt.imshow(result[0] - result.mean(0))
# plt.imshow(np.power(result[0] - result.mean(0),1), cmap='Greys')  # background removal


# plt.show()

# gaussian fiter
# gauss = ndimage.gaussian_filter(result[0], sigma=5)
# plt.imshow(gauss, cmap='gray')

# TODO sobel

# fig = plt.figure(3)
# plt.imshow(binarize(result[-1]-result.mean(0)), cmap='Greys')

# TODO equalizing
# img1 = np.uint8(cv2.normalize(crop_frame[0], None, 0, 255, cv2.NORM_MINMAX))
# img1 = np.uint8(cv2.normalize(crop_frame[-1]-result.mean(0), None, 0, 255, cv2.NORM_MINMAX))

# img_eq = cv2.equalizeHist(img1,0)  # not really good
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img_eq = clahe.apply(img1)

# plt.imshow(img_eq)
# plt.show()

# sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3)


# img_eq = result[0] - result.mean(0)
img_eq = result[50] - result.mean(0)

# TODO background compensation
# plt.figure(2)
time = 0
# time = len(result)-1
img_eq = result[time] - result.mean(0)*time/(len(result)-1)

mag1 = filters.sobel(img_eq)
mag1 = mag1 > mag1.mean()*3
# plt.imshow(mag1)
# plt.show()

alpha = 2
fig = plt.figure(1)
mag1 = filters.sobel(result[0])
mag1 = mag1 > mag1.mean() * alpha
# plt.imshow(mag1)

# fig = plt.figure(2)
# labeled_img = label(mag1)
# border = find_boundaries(labeled_img, mode='inner')
# mag1 = erosion(mag1)
# mag1 = mag1 > mag1.mean() * alpha

# mag1 = binary_fill_holes(mag1)
# plt.imshow(mag1)
# plt.show()

# labeled_samples = ed.edge_detection(crop_frame, 9, track=True)
# print(len(labeled_samples.shape), len(labeled_samples))
# for time in range(465,581):
# for time in range(581, 800):
#     plt.imshow(labeled_samples[time])
#     plt.pause(.0001)
#     plt.draw()

# plt.imshow(labeled_samples[22])
# plt.show()

# Plotting the original image with the samples
# and centroid and plate location
# sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3, ver=2)
# print('done!')
frame = 886

n_rows = 3
n_columns = 3
labeled_samples = ed.edge_detection(crop_frame, 9, track=True)
regprops = ed2.regprop(labeled_samples, crop_frame, n_rows, n_columns)
# sorted_regprops = ed2.sort_regprops(regprops, n_columns, n_rows)
print('done!')
# Plotting the temperature profile of a sample against the temperature profile
# of the plate at a location next to the sample.
# sample_id = 5
# f_1 = plt.figure(1)
# y = s_temp[sample_id]
# y = signal.savgol_filter(y, 101, 3)
# plt.plot(p_temp[sample_id], y)
# plt.ylabel('Temperature of the sample($^\circ$C)')
# plt.xlabel('Temperature of the well plate($^\circ$C)')
# plt.title('Temperature of the sample against the temperature of the plate')
# plt.axis([30, 55, 30, 55])
# plt.grid()

# sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3)
# f_2 = plt.figure(2)
# plt.plot(p_temp[sample_id], s_temp[sample_id])
# plt.ylabel('Temperature of the sample($^\circ$C)')
# plt.xlabel('Temperature of the well plate($^\circ$C)')
# plt.title('Temperature of the sample against the temperature of the plate')
# plt.axis([30, 55, 30, 55])
# plt.grid()

plt.show()

fig = plt.figure(3)
fig_canny = plt.figure(4)
time = [0, int(0.2*len(result)), int(0.4*len(result)), int(0.6*len(result)), int(0.8*len(result)), -1]
edge = True
row = 8
col = 5
alpha = 2
for i in range(5):
    img_raw = result[time[i]]
    ax = fig.add_subplot(row, col, i + 1)
    if i is 0:
        ax.set_ylabel('OG')
    ax.imshow(img_raw)

    # sobel
    mag1 = filters.sobel(img_raw)
    if edge:
        mag1 = mag1 > mag1.mean() * alpha
        mag1 = ndimage.binary_erosion(mag1)
        mag1 = binary_fill_holes(mag1)
        # mag1 = remove_small_objects(mag1, 15)
    ax = fig.add_subplot(row, col, i + 6)
    if i is 0:
        ax.set_ylabel('sobel')
    ax.imshow(mag1)

    mag1_bg = filters.sobel(img_raw - result.mean(0))
    if edge:
        mag1 = mag1_bg > mag1_bg.mean() * alpha
        mag1 = ndimage.binary_erosion(mag1)
        mag1_bg = binary_fill_holes(mag1)
        # mag1 = remove_small_objects(mag1, 15)
    ax = fig.add_subplot(row, col, i + 11)
    if i is 0:
        ax.set_ylabel('bg removal')
    ax.imshow(mag1_bg)

    # linear background removal
    mag1_lin = filters.sobel(result[time[i]] - result.mean(0)*time[i] / (len(result) - 1))
    if edge:
        mag1 = mag1_lin > mag1_lin.mean() * alpha
        mag1 = ndimage.binary_erosion(mag1)
        mag1_lin = binary_fill_holes(mag1)
    # mag1_lin = mag1 + mag1_bg
        # mag1 = remove_small_objects(mag1,15)
    ax = fig.add_subplot(row, col, i + 16)
    if i is 0:
        ax.set_ylabel('lin bg')
    ax.imshow(mag1_lin)

    # progressive background removal
    progressive_background = None
    if time[i] is 0:
        progressive_background = 0
    else:
        progressive_background = result[0:time[i]].mean(0)
    mag1_prog = filters.sobel(img_raw - progressive_background)
    if edge:
        mag1 = mag1_prog > mag1_prog.mean() * alpha
        mag1 = ndimage.binary_erosion(mag1)
        mag1_prog = binary_fill_holes(mag1)
        # mag1 = remove_small_objects(mag1, 15)
    ax = fig.add_subplot(row, col, i + 21)
    if i is 0:
        ax.set_ylabel('prog bg')
    ax.imshow(mag1_prog)

    # combined background
    ax = fig.add_subplot(row, col, i + 26)
    if i is 0:
        ax.set_ylabel('prog & linear')
    ax.imshow(mag1_prog + mag1_lin)

    # combined background
    ax = fig.add_subplot(row, col, i + 31)
    if i is 0:
        ax.set_ylabel('prog & bg')
    ax.imshow(mag1_prog + mag1_bg)

    # combined background all
    mag1 = filters.sobel(result[time[i]] - result.mean(0)*np.power(time[i]/(len(result) - 1),2))
    if edge:
        mag1 = mag1 > mag1.mean() * alpha
        mag1 = ndimage.binary_erosion(mag1)
        mag1 = binary_fill_holes(mag1)
    ax = fig.add_subplot(row, col, i + 36)
    if i is 0:
        ax.set_ylabel('quad bg')
    ax.imshow(mag1)

    # canny
    mag1 = feature.canny(img_raw / 1500)
    mag1 = binary_fill_holes(mag1)
    mag1 = remove_small_objects(mag1, 15)
    ax = fig_canny.add_subplot(2, 5, i +1)
    if i is 0:
        ax.set_ylabel('canny method')
    ax.imshow(mag1)

    # canny with background removal
    mag1 = feature.canny((img_raw - result.mean(0)) / 1500)
    mag1 = binary_fill_holes(mag1)
    mag1 = remove_small_objects(mag1, 15)
    ax = fig_canny.add_subplot(2, 5, i + 6)
    if i is 0:
        ax.set_ylabel('canny & bg')
    ax.imshow(mag1)

# row = result[0].shape[0]
# col = result[0].shape[1]
# a = np.empty(result.shape)
# print(i)
# a[i] = mag1
# print(a.shape)

plt.show()


# loc_index
# for pixel in range(len(loc_index)):
#     x = pixel[1]
#     y = pixel[0]
#     total = total + labeled_frame[y,x]
# sample_temp = total/len(loc_index)
