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
from musicalrobot import edge_detection as edOG
from musicalrobot import edge_detection_ver2 as ed
from musicalrobot import pixel_analysis as pa

frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')  # default
# frames = ed.input_file('D:/wsl/DIRECT/musical-robot-ver2/musicalrobot/data_MN/10_17_19_DDA_shallow_plate.tiff')
# frames = ed.input_file('D:/wsl/DIRECT/musical-robot-ver2/musicalrobot/data_MN/8_15_19_Dodecanoic_acid.tiff')
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
    # crop_frame.append(frame[10:110, 15:160])
    # crop_frame.append(frame)

# upsizing
# for i in range(len(crop_frame)):
#     crop_frame[i] = rescale(crop_frame[i], scale=(2, 2))


#  sharpening image
filter_blurred_f = ndimage.gaussian_filter(crop_frame, 0.1)
# alpha = 1
alpha = 0
result = crop_frame + alpha * (crop_frame - filter_blurred_f)


# plt.imshow(np.power(result[0],2), cmap='Greys')
# plt.imshow(result[0] - result.mean(0))
# plt.imshow(np.power(result[0] - result.mean(0),1), cmap='Greys')  # background removal

# plt.imshow(crop_frame[0])
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
# print(props[0:].centroid[0])
# labeled_samples = ed.edge_detection(crop_frame, 9, track=True)
# regprops = ed.regprop(labeled_samples, crop_frame, n_rows, n_columns)
# sorted_regprops = ed.sort_regprops(regprops, n_columns, n_rows)
# plt.imshow(crop_frame[22])
# plt.show()

print('done!')
# Plotting the temperature profile of a sample against the temperature profile
# of the plate at a location next to the sample.
sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3)
sorted_regprops_og, s_temp_og, p_temp_og, inf_temp_og, m_df_og = edOG.inflection_temp(crop_frame, 3, 3)
sample_id = 7
f_1 = plt.figure(1)
# y = s_temp[sample_id]
# y = signal.savgol_filter(y, 101, 3)
# plt.plot(p_temp[sample_id], y)
# plt.ylabel('Temperature of the sample($^\circ$C)')
# plt.xlabel('Temperature of the well plate($^\circ$C)')
# plt.title('Temperature of the sample against the temperature of the plate')
# plt.axis([30, 55, 30, 55])
# plt.grid()
print(len(sorted_regprops))

time = 22
# plt.imshow(crop_frame[time])
# plt.scatter(sorted_regprops[time]['Plate_coord'],sorted_regprops[time]['Row'],c='orange',s=6)
# plt.scatter(sorted_regprops[time]['Plate_coord'],sorted_regprops[time]['Row'],c='orange',s=6)
# plt.scatter(sorted_regprops[time]['Column'],sorted_regprops[time]['Row'],s=6,c='red')
# plt.axis('off')

# Plotting the original image with the samples
# and centroid and plate location
p1 = f_1.add_subplot(1, 2, 1)
p2 = f_1.add_subplot(1, 2, 2)

while True:
    for time in range(100, 400):
        time = int(time*2)

        p1.imshow(crop_frame[time])
        p2.imshow(crop_frame[time])

        # plt.scatter(sorted_regprops[time]['Plate_coord'],sorted_regprops[time]['Row'],c='orange',s=6)
        p1.scatter(sorted_regprops_og[time]['Column'],sorted_regprops_og[time]['Row'],s=6,c='red')
        p2.scatter(sorted_regprops[time]['Column'],sorted_regprops[time]['Row'],s=6,c='red')

        p1.set_xlabel('Previous Version')
        p2.set_xlabel('With Tracking')

        plt.pause(0.001)
        plt.draw()
        # f_1.clear()
        p1.clear()
        p2.clear()


# time = 0
# p1.imshow(crop_frame[time])
# p2.imshow(crop_frame[time])
# p1.scatter(sorted_regprops[time]['Column'],sorted_regprops[time]['Row'],s=6,c='red')
# plt.show()

# sorted_regprops2, s_temp, p_temp, inf_temp, m_df = edOG.inflection_temp(crop_frame, 3, 3)
# f_2 = plt.figure(2)
# plt.plot(p_temp[sample_id], s_temp[sample_id])
# plt.ylabel('Temperature of the sample($^\circ$C)')
# plt.xlabel('Temperature of the well plate($^\circ$C)')
# plt.title('Temperature of the sample against the temperature of the plate')
# plt.axis([30, 55, 30, 55])
# plt.grid()
# print(inf_temp)

# plt.show()

# fig = plt.figure(3)
# fig_canny = plt.figure(4)
# time = [0, int(0.2*len(result)), int(0.4*len(result)), int(0.6*len(result)), int(0.8*len(result)), -1]
# edge = True
# row = 8
# col = 5
# alpha = 2
# for i in range(5):
#     img_raw = result[time[i]]
#     ax = fig.add_subplot(row, col, i + 1)
#     if i is 0:
#         ax.set_ylabel('OG')
#     ax.imshow(img_raw)
#
#     # sobel
#     mag1 = filters.sobel(img_raw)
#     if edge:
#         mag1 = mag1 > mag1.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1 = binary_fill_holes(mag1)
#         # mag1 = remove_small_objects(mag1, 15)
#     ax = fig.add_subplot(row, col, i + 6)
#     if i is 0:
#         ax.set_ylabel('sobel')
#     ax.imshow(mag1)
#
#     mag1_bg = filters.sobel(img_raw - result.mean(0))
#     if edge:
#         mag1 = mag1_bg > mag1_bg.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_bg = binary_fill_holes(mag1)
#         # mag1 = remove_small_objects(mag1, 15)
#     ax = fig.add_subplot(row, col, i + 11)
#     if i is 0:
#         ax.set_ylabel('bg removal')
#     ax.imshow(mag1_bg)
#
#     # linear background removal
#     mag1_lin = filters.sobel(result[time[i]] - result.mean(0)*time[i] / (len(result) - 1))
#     if edge:
#         mag1 = mag1_lin > mag1_lin.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_lin = binary_fill_holes(mag1)
#     # mag1_lin = mag1 + mag1_bg
#         # mag1 = remove_small_objects(mag1,15)
#     ax = fig.add_subplot(row, col, i + 16)
#     if i is 0:
#         ax.set_ylabel('lin bg')
#     ax.imshow(mag1_lin)
#
#     # progressive background removal
#     progressive_background = None
#     if time[i] is 0:
#         progressive_background = 0
#     else:
#         progressive_background = result[0:time[i]].mean(0)
#     mag1_prog = filters.sobel(img_raw - progressive_background)
#     if edge:
#         mag1 = mag1_prog > mag1_prog.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_prog = binary_fill_holes(mag1)
#         # mag1 = remove_small_objects(mag1, 15)
#     ax = fig.add_subplot(row, col, i + 21)
#     if i is 0:
#         ax.set_ylabel('prog bg')
#     ax.imshow(mag1_prog)
#
#     # combined background
#     ax = fig.add_subplot(row, col, i + 26)
#     if i is 0:
#         ax.set_ylabel('prog & linear')
#     ax.imshow(mag1_prog + mag1_lin)
#
#     # combined background
#     ax = fig.add_subplot(row, col, i + 31)
#     if i is 0:
#         ax.set_ylabel('prog & bg')
#     ax.imshow(mag1_prog + mag1_bg)
#
#     # combined background all
#     mag1 = filters.sobel(result[time[i]] - result.mean(0)*np.power(time[i]/(len(result) - 1),2))
#     if edge:
#         mag1 = mag1 > mag1.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1 = binary_fill_holes(mag1)
#     ax = fig.add_subplot(row, col, i + 36)
#     if i is 0:
#         ax.set_ylabel('quad bg')
#     ax.imshow(mag1)
#
#     # canny
#     mag1 = feature.canny(img_raw / 1500)
#     mag1 = binary_fill_holes(mag1)
#     mag1 = remove_small_objects(mag1, 15)
#     ax = fig_canny.add_subplot(2, 5, i +1)
#     if i is 0:
#         ax.set_ylabel('canny method')
#     ax.imshow(mag1)
#
#     # canny with background removal
#     mag1 = feature.canny((img_raw - result.mean(0)) / 1500)
#     mag1 = binary_fill_holes(mag1)
#     mag1 = remove_small_objects(mag1, 15)
#     ax = fig_canny.add_subplot(2, 5, i + 6)
#     if i is 0:
#         ax.set_ylabel('canny & bg')
#     ax.imshow(mag1)


# fig = plt.figure(figsize=(10,25))
# time = [0, int(0.2*len(result)), int(0.4*len(result)), int(0.6*len(result)), int(0.8*len(result)), -1]
# edge = True
# row = 8
# col = 5
# alpha = 2
# for i in range(5):
#     # fig.tight_layout()
#     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1)
#     img_raw = result[time[i]]
#     ax = fig.add_subplot(row, col, i + 1)
#     plt.axis('off')
#     if i is 0:
#         ax.set_ylabel('OG')
#     ax.imshow(img_raw)
#
#     # sobel
#     mag1 = filters.sobel(img_raw)
#     if edge:
#         mag1 = mag1 > mag1.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1 = binary_fill_holes(mag1)
#         # mag1 = remove_small_objects(mag1, 15)
#
#     mag1_bg = filters.sobel(img_raw - result.mean(0))
#     if edge:
#         mag1 = mag1_bg > mag1_bg.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_bg = binary_fill_holes(mag1)
#
#     # linear background removal
#     mag1_lin = filters.sobel(result[time[i]] - result.mean(0)*time[i] / (len(result) - 1))
#     if edge:
#         mag1 = mag1_lin > mag1_lin.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_lin = binary_fill_holes(mag1)
#
#     # progressive background removal
#     progressive_background = None
#     if time[i] is 0:
#         progressive_background = 0
#     else:
#         progressive_background = result[0:time[i]].mean(0)
#     mag1_prog = filters.sobel(img_raw - progressive_background)
#     if edge:
#         mag1 = mag1_prog > mag1_prog.mean() * alpha
#         mag1 = ndimage.binary_erosion(mag1)
#         mag1_prog = binary_fill_holes(mag1)
#         # mag1 = remove_small_objects(mag1, 15)
#
#     # canny
#     mag1 = feature.canny(img_raw / 1500)
#     mag1 = binary_fill_holes(mag1)
#     mag1 = remove_small_objects(mag1, 15)
#     ax = fig.add_subplot(row, col, i + 6)
#     plt.axis('off')
#     if i is 0:
#         ax.set_ylabel('canny method')
#     ax.imshow(mag1)
#
#     # combined background
#     ax = fig.add_subplot(row, col, i + 11)
#     if i is 0:
#         ax.set_ylabel('prog & linear')
#     plt.axis('off')
#     ax.imshow(mag1_prog + mag1_lin)

plt.show()


# loc_index
# for pixel in range(len(loc_index)):
#     x = pixel[1]
#     y = pixel[0]
#     total = total + labeled_frame[y,x]
# sample_temp = total/len(loc_index)
