import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import rescale
from skimage import filters
from skimage import feature
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes

from sklearn.preprocessing import normalize, binarize
import math
import cv2

# Importing the required modules
from musicalrobot import irtemp
from musicalrobot import edge_detection_MN as ed
from musicalrobot import pixel_analysis as pa

frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')  # default
# frames = ed.input_file('../musicalrobot/data_MN/PPA_Melting_6_14_19.tiff')
# frames = ed.input_file('../musicalrobot/data_MN/10_17_19_quinine_shallow_plate.tiff')

print(frames.shape)
print(type(frames))

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
plt.figure(1)
plt.imshow(result[0])

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

# cv2.imshow("result", result)
# cv2.imwrite("averaged_frame.jpg", result)
# cv2.waitKey(0)

# plt.imshow(np.power(result[0],2), cmap='Greys')
# plt.imshow(result[0] - result.mean(0))
# plt.imshow(np.power(result[0] - result.mean(0),1), cmap='Greys')  # background removal


# plt.show()

# gaussian fiter
# gauss = ndimage.gaussian_filter(result[0], sigma=5)
# plt.imshow(gauss, cmap='gray')

# TODO sobel
# fig = plt.figure(2)
# im = result[0]
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag1 = np.hypot(dx, dy)  # magnitude
# mag1 *= 255.0 / np.max(mag1)  # normalize (Q&D)
#
# im = result[-1]
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag2 = np.hypot(dx, dy)  # magnitude
# mag2 *= 255.0 / np.max(mag2)  # normalize (Q&D)
#
# im = result[0] - result.mean(0)
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag3 = np.hypot(dx, dy)  # magnitude
# mag3 *= 255.0 / np.max(mag3)  # normalize (Q&D)
#
# im = result[-1] - result.mean(0)
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag4 = np.hypot(dx, dy)  # magnitude
# mag4 *= 255.0 / np.max(mag4)  # normalize (Q&D)
#
# ax1 = fig.add_subplot(221)  # left side
# ax2 = fig.add_subplot(222)  # right side
# ax3 = fig.add_subplot(223)  # right side
# ax4 = fig.add_subplot(224)  # right side
# ax1.imshow(mag1 / mag1.max())  # init
# ax2.imshow(mag2)  # fin
# ax3.imshow(mag3)  # avg init
# ax4.imshow(mag4)  # avg fin
# plt.show()

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


# fig = plt.figure(3)
# im = result[-1]/result[-1].max()
# img_eq = result[-1] - result.mean(0)

# img_eq = result[0] - result.mean(0)
img_eq = result[50] - result.mean(0)

# TODO background compensation
plt.figure(2)
time = 0
# time = len(result)-1
img_eq = result[time] - result.mean(0)*time/(len(result)-1)

mag1 = filters.sobel(img_eq)

mag1 = mag1 > mag1.mean()*3
plt.imshow(mag1)
# plt.show()

# plt.figure(20)
# avg = result[0]
# print(type(avg))
# # time = len(result)
# time = 2
# # for i in range(time):
# #     avg = result[i] + avg
# # avg = avg/time
# avg = result[0:time].mean(0)
# plt.imshow(avg)
# # plt.imshow(result[600])
#
#
# plt.figure(21)
# plt.imshow(result.mean(0))
#
# plt.show()


# Plotting the original image with the samples
# and centroid and plate location
# plt.imshow(flip_frames[0])
# plt.scatter(sorted_regprops[0]['Plate_coord'],sorted_regprops[0]['Row'],c='orange',s=6)
# plt.scatter(sorted_regprops[0]['Column'],sorted_regprops[0]['Row'],s=6,c='red')
# plt.title('Sample centroid and plate locations at which the temperature profile is monitored')


# plt.show()

fig = plt.figure(3)

time = [0, 200, 400, 600, len(result)-1]
# selected_frames = 0
for i in range(5):
    img_raw = result[time[i]]
    fig.add_subplot(7, 5, i + 1).imshow(img_raw)

    # sobel
    mag1 = filters.sobel(img_raw)
    # mag1 = mag1 > mag1.mean() * 3
    fig.add_subplot(7, 5, i + 6).imshow(mag1)

    mag1 = filters.sobel(img_raw - result.mean(0))
    # mag1 = mag1 > mag1.mean() * 3
    fig.add_subplot(7, 5, i + 11).imshow(mag1)

    mag1 = filters.sobel(result[time[i]] - result.mean(0)*time[i] / (len(result) - 1))
    # mag1 = mag1 > mag1.mean() * 3
    # mag1 = binary_fill_holes(mag1)
    fig.add_subplot(7, 5, i + 16).imshow(mag1)

    # progressive background removal
    adaptive_background = None
    if time[i] is 0:
        adaptive_background = 0
    else:
        adaptive_background = result[0:time[i]].mean(0)
    mag1 = filters.sobel(img_raw - adaptive_background)
    fig.add_subplot(7, 5, i + 21).imshow(mag1)

    # canny
    mag1 = feature.canny(img_raw / 1500)
    fig.add_subplot(7, 5, i + 26).imshow(mag1)

    # canny with background removal
    mag1 = feature.canny((img_raw - result.mean(0)) / 1500)
    fig.add_subplot(7, 5, i + 31).imshow(mag1)


plt.show()
