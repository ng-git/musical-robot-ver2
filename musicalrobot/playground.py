import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import rescale
from skimage import exposure
from sklearn.preprocessing import normalize, binarize
import math

# Importing the required modules
from musicalrobot import irtemp
from musicalrobot import edge_detection_MN as ed
from musicalrobot import pixel_analysis as pa

# frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')  # default
# frames = ed.input_file('../musicalrobot/data_MN/PPA_Melting_6_14_19.tiff')
frames = ed.input_file('../musicalrobot/data_MN/10_17_19_quinine_shallow_plate.tiff')

print(frames.shape)
print(type(frames))

crop_frame = []
for frame in frames:
    frame = normalize(frame, norm='max')
    # frame = frame % 0.89
    crop_frame.append(frame[35:85, 40:120])
    # crop_frame.append(frame[20:100, 15:140])
    # crop_frame.append(frame)

# result = crop_frame
for i in range(len(crop_frame)):
    crop_frame[i] = rescale(crop_frame[i], scale=(2, 2))

# f_1 = plt.figure(1)
# plt.imshow(crop_frame[-1], cmap='Greys')

# increasing gamma
# for i in range(len(crop_frame)):
#     crop_frame[i] = exposure.adjust_gamma(crop_frame[i], gamma=2, gain=1)

#  sharpening image
filter_blurred_f = ndimage.gaussian_filter(crop_frame, 0.1)
# alpha = 1
alpha = 0
result = crop_frame + alpha * (crop_frame - filter_blurred_f)

# plt.imshow(result[-1], cmap='Greys', vmin=32700, vmax=33000)
# plt.imshow(result[0], cmap='Greys')
# plt.imshow(np.power(result[0],2), cmap='Greys')
# plt.imshow(result[0] - result.mean(0))
# plt.imshow(np.power(result[0] - result.mean(0),1), cmap='Greys')  # background removal

# gauss = ndimage.gaussian_filter(result[0], sigma=5)
# plt.imshow(gauss, cmap='gray')

fig = plt.figure(2)

im = result[0]
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag1 = np.hypot(dx, dy)  # magnitude
mag1 *= 255.0 / np.max(mag1)  # normalize (Q&D)

im = result[-1]
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag2 = np.hypot(dx, dy)  # magnitude
mag2 *= 255.0 / np.max(mag2)  # normalize (Q&D)

im = result[0] - result.mean(0)
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag3 = np.hypot(dx, dy)  # magnitude
mag3 *= 255.0 / np.max(mag3)  # normalize (Q&D)

im = result[-1] - result.mean(0)
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag4 = np.hypot(dx, dy)  # magnitude
mag4 *= 255.0 / np.max(mag4)  # normalize (Q&D)


# mag = mag1
# mag = mag2
mag = mag1+mag2

ax1 = fig.add_subplot(221)  # left side
ax2 = fig.add_subplot(222)  # right side
ax3 = fig.add_subplot(223)  # right side
ax4 = fig.add_subplot(224)  # right side
ax1.imshow(mag1)  # init
ax2.imshow(mag2)  # fin
ax3.imshow(mag3)  # avg init
ax4.imshow(mag4)  # avg fin
# ax5.imshow(mag4+mag3)
# plt.show()


fig = plt.figure(3)
# plt.imshow(binarize(result[-1]-result.mean(0)), cmap='Greys')
mixture = mag1 + mag4
plt.imshow(mixture)
plt.colorbar()
# plt.imshow(mag, cmap='Greys')

# print(np.amin(result[0]))
# fig.colorbar()

# plt.clim(30800, 30300)

# flip_frames, sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3)

# Plotting the original image with the samples
# and centroid and plate location
# plt.imshow(flip_frames[0])
# plt.scatter(sorted_regprops[0]['Plate_coord'],sorted_regprops[0]['Row'],c='orange',s=6)
# plt.scatter(sorted_regprops[0]['Column'],sorted_regprops[0]['Row'],s=6,c='red')
# plt.title('Sample centroid and plate locations at which the temperature profile is monitored')


plt.show()