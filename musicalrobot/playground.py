import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import rescale
from skimage import exposure
from sklearn.preprocessing import normalize

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
    # frame = frame % 0.89
    crop_frame.append(frame[35:85, 40:120])
    # crop_frame.append(frame[20:100, 15:140])
    # crop_frame.append(frame)

# result = crop_frame
# for i in range(len(crop_frame)):
#     crop_frame[i] = rescale(crop_frame[i], scale=(2, 2))

# f_1 = plt.figure(1)
# plt.imshow(crop_frame[-1], cmap='Greys')

# increasing gamma
# for i in range(len(crop_frame)):
#     crop_frame[i] = exposure.adjust_gamma(crop_frame[i], gamma=2, gain=1)

#  sharpening image
filter_blurred_f = ndimage.gaussian_filter(crop_frame, 1)
# alpha = 15
alpha = 0
result = crop_frame + alpha * (crop_frame - filter_blurred_f)

f_2 = plt.figure(2)
# plt.imshow(result[-1], cmap='Greys', vmin=32700, vmax=33000)
# plt.imshow(result[-1], cmap='Greys')
plt.imshow(result[0])
plt.colorbar()

# plt.clim(30800, 30300)

flip_frames, sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(crop_frame, 3, 3)

# Plotting the original image with the samples
# and centroid and plate location
plt.imshow(flip_frames[0])
plt.scatter(sorted_regprops[0]['Plate_coord'],sorted_regprops[0]['Row'],c='orange',s=6)
plt.scatter(sorted_regprops[0]['Column'],sorted_regprops[0]['Row'],s=6,c='red')
plt.title('Sample centroid and plate locations at which the temperature profile is monitored')


plt.show()