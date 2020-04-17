import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import rescale

# Importing the required modules
from musicalrobot import irtemp
from musicalrobot import edge_detection_MN as ed
from musicalrobot import pixel_analysis as pa

frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')  # default
# frames = ed.input_file('../musicalrobot/data_MN/PPA_Melting_6_14_19.tiff')

print(frames[0].shape)
print(type(frames))

crop_frame = []
for frame in frames:
    crop_frame.append(frame[35:85, 40:120])
    # crop_frame.append(frame[20:100, 15:140])
    # crop_frame.append(frame)

# result = crop_frame
# for i in range(len(crop_frame)):
#     crop_frame[i] = rescale(crop_frame[i], scale=(2, 2))

filter_blurred_f = ndimage.gaussian_filter(crop_frame, 1)
alpha = 8
# alpha = 0
result = crop_frame + alpha * (crop_frame - filter_blurred_f)

# plt.imshow(result[-1], cmap='Greys', vmin=32700, vmax=33000)
plt.imshow(result[-1], cmap='Greys')
plt.colorbar()
# plt.clim(30800, 30300)
flip_frames, sorted_regprops, s_temp, p_temp, inf_temp, m_df = ed.inflection_temp(result, 3, 3)

plt.show()