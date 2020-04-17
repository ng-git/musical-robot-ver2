import sys
import matplotlib.pyplot as plt
import numpy as np

# Importing the required modules
from musicalrobot import irtemp
from musicalrobot import edge_detection as ed
from musicalrobot import pixel_analysis as pa

frames = ed.input_file('../musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')

# f_1 = plt.figure(1)
# plt.imshow(frames[-1])
# # f_1.show()
#
# f_2 = plt.figure(2)
# plt.imshow(frames[0])
# # f_2.show()
#
# plt.show()

crop_frame = []
# for frame in frames:
#     crop_frame.append(frame[35:85, 40:120])
# plt.imshow(crop_frame[0])
# plt.colorbar()

for frame in frames:
    # crop_frame.append(frame[20:100, 15:140])
    crop_frame.append(frame)
plt.imshow(crop_frame[-1], cmap='Greys')
plt.colorbar()
# plt.clim(30800, 30300)


plt.show()