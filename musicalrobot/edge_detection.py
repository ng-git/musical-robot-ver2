import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import skimage
import numpy as np
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from skimage import io
from skimage import feature
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects  
from scipy.signal import find_peaks
from scipy.interpolate import BSpline
from irtemp import centikelvin_to_celsius

##########################################################################################################################################################################
##########################################################################################################################################################################
                                                #######Image Processing Functions#########
##########################################################################################################################################################################
##########################################################################################################################################################################
# Function to load the input file
def input_file(file_name):
    ''' To load the imput file as an array.
    Args:
        file_name (string) : Name of the file to be loaded as it is 
        saved on the disk. Provide file path if it is not in
        the same directory as the jupyter notebook.
    Returns:
        In case of a video, returns an array for each frame
        in the video.
        In case of an image, return an array.    
    '''
    frames = io.imread(file_name)
    return frames

# Function to flip the frames horizontally and vertically to correct for the mirroring 
# during recording.
def flip_frame(frames):
    ''' To flip all the loaded frames horizontally and vertically
        to correct for the mirroring during recording.
        Args:
        frames(Array): An array containing an array for each frame
        in the video or just a single array in case of an image.
        Returns:
        Flipped frames that can be processed to get temperature data.
    '''
    flip_frames = []
    for frame in frames:
        f_frame = np.fliplr(frame)
        flip_frames.append(np.flipud(f_frame))
    return flip_frames

# Function to detect edges, fill and label the samples.
def edge_detection(frames, n_samples):
    ''' To detect the edges of the wells, fill and label them to
    determine their centroids.
    Args:
    frames: The frames to be processed and determine the
    sample temperature from.
    Returns:
    labeled_samples: All the samples in the frame are labeled
    so that they can be used as props to get pixel data from.
    '''
    for size in range(15,9,-1):
        for thres in range(1500,900,-100):
            edges = feature.canny(frames[0]/thres)
            filled_samples = binary_fill_holes(edges)
            cl_samples = remove_small_objects(filled_samples,min_size = size)
            labeled_samples = label(cl_samples)
            props = regionprops(labeled_samples, intensity_image=frames[0])
            if len(props) == n_samples:
                break
#             if thres == 1000 and len(props) != n_samples:
#                 print('Not all the samples are being recognized with the set threshold range for size ',size) 
        if len(props) == n_samples:
            break
    if size == 10 and thres == 1000 and len(props) != n_samples:
        print('Not all the samples are being recognized with the set minimum size and threshold range')
    return labeled_samples

# Function to determine centroids of all the samples
def regprop(labeled_samples, frames, n_rows, n_columns):
    ''' Determines the area and centroid of all samples.
        Args:
        labeled_samples(array): An array with labeled samples.
        flip_frames (array) : Original intensity image to determine
        the intensity at sample centroids.
        n_samples: Number of samples in the video being analyzed.
        n_rows: Number of rows of sample
        n_columns: Number of columns of sample
        Returns:
        A dictionary of dataframe with information about samples in every
        frame of the video.
    '''
    regprops = {}
    n_samples = n_rows * n_columns
    unique_index = random.sample(range(100),n_samples)
    for i in range(len(frames)): 
        props = regionprops(labeled_samples, intensity_image=frames[i])
        # Initializing arrays for all sample properties obtained from regprops.
        row = np.zeros(len(props)).astype(int)
        column = np.zeros(len(props)).astype(int)
        area = np.zeros(len(props))
        radius = np.zeros(len(props))
        perim = np.zeros(len(props))
        intensity = np.zeros(len(props),dtype=np.float64)
        plate = np.zeros(len(props),dtype=np.float64)
        plate_coord = np.zeros(len(props))
       
        c = 0
        for prop in props:
            row[c] = int(prop.centroid[0])
            column[c] = int(prop.centroid[1])
            #print(y[c])
            area[c] = prop.area
            perim[c] = prop.perimeter
            radius[c] = prop.equivalent_diameter/2
            intensity[c] = frames[i][row[c]][column[c]]
            plate[c] = frames[i][row[c]][column[c]+int(radius[c])+3]
            plate_coord[c] = column[c]+radius[c]+3
            c = c + 1
        regprops[i] = pd.DataFrame({'Row': row, 'Column': column,'Plate_temp(cK)':plate,'Radius':radius,'Plate_coord':plate_coord ,'Area': area,
                                'Perim': perim, 'Sample_temp(cK)': intensity,'unique_index':unique_index},dtype=np.float64)
        if len(regprops[i]) != n_samples:
            print('Wrong number of samples are being detected in frame %d' %i)    
        regprops[i].sort_values(['Column','Row'],inplace=True)
    return regprops
   
def sort_regprops(regprops, n_columns, n_rows):
    '''
    Function to sort the regprops to match the order in which the samples
    are pipetted.
    Args:
    regprops(dict): A dictionary of dataframes containing information about the sample.
    n_columns: Number of columns of samples
    n_rows: Number of rows of samples
    '''
    sorted_regprops = {}
    n_samples = n_columns * n_rows
    # After sorting the dataframe according by columns in ascending order.
    sorted_rows = []
    # Sorting the dataframe according to the row coordinate in each column.
    # The samples are pipetted out top to bottom from left to right.
    # The order of the samples in the dataframe should match the order of pipetting.
    for j in range(0,n_columns):
        df = regprops[0][j*n_rows:(j+1)*n_rows].sort_values(['Row'])
        sorted_rows.append(df)
    regprops[0] = pd.concat(sorted_rows)
    # Creating an index to be used for reordering all the dataframes. The unique index is the sum of
    # row and column coordinates.
    reorder_index = regprops[0].unique_index
    for k in range(0,len(regprops)):
        regprops[k].set_index('unique_index',inplace=True)
        sorted_regprops[k] = regprops[k].reindex(reorder_index)
    return sorted_regprops

# Function to obtain temperature of samples and plate temp
def sample_temp(sorted_regprops,frames):
    ''' Function to concatenate all the obtained temperature data
        from the pixel values into lists.
        Args:
        sorted_regprops(dictionary): The dictionary of sorted dataframes containing temperature data.
        frames(array): The array of frames to be processed to obtain temperature data.
        Returns:
        temp(list): Temperature of all the samples in every frame of the video.
        plate_temp(list): Temperature of the plate next to every sample in every
        frame of the video.
    '''
    temp = []
    plate_temp = []
    for j in range(len(sorted_regprops[1])):
        temp_well = []
        plate_well_temp = []
        for i in range(len(frames)):
            temp_well.append(centikelvin_to_celsius(list(sorted_regprops[i]['Sample_temp(cK)'])[j]))
            plate_well_temp.append(centikelvin_to_celsius(list(sorted_regprops[i]['Plate_temp(cK)'])[j]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp,plate_temp

# Function to obtain melting point by extracting the inflection point
def peak_detection(temp_profiles):
    '''Function to determine inflection point in the sample temperature
    profile(melting point)
    Args:
    temp(list): Temperature of all the samples or plate locations in every frame of the video.
    Returns:
    peaks(list): List of two highest peak(inflection points) indices in the
    given temperature profiles.
    infl(list):  List of temperature at inflection points for given temperature profiles.
    '''
    infl = []
    peak_indices = []
    for temp in temp_profiles:
        frames = np.linspace(1,len(temp),len(temp))
        # Fitting a spline to the temperature profile of the samples.
        bspl = BSpline(frames,temp,k=3)
        # Stacking x and y to calculate gradient.
        gradient_array = np.column_stack((frames,bspl(frames)))
        # Calculating gradient
        gradient = np.gradient(gradient_array,axis=0)
        # Calculating derivative
        derivative = gradient[:,1]/gradient[:,0]
        # Finding peaks in the derivative plot.
        peaks, properties = find_peaks(derivative,height=0)
        max_height1 = np.max(properties['peak_heights'])
        # To find the second highest peak
        a = list(properties['peak_heights'])
        a.remove(max_height1)
        max_height2 = np.max(a)
        # Appending the index of the two highest peaks to lists.
        inf_index1 = list(properties['peak_heights']).index(max_height1)
        inf_index2 = list(properties['peak_heights']).index(max_height2)
        # Appending the frame number in which these peaks occur to a list
        peak_indices.append([peaks[inf_index1],peaks[inf_index2]])
        # Appending the temperature at the peaks.
        infl.append([temp[peaks[inf_index1]],temp[peaks[inf_index2]]])
    return peak_indices, infl
    

def inflection_point(s_temp, p_temp, s_peaks, p_peaks):
    '''
    Function to get the inflection point(melting point) for each sample.
    Args:
    s_temp(list): Sample temperature profiles
    p_temp(list): Plate location temperature profiles
    s_peaks(list): List of two highest peak(inflection points) indices in the
    temperature profile of the samples.
    p_peaks(list): List of two highest peak(inflection points) indices in the
    temperature profile of the plate.
    Returns:
    inf_temp(list): List of temperature at inflection points for each sample
    '''
    inf_peak = [] ; inf_temp = []
    for i,peaks in enumerate(s_peaks):
        for peak in peaks:
            # Making sure the peak is present only in the sample temp profile
            if abs(peak - p_peaks[i][0]) >= 3:
                inf_peak.append(peak)
                break
            else:
                pass
    # Appending the temperature of the sample at the inflection point
    for i,temp in enumerate(s_temp):
        inf_temp.append(temp[inf_peak[i]])
    return inf_temp


#### Wrapping functions ######
# Wrapping function to get the inflection point
def inflection_temp(frames ,n_rows, n_columns):
    ''' Function to obtain sample temperature and plate temperature
        in every frame of the video using edge detection.
    Args:
        frames(List): An list containing an array for each frame
        in the video or just a single array in case of an image.
        n_samples: Number of samples in the video
        n_rows: Number of rows of sample
        n_columns: Number of columns of sample
        Returns:
        flip_frames(array) : An array of images which are flipped to correct the
        rotation caused by the IR camera
        regprops(dictionary) : A dictionary of dataframes containing temperature data.
        s_temp(List): A list containing a list a temperatures for each sample
        in every frame of the video 
        plate_temp(List): A list containing a list a temperatures for each plate
        location in every frame of the video.
        inf_temp: A list containing melting point of all the samples obtained by the plot.
        m_df(Dataframe): A dataframe containing row and column coordinates of each sample 
        and its respective inflection point obtained.
    '''

    # Determining the number of samples
    n_samples = n_columns * n_rows
    # Use the function 'flip_frame' to flip the frames horizontally 
    #and vertically to correct for the mirroring during recording
    flip_frames = flip_frame(frames)
    # Use the function 'edge_detection' to detect edges, fill and 
    # label the samples.
    labeled_samples = edge_detection(flip_frames, n_samples)
    # Use the function 'regprop' to determine centroids of all the samples
    regprops = regprop(labeled_samples, flip_frames, n_rows, n_columns)
    # Use the function 'sort_regprops' to sort the dataframes in regprops
    sorted_regprops = sort_regprops(regprops, n_columns, n_rows)
    # Use the function 'sample_temp' to obtain temperature of samples 
    # and plate temp
    s_temp, p_temp = sample_temp(sorted_regprops,flip_frames)
    # Use the function 'sample_peaks' to determine the inflections points
    # and temperatures in sample temperature profiles
    s_peaks, s_infl = peak_detection(s_temp)
    # Use the function 'plate_peaks' to determine the inflections 
    # in plate temperature profiles
    p_peaks, p_infl = peak_detection(p_temp)
    # Use the function 'infection_point' to obtain melting point of samples
    inf_temp = inflection_point(s_temp,p_temp, s_peaks, p_peaks)
    # Creating a dataframe with row and column coordinates of sample centroid and its
    # melting temperature (Inflection point).
    m_df = pd.DataFrame({'Row':regprops[0].Row,'Column':regprops[0].Column,'Melting point':inf_temp})
    return flip_frames, sorted_regprops, s_temp, p_temp, inf_temp, m_df
