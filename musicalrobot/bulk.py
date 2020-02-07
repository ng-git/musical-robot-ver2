import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import musicalrobot

from musicalrobot import edge_detection as ed
from musicalrobot import pixel_analysis as p

def image_crop (tocrop, top, bottom, left, right):
    """
    Function is used within the auto_crop function to  crop using the inputs
    given by the user.

    Parameters:
    -----------
    tocrop : array
        The raw tiff file that is stored in a dictionary and pulled from each
        key using a wrapper. Acts as the base image for the auto_crop

    left : int
        Number of pixels taken off of the left side of the image

    right : int
        Number of pixels taken off of the right side of the image

    top : int
        Number of pixels taken off of the top side of the image

    bottom : int
        Number of pixels taken off of the bottom side of the image

    Returns
    --------
    crop : array
        The array of the tiff file with the requested columns/rows removed

    """
    crop = []
    frames, height, width = tocrop.shape
    for frame in tocrop:
        crop.append(frame[0 + top: height - bottom, 0 + left: width - right])

    return crop

def plot_image (crop, plotname):
    """
    Plots the given cropped image - used as an internal function

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    No returns : will print the plot

    """

    plt.imshow(crop[50])
    plt.colorbar()
    plt.title(plotname)
    plt.show()

    return


def choose_crop (tocrop, plotname):
    """
    Will ask user to choose if the image will be cropped or not. Will skip the
        specific image

    Allowed inputs are y or n. Any other inputs will result in a re-request

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    crop: array
        The array of the tiff file with the requested columns/rows removed. Needs
        to be returned twice to save to the dictionary and then be able to be
        out of the function for use in next functions.

    """
    return crop, crop

def auto_crop (tocrop, plotname):
    """
    Will request an imput from the user to determine how much of the image to
        crop off the sides, top, and bottom. Will produce a cropped image

    Inputs MUST be numerical. the program will fail if not numerical

    Parameters:
    -----------
    tocrop : array
        The raw tiff file that is stored in a dictionary and pulled from each
        key using a wrapper. Acts as the base image for the auto_crop

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    """
    #intro constants
    TotalChange = 1
    left = 0
    right = 0
    top = 0
    bottom = 0

    #User inputs - plot will show between each iteration and will show updates with inputs
    while TotalChange != 0:
        crop = image_cropping(tocrop, top, bottom, left, right)
        plot_image(crop, plotname)

        TotalChange = 0
        change = int(input("Enter the change you want for LEFT "))
        left = left + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for RIGHT "))
        right = right + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for TOP "))
        top = top + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for BOTTOM "))
        bottom = bottom + int(change)
        TotalChange = TotalChange + abs(change)

    return crop


def inflection_points (crop):
    """
    This is a rewrap of the inflection point analysis function using the additive
        rows and columns to find the centriods. All function are the same, but
        the variable names have been changed to match the rest of the bulk
        wrapping functions

    IMPORTANT: This function assumes that the sample is being run on a 96 well
        plate. If this is not correct the number of detected wells will be off

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    Returns:
    --------
    inf_temp : list
        the inflection points of the wells in the video

    """

    img_eq = pa.image_eq(len(crop), crop)
    column_sum, row_sum = pa.pixel_sum(img_eq)

    r_peaks, c_peaks = pa.peak_values(column_sum, row_sum, 12, 8, img_eq)
    sample_location = pa.locations(r_peaks, c_peaks, img_eq)

    temp, plate_temp = pa.pixel_intensity(sample_location, crop, 'Row', 'Column', 'plate_location')

    s_peaks, s_infl = ed.peak_detection(temp)
    p_peaks, p_infl = ed.peak_detection(plate_temp)
    inf_temp = ed.inflection_point(temp, plate_temp, s_peaks, p_peaks)

    return inf_temp

def bulk_crop (cv_file_names):
    """
    Wrapper for all of the bulk cropping functions. Wraps through all of the
        files in the inputed folder, asks for input if the user would like to
        crop the specific function, then asks for inputs for cropping then
        crops the specifed folder in the way requested. Then continues to loop
        through all of the files

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs
        to be created before running the bulk wrapper

    Returns:
    --------
    d_crop : dictionary
        A dictionary of all of the information from the raw tiff files for all of
        the files in the specifed folder

    d_names : dictionary
        A dictionary of all of the file names from all of the files in the specified
        folder. Will correlate with the keys in the d_crop dictionary

    """

    #file input
    for i,file in enumerate(cv_file_names):
        d_files['%s' % i] = ed.input_file('../../MR_Validation/CameraHeight/'+str(file))
        tocrop = d_files['%s' %i]

        # create names
        hold_name = cv_file_names[i]
        d_names['%s' % i] = hold_name[:-5]
        plotname = d_names[str(i)]
        keyname = str(i)

        #auto crop
        d_crop['%s' % i], crop = choose_crop(tocrop, plotname)

    return d_crop, d_names

def bulk_analyze (cv_file_names, d_crop, d_names):
    """
    Wrapper for all of the bulk analysis functions. Wraps through all of the
        files in the inputed folder. Runs analysis functions and then continues to loop
        through all of the files

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs
        to be created before running the bulk wrapper

    d_crop : dictionary
        A dictionary of all of the information from the raw tiff files for all of
        the files in the specifed folder

    d_names : dictionary
        A dictionary of all of the file names from all of the files in the specified
        folder. Will correlate with the keys in the d_crop dictionary

    Returns:
    --------
    d_inftemp : dictionary
        A dictionary of all the inflection temperatures for each file in the
        specifed folder

    all_inf : dataframe
        a dataframe with all of the sample wells and all of the frames. The
        columns will have the file name and the rows will have the well index

    """

    for i, file in enumerate (cv_file_names):
        plotname = d_names[str(i)]
        keyname = str(i)

        crop = d_crop[keyname]
        #save inftemps
        d_inftemp['%s' % i], inf_temp = inflection_points(crop)
        #create df output
        all_inf[plotname] = inf_temp

    return d_inftemp, all_inf

def bulk_process (cv_file_names):
    """
    Wrapper for all of the bulk functions. Runs the bulk cropper followed by the
        bulk analyzer.

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs
        to be created before running the bulk wrapper

    Returns:
    --------
    d_crop : dictionary
        A dictionary of all of the information from the raw tiff files for all of
        the files in the specifed folder

    d_inftemp : dictionary
        A dictionary of all the inflection temperatures for each file in the
        specifed folder

    all_inf : dataframe
        a dataframe with all of the sample wells and all of the frames. The
        columns will have the file name and the rows will have the well index

    """
    d_crop, d_names = bulk_crop(cv_file_names)

    d_inftemp, all_inf = bulk_analyze(cv_file_names, d_crop, d_names)

    return d_crop, d_inftemp, all_inf
