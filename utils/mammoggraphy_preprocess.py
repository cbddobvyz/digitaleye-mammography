import pydicom
import numpy as np
from skimage.transform import resize
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label, regionprops


def read_dicom_image(dicom_path):
    """
    Reads a dicom file and returns as a image array.

    Parameters
    ----------
    dicom_path : str
        dicom file path

    Returns
    -------
    array(uint16)
         image array

    """
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_image = dicom_data.pixel_array
    return dicom_image


def resize_rescale_image(image_array, new_image_size_H, new_image_size_W):
    """
    Resizes to image to desired dimension and rescales image to [0-255] interval.


    Parameters
    ----------
    image_array : array (uint16)
        image array
    new_image_size : int
        desired dimension for resizing

    Returns
    -------
    array(float64)
         image array

    """
    resized_image = resize(image_array, (new_image_size_H, new_image_size_W),
                           preserve_range=True)
    rescaled_image = (255.0 / resized_image.max() *
                      (resized_image - resized_image.min()))
    return rescaled_image


def invert_image(image_array):
    """
    Inverts a image.

    This function must be used only for FUJI Coorparation brand.

    Parameters
    ----------
    image_array : array(float64)
        image array
 

    Returns
    -------
    array(float64)
         image array

    """
    inverted_image = 255 - image_array
    return inverted_image


def get_binary_masks(image_array):
    """
    Builds binary mask with triange thresholding tecnique.


    Parameters
    ----------
    image_array : array(float64)
        image array
    

    Returns
    -------
    array(Boolean)
         Array for binary mask

    """
    threshold = threshold_triangle(image_array, nbins=128)
    binary_image = image_array > threshold
    return binary_image


def get_image_regions(binary_image_array):
    """
    Finds image regions for binary image


    Parameters
    ----------
    binary_image_array : array(boolean)
        Boolean image array
    
    Returns
    -------
    list
        list of detected regions

    """
    image_label = label(binary_image_array)
    regions = regionprops(image_label)
    return regions


def get_max_region(regions, min_area):
    """
    Detecs a region having maximum area and density.

    This function assumes that breast part of a binary image is a largest and most density object in a binary image.

    Parameters
    ----------
    regions : list
        Coordinate list of detected regions
    min_area : int
        Threshold that determines which regions is ignored

    Returns
    -------
    int
        Selected region index

    """
    object_area = []
    filled_area = []
    for i in regions:
        object_area.append(i.bbox_area)
        filled_area.append(i.filled_area)
    object_counts = len([i for i in object_area if i > min_area])
    if(object_counts > 1):
        top_2_area, top_1_area = np.array(object_area).argsort()[-2:]
        ratio_1_area = filled_area[top_1_area]/object_area[top_1_area]
        ratio_2_area = filled_area[top_2_area]/object_area[top_2_area]
        if(ratio_1_area > ratio_2_area):
            return top_1_area
        else:
            return top_2_area
    else:
        max_object_area_index = np.argmax(object_area)
        return max_object_area_index


def clear_unnecessary_regions(regions, binary_image_array):
    """
    Clears all regions in a binary image except maximum largest and most density object.


    Parameters
    ----------
    regions : list
        Coordinate list of detected regions
    binary_image_array : array(boolean)
        Boolean binary image

    Returns
    -------
    array(Boolean)
        Cleared boolean binary image

    """
    max_region_index = get_max_region(regions, 70000)
    for i in range(len(regions)):
        if(i != max_region_index):
            rows, cols = zip(*regions[i].coords)
            binary_image_array[rows, cols] = 0
    return binary_image_array


def get_cleared_image(image_array, binary_mask):
    """
    clears original image using binary mask.


    Parameters
    ----------
    image_array : array(float6)
        Resized and rescaled image array
    binary_mask : array(boolean)
        Cleared binary mask

    Returns
    -------
    array(float64)
         Array for cleared image

    """
    cleared_image = binary_mask*image_array
    return cleared_image