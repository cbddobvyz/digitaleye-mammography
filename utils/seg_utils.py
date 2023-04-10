import itertools
import numpy as np
import scipy as sp


def rescale(img_arr):
    img_arr = 255.0 / (img_arr.max() - img_arr.min()) * (img_arr - img_arr.min())
    img_arr = img_arr.astype('uint8')
    return img_arr


def resize_and_rescale_sequence(sequence):
        
    """
    Reads an image and returns the same image resized and rescaled.
    
    Parameters
    ----------
    sequence : str
        image array
    
    Returns
    -------
    
    array(float32)
        image array
    
    """

    initial_size_x = sequence.shape[0]
    initial_size_y = sequence.shape[1]


    new_size_x = 512
    new_size_y = 512


    delta_x = initial_size_x/new_size_x
    delta_y = initial_size_y/new_size_y


    resized = np.zeros((new_size_x,new_size_y))

    for x, y in itertools.product(range(new_size_x), range(new_size_y)):
        resized[x][y] = sequence[int(x*delta_x)][int(y*delta_y)]

    
    resized=resized.astype('float32')

    rescaled=(255.0 / resized.max() * (resized - resized.min()))
    rescaled=rescaled/255.0

    return rescaled


def output_to_mask(output):
    """
    Converting the output of a model to numpy, rescaling it to 0-1 and rounding the values to show the mask
    
    Parameters
    ----------
    
    output: array
        output of the model
    
    Returns
    -------
    
    numpy array
        image array of the output mask
    """
    output_np=output.data.cpu().numpy()
    output_np=output_np-np.min(output_np)
    output_np=output_np/np.max(output_np)
    output_np=np.around(output_np, decimals=0, out=None)
    
    return output_np


def flood_fill(test_array,h_max=255):
    """
    Maskelerin boşluklarının doldurulma işlemidir.

    Parametreler
    ----------
    test_array : array

    Returns
    -------
    output_array: array
        işlenmiş maske

    """
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array 


