import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import marching_cubes
from typing import Optional, Union
#def labels_get_indices(labels : np.ndarray, exclude_background : bool, add_unravelled_indices : bool) -> dict:
#    unique_labels, unique_inverse, unique_counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)
#    all_indices = np.argsort(unique_inverse)
#    if(exclude_background):
#        if(unique_labels[0] == 0):
#            num_zero_labels = unique_counts[0]
#            unique_labels = unique_labels[1:]
#            unique_counts = unique_counts[1:]
#            all_indices = all_indices[num_zero_labels:]
#    indices_dict = {}
#    ref = 0
#    for label, count in zip(unique_labels, unique_counts):
#        indices_flat = all_indices[ref:ref+count]
#        indices_dict[label] = {'indices_flattened': indices_flat, 'count' : count}
#        if(add_unravelled_indices):
#            unravelled_indices = np.unravel_index(indices_flat, labels.shape)
#            indices_dict[label]['indices'] = unravelled_indices
#        ref += count
#    return indices_dict

def labels_statistics(labels : np.ndarray, intensity : np.ndarray, spacing : Union[np.ndarray , float, None] = None , add_meshes : bool = False, target_labels : np.ndarray = None) -> tuple:
    objects = find_objects(labels)
    indices = np.indices(labels.shape)
    num_dimensions = labels.ndim
    if(spacing is None):
        spacing_array = np.ones(num_dimensions)
    elif(np.isscalar(spacing)):
        spacing_array = np.ones(num_dimensions)*spacing
    else:
        spacing_array = np.asarray(spacing)
        if(spacing_array.size != num_dimensions):
            raise ValueError("Spacing must be a scalar or an array with the same number of elements as the number of dimensions of the labels array.")
    
    statistics_dict = {}
    for index, slices in enumerate(objects):
        if(slices is None):
            continue
        label = index + 1
        mask = labels[slices] == label

        sliced_intensity = intensity[slices]
        masked_intensity = sliced_intensity*mask[...,None]
        sliced_indices = indices[(slice(None),) + slices]
        masked_indices = sliced_indices*mask[None,...]
        moment0 = np.sum(mask)
        wmoment0 = np.sum(masked_intensity, axis=tuple(range(num_dimensions)))
        moment1 = np.sum(masked_indices, axis=tuple(range(1,num_dimensions+1)))/moment0
        moment2 = np.sum(masked_indices[:,None,...]*masked_indices[None,:,...], axis=tuple(range(2,num_dimensions+2)))/moment0
        wmoment1 = np.sum(masked_indices[...,None]*masked_intensity[None,...], axis=tuple(range(1,num_dimensions+1)))/wmoment0[None,:]
        wmoment2 = np.sum(masked_indices[:,None,...,None]*masked_indices[None,:,...,None]*masked_intensity[None,None,...], axis=tuple(range(2,num_dimensions+2)))/wmoment0[None,None,:]
        
        moment1 *= spacing_array
        moment2 *= spacing_array[:,None]*spacing_array[None,:]
        wmoment1 *= spacing_array[:,None]
        wmoment2 *= spacing_array[:,None,None]*spacing_array[None,:,None]
        moment2 = moment2 - moment1[:,None]*moment1[None,:]
        wmoment2 = wmoment2 - wmoment1[:,None,:]*wmoment1[None,:,:] 
        if(target_labels is not None):
            sliced_target_labels = target_labels[slices]*mask
            target_labels_unique, target_labels_counts = np.unique(sliced_target_labels, return_counts=True)
            if(target_labels_unique[0] == 0):
                target_labels_unique = target_labels_unique[1:]
                target_labels_counts = target_labels_counts[1:]
        if(add_meshes and num_dimensions == 3):
            inflated_slices = []
            for dim, slc in enumerate(slices):
                start = slc.start - 1 if slc.start > 0 else slc.start
                stop = slc.stop + 1 if slc.stop < labels.shape[dim] else slc.stop
                inflated_slices.append(slice(start, stop, None))
            inflated_slices = tuple(inflated_slices)
            translation_vector = np.array([s.start for s in slices])*spacing_array
            inflated_mask = labels[inflated_slices] == label
            try:
                verts, faces, normals, values = marching_cubes(inflated_mask.astype(np.float32), level=0.5, spacing=spacing_array)
                verts += translation_vector[None,:]
                mesh = {'verts': verts, 'faces': faces, 'normals': normals, 'values': values}
            except Exception as e:
                mesh = None
        result_dict = {'moment0': moment0, 'moment1': moment1, 'moment2': moment2, 'wmoment0': wmoment0, 'wmoment1': wmoment1, 'wmoment2': wmoment2}
        if(add_meshes and num_dimensions == 3):
            result_dict['mesh'] = mesh
        statistics_dict[label] = result_dict
    return statistics_dict

def extract_nematics_3d(covariance_matrices : np.ndarray) -> tuple:
    if(covariance_matrices.shape[-1] != covariance_matrices.shape[-2]):
        raise ValueError("The last two dimensions of the covariance_matrices array must be equal.")
    if(covariance_matrices.shape[-1] != 3):
        raise ValueError("The covariance_matrices must be 3x3 matrices.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices)
    q_values = 0.5*np.log(eigenvalues)
    q_values -= np.sum(q_values, axis=-1, keepdims=True)/3.0
    prolate_mask = q_values[...,-1] > np.abs(q_values[...,0])
    oblate_mask = ~prolate_mask
    S = np.zeros_like(q_values.shape[:-1])
    S[prolate_mask] = q_values[prolate_mask,-1]
    S[oblate_mask] = q_values[oblate_mask,0]
    T = np.zeros_like(S)
    T[prolate_mask] =  q_values[prolate_mask,1] - 0.5*q_values[prolate_mask,2]
    T[oblate_mask] = q_values[oblate_mask,1] - 0.5*q_values[oblate_mask,0]
    return S, T