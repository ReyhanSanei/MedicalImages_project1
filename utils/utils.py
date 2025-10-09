import numpy as np
import sigpy.mri
import sigpy.plot

def translate_spokes_to_img_coord(spokes, im_size=(256,256)):
    x_shift = np.floor(im_size[1]/2) # x_shift determines how columns will be shifted
    y_shift = np.floor(im_size[0]/2) # y_shift determines how rows will be shifted
    new_spokes = []
    for spoke in spokes:
        new_spoke = []
        for point in spoke:
            new_point = [point[0] + x_shift, abs(point[1] - y_shift)]
            new_spoke.append(new_point)
        new_spokes.append(new_spoke)
    return np.array(new_spokes)

def radial_sampling(ksp, n_spokes, n_samples=256):
    coord = sigpy.mri.radial([n_spokes,n_samples,2],ksp.shape,golden=False)
    dcf = ((coord[...,0])**2+(coord[...,1])**2)**0.5
    coord = translate_spokes_to_img_coord(coord, ksp.shape)
    sampled = sigpy.interpolate(ksp,coord)
    return (sampled*dcf), coord # Density-compensated sampled k-space and coordinates

def scale_by_energy(reco, original):
    reco = np.abs(reco)
    scale = np.sqrt(np.sum(original**2)/np.sum(reco**2))
    reco = (reco * scale).astype(np.int16)
    return reco

def scale_by_max(reco, original):
    reco = np.abs(reco)
    reco = (reco/reco.max() * original.max()).astype(np.int16)
    return reco

def reconstruct_radial(ksp_sampled, coord, shape=(256,256)):
    reco = sigpy.nufft_adjoint(ksp_sampled, coord, oshape=shape)
    return reco
