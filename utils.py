import numpy as np
import sigpy
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

def radial_sampling(ksp, n_spokes, n_samples=256, dcomp=True):
    coord = sigpy.mri.radial([n_spokes,n_samples,2],ksp.shape,golden=False)
    if dcomp:
        dcf = ((coord[...,0])**2+(coord[...,1])**2)**0.5
    else:
        dcf = 1
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

def reconstruct_L1WaveletRecon(ksp_sampled, coord , shape=(256,256)):
    dcf = ((coord[...,0])**2+(coord[...,1])**2)**0.5
    sampled_ksp_compensated = ksp_sampled*dcf
    gridded_ksp = sigpy.gridding(sampled_ksp, coord, (256,256))
    gridded_ksp = gridded_ksp[np.newaxis,...]
    mps = sigpy.mri.app.EspiritCalib(gridded_ksp).run()
    reco = sigpy.mri.app.L1WaveletRecon(ksp_sampled, mps, lamda,wave_name='db4', oshape=shape).run()
    return reco  

def get_radial_across(coord):
    '''
    Get a radial trajectory with even number of spokes and combine each pair of
    symmetric-with-respect-to-origin spokes to obtain a radial trajectory whose
    spokes go through the origin (slices).
    '''
    if not(len(coord) % 2 == 0):
        raise ValueError("The number of spokes in the radial trajectory must be even")

    coord_across = []
    ns2 = int(len(coord)/2) # Half the number of spokes in the trajectory
    for sp1, sp2 in zip(coord[:ns2], coord[ns2:]):
        coord_across.append(np.vstack([sp2[::-1], sp1]))
    return np.array(coord_across)

def slice_sampling(ksp, n_angles, n_samples=256, dcomp=False):
    '''
    Sample k-space in a radial trajectory, with spokes going through the origin (slices).
    Return the sampled k_space and the coordinates of the sampling points.

    Note: This function obtains the sampled k-space in the correct shape to reconstruct
    with CT methods.
    '''
    coord = sigpy.mri.radial([n_angles*2, int(n_samples/2), 2], ksp.shape, golden=False)
    coord = get_radial_across(coord)
    if dcomp:
        dcf = ((coord[...,0])**2+(coord[...,1])**2)**0.5
    else:
        dcf = 1
    coord = translate_spokes_to_img_coord(coord, ksp.shape)
    sampled = sigpy.interpolate(ksp, coord)
    return (sampled*dcf), coord


