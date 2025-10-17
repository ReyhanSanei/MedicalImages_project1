import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sigpy.mri
import sigpy.plot
from skimage.transform import iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

sitk_t1=sitk.ReadImage('../files/t1.nii.gz')
t1=sitk.GetArrayFromImage(sitk_t1)

original = t1[45]
ksp = sigpy.fft(original)

N_angles = [180, 120, 70, 60, 50, 40, 35, 30, 27, 25, 20] # Equivalent to number of projections in the sinogram
N_samples = 258 # Total samples in the slice, not the "spoke"
acc_factors = []
SNRs = []
SSIMs = []

for n in N_angles:
    acc_factors.append(256/n)
    ksp_sampled, coord = slice_sampling(ksp, n, N_samples, dcomp=False)
    sino = np.array([sigpy.ifft(ksp_slice, oshape=(256,), center=True) for ksp_slice in ksp_sampled])
    angles = np.linspace(0,180,n)
    reco = iradon(np.real(sino.T), theta=angles, filter_name='ramp')
    reco = np.flip(reco)
    reco = scale_by_max(reco, original)
    difference = (original - reco)/(original.max()-original.min())
    psnr = peak_signal_noise_ratio(original, reco)
    ssim, S = structural_similarity(original, reco, full=True, data_range=(original.max() - original.min()))
    # Plot
    fig, axs = plt.subplots(1, 4)
    fig.set_figwidth(18)
    plt.axis('off')
    figs = []
    figs.append(axs[0].imshow(original, cmap='gray',))
    figs.append(axs[1].imshow(reco, cmap='gray'))
    figs.append(axs[2].imshow(difference, cmap='gray'))
    figs.append(axs[3].imshow(S, cmap='gray'))
    for f, a in zip(figs,axs):
        fig.colorbar(f, ax=a, shrink=0.5)
    axs[0].set_title('Original')
    axs[1].set_title(f'Reconstruction, accf={256/n:.2g}')
    axs[2].set_title('Difference')
    axs[3].set_title('Full SSIM')
    plt.show()
    print(f"Acceleration factor: {256/n},    Number of k-space slices: {n}")
    print(f'FBP rms reconstruction error: {np.sqrt(np.mean(difference**2)):.3g}')
    print(f"PSNR = {psnr},    MSSIM = {ssim}")
    print("============================================================")
    SNRs.append(psnr)
    SSIMs.append(ssim)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Acceleration factor')
ax1.set_ylabel('PSNR', color='blue')
ax1.plot(acc_factors, SNRs, color='blue', label='PSNR')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.set_ylabel('MSSIM', color='red')
ax2.plot(acc_factors, SSIMs, color='red', label='MSSIM')
ax2.tick_params(axis='y', labelcolor='red')
# ax2.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
plt.show()
