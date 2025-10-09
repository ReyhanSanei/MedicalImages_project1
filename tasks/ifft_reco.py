# Reconstruction using standard ifft
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sigpy.mri
import sigpy.plot
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils import *

if __name__ == "__main__":

    sitk_t1=sitk.ReadImage('../files/t1.nii.gz')
    t1=sitk.GetArrayFromImage(sitk_t1)
    original = t1[45]
    ksp = sigpy.fft(original)
    
    acc_factors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    N_spokes = (256/acc_factors).astype(int)
    SNRs_ifft = []
    SSIMs_ifft = []
    for Nsp, acc in zip(N_spokes, acc_factors):
        ksp_sampled, coord = radial_sampling(ksp, Nsp)
        ksp_sampled_img = sigpy.gridding(ksp_sampled, coord, np.array([256,256]))
        reco = sigpy.ifft(ksp_sampled_img)
        reco = scale_by_max(reco, original)
        # Metrics
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
        axs[1].set_title(f'Reconstruction, accf={acc}')
        axs[2].set_title('Difference')
        axs[3].set_title('Full SSIM')
        plt.show()
        print(f"Acceleration factor: {acc},    Number of spokes: {Nsp}")
        print(f"PSNR = {psnr},    MSSIM = {ssim}")
        print("============================================================")
        SNRs_ifft.append(psnr)
        SSIMs_ifft.append(ssim)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Acceleration factor')
    ax1.set_ylabel('PSNR', color='blue')
    ax1.plot(acc_factors, SNRs_ifft, color='blue', label='PSNR')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('MSSIM', color='red')
    ax2.plot(acc_factors, SSIMs_ifft, color='red', label='MSSIM')
    ax2.tick_params(axis='y', labelcolor='red')
    # ax2.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
    plt.show()
