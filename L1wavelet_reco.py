# Testing reconstruction with L1 Wavelet
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
    #lamda = np.array([0.005, 0.007])
    lamda = 0.005
    N_spokes = (256/acc_factors).astype(int)

    SNRs = []
    SSIMs = []

     for Nsp, acc, lamda in zip(N_spokes, acc_factors, lamda):
        coord = sigpy.mri.radial([Nsp,128,2],[256,256],golden=False)
        coord = translate_spokes_to_img_coord(coord, ksp.shape)
        ksp_sampled = sigpy.interpolate(ksp,coord)
        reco = reconstruct_L1WaveletRecon(ksp_sampled, coord) 
        # print(reco.shape)
        reco = scale_by_max(reco, original)
        # print(reco.shape)
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
        print(f"Acceleration factor: {acc},    Number of spokes: {Nsp}, lamda: {lamda} ")
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



