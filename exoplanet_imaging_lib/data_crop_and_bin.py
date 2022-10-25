import os
from matplotlib.pyplot import axis


import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits

import useful_functions_imaging as useful


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"


    channel = 0

    first_epoch_data = './SPHERE_DC_DATA/raw_data/HIP 27321_DB_H23_2016-11-17_ird_convert_recenter_dc5_128361/'

    datacube_SPHERE = fits.getdata(first_epoch_data + "ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")[channel,:1050,512-100:512+100,512-100:512+100]

    parangs = fits.getdata(first_epoch_data + "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")[:1050]

    datacube_SPHERE_binned = useful.binning_datacube(datacube_SPHERE, sizebin = 10)
    parangs_binned = useful.binning_parangs(parangs, sizebin = 10)

    # print(datacube_SPHERE_binned.shape)
    # print(parangs_binned.shape)

    # we flip the dataset (and therefore inverse the parangs) to obtain
    # the good PA after pyklip reduction
    parangs_binned = -parangs_binned
    # for i in range(datacube_SPHERE_binned.shape[0]):
    #     datacube_SPHERE_binned[i] = np.flip(datacube_SPHERE_binned[i],
    #                                         axis=0)

    # frame_removed = (28, 31, 32, 41)
    # datacube_SPHERE_binned = np.delete(datacube_SPHERE_binned,frame_removed ,
    #                                          0)
    # parangs_binned = np.delete(parangs_binned, frame_removed,
    #                                          0)

    unsaturatedpsf = fits.getdata(first_epoch_data + "ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits")[channel,0,:,:]
    size_datacube = datacube_SPHERE_binned.shape
    centers_images = np.zeros((size_datacube[0],2)) + 100

    fits.writeto(first_epoch_data + "centers_images.fits",centers_images, overwrite=True)
    fits.writeto(first_epoch_data + "datacube_SPHERE_binned_centered.fits",datacube_SPHERE_binned, overwrite=True)
    fits.writeto(first_epoch_data + "parangs_binned.fits",parangs_binned, overwrite=True)
    fits.writeto(first_epoch_data + "unsaturated_psf.fits",unsaturatedpsf, overwrite=True)

    centers_images = fits.getdata(first_epoch_data + "centers_images.fits")
    datacube_SPHERE_binned = fits.getdata(first_epoch_data + "datacube_SPHERE_binned_centered.fits")
    parangs_binned = fits.getdata(first_epoch_data + "parangs_binned.fits")
    unsaturatedpsf = fits.getdata(first_epoch_data + "unsaturated_psf.fits")

    # PCA_components = [30]
    # reduc_pca = useful.simple_pca_already_centered(datacube_SPHERE_binned, centers_images, parangs_binned, PCA_components, "/Users/jmazoyer/Desktop/toto/")

    reduc_pca = useful.simple_pca_already_centered(datacube_SPHERE_binned, 
                                               centers_images, 
                                               parangs_binned,
                                               [30],
                                               "/Users/jmazoyer/Desktop/toto/")


    # Simple rotation
    # simple_rotate = ndimage.rotate(datacube_SPHERE_binned[0] - datacube_SPHERE_binned[-1],-parangs_binned[0], reshape=False)
    # fits.writeto(  "/Users/jmazoyer/Desktop/toto/reduc_simple_rotate.fits",simple_rotate, overwrite=True)

    # # classical_adi
    # datacube_SPHERE_classical_adi_sub = useful.subtract_classical_adi_median(datacube_SPHERE_binned)
    # fits.writeto(  "/Users/jmazoyer/Desktop/toto/datacube_SPHERE_classical_adi_sub.fits",datacube_SPHERE_classical_adi_sub, overwrite=True)


    


    # reduc_classical_adi = useful.derotate_and_mean_classical_adi(datacube_SPHERE_classical_adi_sub,parangs_binned)
    # fits.writeto(  "/Users/jmazoyer/Desktop/toto/reduc_classical_adi.fits",reduc_classical_adi, overwrite=True)

    










