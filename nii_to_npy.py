import glob
import operator
import os
import nibabel as nib
import numpy as np
from skimage.transform import downscale_local_mean


def main():
    """
    This script converts MRI scans in nifti format from a specified data path to numpy arrays (.npy). This is the
    required format to be used in the data generators of the Keras deep learning model created for the MRI-based
    classification of Alzheimer's Disease.

    In the settings can be specified which data set and data type should be converted. Also can be specified whether
    images should be converted to whole brain and/or down sampled images. A mask is applied to all images so that
    the area outside the brain is set to zero. Furthermore the images are cropped to this mask.

    This script should be run to create data before a model can be trained on this data with the 'main.py' script.
    """

    # SETTINGS
    dataset = "ADNI"            # ADNI / PSI
    datatype = "GM"             # T1 / T1m / GM
    WB = True                   # if True: whole brain images
    downsampling = False        # if True: down sampled by factor 4
    testing = False             # if True: only applied to 4 subjects

    # create output dir
    if WB:
        save_path_WB = f"/path/to/save/WB/data/in/data_{dataset}_{datatype}_WB/"
        create_data_directory(save_path_WB)
    if downsampling:
        save_path_f4 = f"/path/to/save/downsampled/data/in/data_{dataset}_{datatype}_f4/"
        create_data_directory(save_path_f4)

    # set path to data
    if dataset == "ADNI":
        if datatype == "GM":
            data_path = "/mnt/cartesius/ADNI/Template_space/*_bl/Brain_image_in_template_space/gmModulatedJacobian.nii.gz"
        elif datatype == "T1":
            data_path = "/mnt/cartesius/ADNI/Template_space/*_bl/Brain_image_in_MNI_space/result.nii.gz"
        elif datatype == "T1m":
            data_path = "/mnt/cartesius/ADNI/Template_space/*_bl/Brain_image_in_template_space/T1wModulatedJacobian.nii.gz"
    elif dataset == "PSI":
        if datatype == "GM":
            data_path = "/mnt/cartesius/Parelsnoer/Template_space/PSI_*/Brain_image_in_template_space/gmModulatedJacobian.nii.gz"
        elif datatype == "T1":
            data_path = "/mnt/cartesius/Parelsnoer/Template_space/PSI_*/Brain_image_in_MNI_space/result.nii.gz"
        elif datatype == "T1m":
            data_path = "/mnt/cartesius/Parelsnoer/Template_space/PSI_*/Brain_image_in_template_space/T1wModulatedJacobian.nii.gz"

    # create mask of brain
    template_file = "/path/to/brain/mask/brain_mask_in_template_space.nii.gz"
    template = nib.load(template_file).get_fdata()
    mask = np.zeros(template.shape)
    mask[np.where(template != 0)] = 1

    cnt = 0

    # loop over all files in data path
    for filename in glob.glob(data_path):
        cnt += 1

        # get subject ID
        if dataset == "ADNI":
            subject = filename[35:45]
        elif dataset == "PSI":
            subject = filename[41:50]

        # for testing break early
        if testing:
            if cnt % 5 == 0:
                break
            print(subject)

        # print progress
        if cnt % 50 == 0:
            print('\n----- Working on subject number ' + str(cnt) + ' -----')

        # load .nii data as 3d numpy array
        image = nib.load(filename).get_fdata()

        # normalization for signal intensity
        if datatype == "T1":
            image = normalize(operator.mul(image, mask))

        # apply mask + crop image
        masked = apply_mask(image, mask)

        # save as .npy
        if downsampling:
            # down sample by factor 4 based on local mean
            downsampled = downscale_local_mean(masked, (4, 4, 4))
            np.save(save_path_f4 + subject + '.npy', downsampled)
        if WB:
            np.save(save_path_WB + subject + '.npy', masked)


def normalize(image):
    """
    Normalize image for signal intensity by subtracting mean and dividing by standard deviation.
    Only uses non-zero values within the brain mask, for normalization.

        INPUT:
            image - image to be normalized

        OUTPUT:
            image - normalized image
    """

    mean = image[np.nonzero(image)].mean()
    std = image[np.nonzero(image)].std()
    image = (image - mean) / std

    return image


def apply_mask(image, mask):
    """
    Apply the mask to an image and crop this image based on the size of the mask

        INPUT:
            image - 3D numpy matrix of an image
            mask - 3D mask to be applied (zeros = non-data)

        OUTPUT:
            imageCropped - masked + cropped image
    """

    # apply mask to image
    r = operator.mul(image, mask)

    # get only real data points
    l = np.where(mask != 0)

    # determine the boundaries and corresponding dimensions
    minimum = (min(l[0]), min(l[1]), min(l[2]))
    maximum = (max(l[0]), max(l[1]), max(l[2]))

    x = maximum[0] - minimum[0] + 1
    y = maximum[1] - minimum[1] + 1
    z = maximum[2] - minimum[2] + 1

    imageCropped = np.zeros((x, y, z))

    # extract data points corresponding to mask
    imageCropped[:x, :y, :z] = r[minimum[0]:(maximum[0] + 1), minimum[1]:(maximum[1] + 1), minimum[2]:(maximum[2] + 1)]

    return imageCropped


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"\nWARNING: PATH ALREADY EXISTS\t{path}\n")


if __name__ == '__main__':
    main()

