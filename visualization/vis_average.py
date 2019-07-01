import sys
import numpy as np
import nibabel as nib
import os

from visualization.vis_config import class_limit, gc_layer, roi, task, data_set, label, classification_type


def main():
    """
    Calculates the average Grad-CAM image of multiple runs.
    Converts this final Grad-CAM image to nifti format mapped to MNI152.
    """

    # set info paths
    info_path = f"/path/to/gradcam/info/{roi}/{task}/info/"

    # load subject file
    all_subjects = np.load(f"{info_path}{classification_type}_classified_subjects_{label}.npy")
    k_splits = all_subjects.shape[0]

    save_path = f"/save/path/for/gradcam/{roi}/{task}/{sys.argv[2]}_n{k_splits * class_limit}_{label}_c{gc_layer}_{classification_type}/"

    gc_mean = 0
    gc_var = 0
    gbs = 0

    # loop over all subject splits
    for k in range(0, k_splits):

        # load the gradcam mean file of each split and add this to the gc_mean variable
        gc_img = np.load(f"{save_path}{sys.argv[2]}_run{k}_{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_{analysis}/gradcam_c{gc_layer}_{label}.npy")
        gc_mean += gc_img

        # load the gradcam var file of each split and add this to the gc_var variable
        gc_img = np.load(f"{save_path}{sys.argv[2]}_run{k}_{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_{analysis}/gradcam-VAR_c{gc_layer}_{label}.npy")
        gc_var += gc_img

    # map the mean gradcam to MNI152 format and store nifti file
    nii_gc_mean = to_mni152(gc_mean / k_splits)
    nib.save(nii_gc_mean, f"{save_path}MEAN_{roi}_n{k_splits * class_limit}_{data_set}_gradcam_c{gc_layer}_{label}.nii")

    # map the var gradcam to MNI152 format and store nifti file
    nii_gc_var = to_mni152(gc_var / k_splits)
    nib.save(nii_gc_var, f"{save_path}VAR_{roi}_n{k_splits * class_limit}_{data_set}_gradcam_c{gc_layer}_{label}.nii")


def to_mni152(image):
    """
    This function converts a given numpy matrix to a nifti file mapped to MNI152.

        INPUT:
            image - gets a 3D numpy matrix as input image

        OUTPUT:
            nii_im - returns a 3D nifti file with MNI152 mapping
    """

    # create mask of brain
    template_file = "/path/to/brain/mask/brain_mask_in_template_space.nii.gz"
    template = nib.load(template_file).get_fdata()
    mask = np.zeros(template.shape)
    mask[np.where(template != 0)] = 1

    # get only real data points
    l = np.where(mask != 0)

    # determine the boundaries and corresponding dimensions
    minimum = (min(l[0]), min(l[1]), min(l[2]))
    maximum = (max(l[0]), max(l[1]), max(l[2]))

    # extract data points corresponding to mask
    imagePad = np.zeros(mask.shape)
    imagePad[minimum[0]:(maximum[0] + 1), minimum[1]:(maximum[1] + 1), minimum[2]:(maximum[2] + 1)] = image

    # scale all values between 0 and 1
    imagePad = imagePad / imagePad.max()

    # map to mni152
    mni = nib.load("/path/to/MNI/template/MNI152_T1_1mm_brain.nii.gz")
    nii_im = nib.Nifti1Image(imagePad, affine=mni.affine)

    return nii_im


def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()