from pathlib import Path

import torch
import numpy as np
import pydicom
import nibabel as nib
import argparse

import torchio as tio
from skimage.exposure import equalize_adapthist


def get_desired_size(current_shape, divisible_by=(32, 32, 4)):
    # get desired closest divisible, bigger shape
    # 32,32,4 as defined by current model architecture
    x = int(np.ceil(current_shape[0] / divisible_by[0]) * divisible_by[0])
    y = int(np.ceil(current_shape[1] / divisible_by[1]) * divisible_by[1])
    z = current_shape[2]
    return x, y, z


def preprocess(img, img_affine, target_spacing, divisible_by=(32, 32, 4)):
    # make sure image is within 0-1, assume already the case or 0-255
    img = img / img.max()

    # equalize adapt hist (same as in training data)
    # frame by frame
    for i in range(img.shape[-1]):
        img[0, ..., i] = equalize_adapthist(img[0, ..., i], clip_limit=0.01)

    # resample to common spacing that model expects
    resample_transform = tio.Resample(target_spacing)
    img = resample_transform(tio.ScalarImage(tensor=img, affine=img_affine))

    # crop or pad to match model stride
    croporpad_transform = tio.CropOrPad(get_desired_size(img.shape[1:], divisible_by))
    img = croporpad_transform(img)
    resampled_affine = img.affine

    # add batch dim and convert to proper typed tensor
    img = img.tensor.type(torch.float).unsqueeze(0)

    return img, resampled_affine


def post_process(outputs, current_affine, target_spacing, target_shape):
    # transforms
    resample_transform = tio.Resample(target_spacing)
    croporpad_transform = tio.CropOrPad(target_shape)

    # apply transforms to each output
    segmentation = tio.LabelMap(tensor=outputs[0][None,].cpu(), affine=current_affine)
    segmentation = croporpad_transform(resample_transform(segmentation))

    merged_reward = tio.ScalarImage(tensor=outputs[1][None,].cpu(), affine=current_affine)
    merged_reward = croporpad_transform(resample_transform(merged_reward))

    anat_reward = tio.ScalarImage(tensor=outputs[2][0][None,].cpu(), affine=current_affine)
    anat_reward = croporpad_transform(resample_transform(anat_reward))

    lm_reward = tio.ScalarImage(tensor=outputs[2][0][None,].cpu(), affine=current_affine)
    lm_reward = croporpad_transform(resample_transform(lm_reward))

    return segmentation.numpy().squeeze(), merged_reward.numpy().squeeze(), anat_reward.numpy().squeeze(), lm_reward.numpy().squeeze()


def main():
    parser = argparse.ArgumentParser(description="Run lightweight TorchScript RL4Seg model on a NIfTI image or folder")
    parser.add_argument("--input", "-i", required=True, help="Path to input NIfTI image or folder")
    parser.add_argument("--output", "-o", required=False, help="Path to save output NIfTI files", default='./OUT/')
    parser.add_argument("--ckpt", "-c", default='rl4seg3d_torchscript_TTA.pt',
                        help="Path to TorchScript checkpoint. Default is ./rl4seg3d_torchscript_TTA.pt")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    #
    # Load dicom
    #
    dcm = pydicom.dcmread(args.input)
    arr = dcm.pixel_array
    if len(arr.shape) > 3:
        arr = arr.mean(-1)

    spacing = None
    if 'PixelSpacing' in dcm:
        spacing = [float(x) for x in dcm.PixelSpacing]
    elif 'ImagerPixelSpacing' in dcm:
        spacing = [float(x) for x in dcm.ImagerPixelSpacing]
    elif 'SequenceOfUltrasoundRegions' in dcm:
        seq = dcm.SequenceOfUltrasoundRegions[0]
        if hasattr(seq, 'PhysicalDeltaX') and hasattr(seq, 'PhysicalDeltaY'):
            spacing = [abs(float(seq.PhysicalDeltaX)) * 10, abs(float(seq.PhysicalDeltaY)) * 10, 1.0]
    else:
        spacing = [1.0, 1.0]  # default fallback if no calibration info

    image = arr.transpose((2, 1, 0))[None,] # make sure it is in C H W T format
    aff = np.diag([spacing[1], spacing[0], 1, 0])
    initial_shape = image.shape
    name = Path(args.input).stem
    print("Initial dicom shape:", arr.shape)

    #
    # Preprocess
    #
    image, intermediate_aff = preprocess(image, aff, target_spacing=(0.37, 0.37, 1))
    image = image.to(device)
    print("Shape presented to model:", image.shape) # B C H W T --> 1 1 H W T

    #
    # Run
    #
    with torch.no_grad():
        # Test-time augmentation always set to true by default
        output = model(image)
        # otherwise
        # output = model(image, tta=False)

    #
    # Post-process
    #
    segmentation, merged_reward, anat_reward, lm_reward = post_process(output, intermediate_aff, spacing, initial_shape[1:])
    print("Output shape:", segmentation.shape)

    #
    # Save to nifti
    #
    Path(args.output).mkdir(exist_ok=True, parents=True)
    nib.save(nib.Nifti1Image(segmentation.astype(np.int32), affine=aff), f"{args.output}/{name}.nii.gz")
    nib.save(nib.Nifti1Image(merged_reward, affine=aff), f"{args.output}/{name}_merged_reward.nii.gz")
    nib.save(nib.Nifti1Image(anat_reward, affine=aff), f"{args.output}/{name}_anat_reward.nii.gz")
    nib.save(nib.Nifti1Image(lm_reward, affine=aff), f"{args.output}/{name}_LM_reward.nii.gz")
    print(f"Saved all outputs to {args.output}")


if __name__ == "__main__":
    main()
