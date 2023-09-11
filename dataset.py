import os
import random

import nibabel as nib
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume, normalize_to_uint8


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 2
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=50,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        adc_paths =[]
        dwi_paths = []
        flair_paths = []
        mask_paths = []
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            for filename in sorted(
                filter(lambda f: ".nii.gz" in f, filenames),
                key=lambda x: int(x.split("_")[0][-4:]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "adc" in filename:
                    adc_paths.append(filepath)
                elif "dwi" in filename:
                    dwi_paths.append(filepath)
                elif "flair" in filename:
                    flair_paths.append(filepath)
                elif "msk" in filename:
                    mask_paths.append(filepath)
                           
        adc_paths.sort()
        dwi_paths.sort()
        flair_paths.sort()
        mask_paths.sort()
        for i in range(len(adc_paths)):
            raw_adc_image = nib.load(adc_paths[i]).get_fdata()
            raw_dwi_image = nib.load(dwi_paths[i]).get_fdata()
            raw_mask_image = nib.load(mask_paths[i]).get_fdata()

            transformed_adc = np.transpose(raw_adc_image, (2, 0, 1))
            transformed_dwi = np.transpose(raw_dwi_image, (2, 0, 1))
            transformed_mask = np.transpose(raw_mask_image, (2, 0, 1))
            
            normalized_adc = normalize_to_uint8(transformed_adc)
            normalized_dwi = normalize_to_uint8(transformed_dwi)
            normalized_mask = normalize_to_uint8(transformed_mask)

            combined_volume = np.stack((normalized_adc, normalized_dwi), axis=-1)
            
            case_id = adc_paths[i].split("/")[-3]
            volumes[case_id] = np.array(combined_volume[10:-10])
            masks[case_id] = np.array(normalized_mask[10:-10])


        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor
