from pathlib import Path
import nibabel as nib
from scipy import stats
import skimage.exposure as exp
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    data_path = "<INPUT_PATH>"
    img_folder = "img/"
    output_path = "<OUTPUT_PATH>"
    files = [p for p in Path(data_path + img_folder).rglob('*.nii.gz')]
    print(len(files))
    for p in tqdm(files, total=len(files)):
        print(p)
        img = nib.load(p)
        data = img.get_fdata()

        data = data / 255
        print(data.shape)

        for i in range(data.shape[-1]):
            data[..., i] = exp.equalize_adapthist(data[..., i], clip_limit=0.01)

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_path = output_path + p.relative_to(data_path).as_posix()
        print(out_path)
        print(data.min(), data.max())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, out_path)

        print("\n")
