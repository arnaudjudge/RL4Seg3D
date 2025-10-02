from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt, animation
from rl4seg3d.utils.cardiac_cycle_utils import estimate_num_cycles

from vital.utils.image.us.measure import EchoMeasure

if __name__ == "__main__":
    GT_PATH = '/data/landmarks_cardinal-icardio/gt/'

    GIF_PATH = './cycle_gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    count = 0
    for p in Path(GT_PATH).rglob('*.nii.gz'):
        print(p)

        gt = nib.load(p).get_fdata()
        img_nifti = nib.load(p.as_posix().replace('gt', 'img'))
        img = img_nifti.get_fdata()

        lv_area = EchoMeasure.structure_area(gt.transpose((2, 0, 1)), labels=1)

        n, p, v = estimate_num_cycles(lv_area, plot=False)

        gt = gt.transpose((2, 1, 0))
        img = img.transpose((2, 1, 0))
        fig, ax = plt.subplots(2, 1)
        im = ax[0].imshow(img[0], animated=True)
        se = ax[0].imshow(gt[0], animated=True, alpha=0.35)
        # plt.figure()
        ax[1].plot(lv_area)
        for p_ in p:
            ax[1].scatter(p, lv_area[p], marker='x', c='red')

        for v_ in v:
            ax[1].scatter(v, lv_area[v], marker='o', c='green')
        line = ax[1].axvline(x=0, color='r', ls='--')
        x_line = np.arange(0, len(gt))
        def update(i):
            im.set_array(img[i])
            se.set_array(gt[i])
            line.set_xdata(x_line[i])
            return im, se, line
        animation_fig = animation.FuncAnimation(fig, update, frames=len(gt), interval=75, blit=True,
                                                repeat_delay=500,)
        plt.show()

        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=10,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        animation_fig.save('test.gif', writer=writer)

        count += 1
        if count > 15:
            break

