import matplotlib
matplotlib.use('Agg')

import os

import numpy as np
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt, animation
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def put_text_to_image(img, text):
    return cv2.putText(img, "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)


def log_image(logger, img, title, number=0, img_text=None, epoch=0):
    img = img.cpu().numpy()
    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8).squeeze(0)
    if img_text:
        img = put_text_to_image(img, img_text)
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_image(title, img[None,], global_step=number)
    if isinstance(logger, CometLogger):
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        # logger.experiment.log_image(title, img, number)
        logger.experiment.log_figure("{}_{}".format(title, number), fig, step=epoch)
        plt.close()


def log_sequence(logger, img, title, number=0, img_text=None, epoch=0):
    img = img.cpu().numpy()

    # concat images together into bigger image, maximum 4
    img = np.concatenate([img[..., i].transpose((0, 2, 1)) for i in range(min(img.shape[-1], 4))], axis=2)

    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = torch.tensor(put_text_to_image(img, img_text))

    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_image(title, img, global_step=number)
    if isinstance(logger, CometLogger):
        fig = plt.figure()
        if img.shape[0] == 1:
            img = img.squeeze(0)
        else:
            img = img[1]
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        logger.experiment.log_figure("{}_{}".format(title, number), fig, step=epoch)
        plt.close()


def log_video(logger, img, title, background=None, number=0, img_text=None, epoch=0):
    if logger is None:
        return
    if not isinstance(img, np.ndarray):
        img = img.cpu().numpy()
    img = img.transpose((0, 2, 1, 3))
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = put_text_to_image(img, img_text)

    if background is not None:
        if not isinstance(background, np.ndarray):
            background = background.cpu().numpy()
        background = background.transpose((0, 2, 1, 3))
        background = (background.copy() * (255 / max(background.max(), 1))).astype(np.uint8)
        if img_text:
            background = put_text_to_image(background, img_text)

    if isinstance(logger, TensorBoardLogger):
        # should have shape N T C H W
        # should have 3 color channels
        img = np.repeat(img[None,].transpose((0, 4, 1, 2, 3)), repeats=3, axis=2)
        if background is not None:
            background = np.repeat(background[None,].transpose((0, 4, 1, 2, 3)), repeats=3, axis=2)
            for i in range(img.shape[1]):
                img[0, i, ...] = np.asarray(Image.blend(Image.fromarray(background[0, i, ...].transpose(1, 2, 0)),
                                                        Image.fromarray(img[0, i, ...].transpose(1, 2, 0)),
                                                        alpha=0.45)).transpose((2, 0, 1))
        logger.experiment.add_video(title, torch.tensor(img), global_step=number)
    if isinstance(logger, CometLogger):
        img = img.squeeze(0)

        fig, ax = plt.subplots()
        if background is not None:
            background = background.squeeze(0)
            bk = ax.imshow(background[..., 0], animated=True, cmap='gray')
            custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        im = ax.imshow(img[..., 0], animated=True,
                       cmap=custom_cmap if background is not None else 'gray',
                       alpha=0.35 if background is not None else 1.0,
                       interpolation='none')
        ax.axis("off")

        def update(i):
            im.set_array(img[..., i])
            if background is not None:
                bk.set_array(background[..., i])
                return im, bk
            return im,
        animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                repeat_delay=10, )
        animation_fig.save("tmp.gif")
        # Log gif file to Comet
        logger.experiment.log_video("tmp.gif", name="{}_{}".format(title, number), overwrite=True, step=epoch)
        plt.close()
        os.remove("tmp.gif")
