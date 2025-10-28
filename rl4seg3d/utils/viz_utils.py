from matplotlib import pyplot as plt
from matplotlib import animation


def save_to_gif(img, save_path='out.gif', overlay=None):
    """
    Create an animated figure of img with optional overlay, saved to save_path
    :param img: numpy array organized batchwise (T, H, W)
    :param save_path: path to output file
    :param overlay: numpy array organized batchwise (T, H, W)
    :return: None, saved file
    """
    f, ax = plt.subplots()
    im = ax.imshow(img[0].T, animated=True, cmap='gray')
    if overlay is not None:
        ov = ax.imshow(overlay[0].T, animated=True, cmap='gray', alpha=0.4)

    def update(i):
        im.set_array(img[i].T)
        if overlay is not None:
            ov.set_array(overlay[i].T)
            return im, ov,
        return im,

    animation_fig = animation.FuncAnimation(f, update, frames=img.shape[0], interval=100, blit=True, repeat_delay=10)
    animation_fig.save(save_path)
    plt.close()

