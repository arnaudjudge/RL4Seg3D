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


def save_reward_gif(img, rewards, save_path='reward.gif', overlay=None, names=None):
    """
    Create an animated multi-panel figure: image (+ optional segmentation overlay) next to
    one panel per reward map, saved to save_path.
    :param img: numpy array organized batchwise (T, H, W)
    :param rewards: list of reward maps, each a numpy array organized batchwise (T, H, W),
                    already in [0, 1] (post-sigmoid)
    :param save_path: path to output file
    :param overlay: segmentation to overlay on the image panel, batchwise (T, H, W)
    :param names: optional list of panel titles for `rewards` (defaults to reward_0, ...)
    :return: None, saved file
    """
    names = list(names) if names else []
    # pad rather than zip-truncate, so a short/missing name list never drops a panel
    names += [f"reward_{i}" for i in range(len(names), len(rewards))]

    n_panels = 1 + len(rewards)
    f, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    axes = [axes] if n_panels == 1 else list(axes)

    artists = []
    im = axes[0].imshow(img[0].T, animated=True, cmap='gray')
    axes[0].set_title("image", fontsize=9)
    artists.append((im, img))
    if overlay is not None:
        ov = axes[0].imshow(overlay[0].T, animated=True, cmap='gray', alpha=0.4)
        artists.append((ov, overlay))

    # Fixed [0, 1] scale so panels and frames stay comparable (no per-frame autoscaling).
    for ax, r, name in zip(axes[1:], rewards, names):
        rim = ax.imshow(r[0].T, animated=True, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        artists.append((rim, r))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    f.tight_layout()

    def update(i):
        for artist, data in artists:
            artist.set_array(data[i].T)
        return tuple(a for a, _ in artists)

    animation_fig = animation.FuncAnimation(f, update, frames=img.shape[0], interval=100, blit=True,
                                            repeat_delay=10)
    animation_fig.save(save_path)
    plt.close()

