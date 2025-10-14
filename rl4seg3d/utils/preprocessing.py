import skimage.exposure as exp


def rescale(img):
    # rescale to 0 - 1, as expected by rl4seg3d
    img /= img.max() if img.max() > 0 else 1.0  # normalize safely
    return img

def apply_eq_adapthist(img):
    for i in range(img.shape[-1]): # assume temporal is last dim
        img[0, ..., i] = exp.equalize_adapthist(img[0, ..., i], clip_limit=0.01)
    return img