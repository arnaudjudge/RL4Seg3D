from pathlib import Path

import numpy as np
from rl4seg3d.utils.corrector_utils import MorphologicalAndTemporalCorrectionAEApplicator
# from bdicardio.utils.ransac_utils import ransac_sector_extraction
from rl4seg3d.utils.corrector_utils import compare_segmentation_with_ae
from scipy import ndimage
from vital.metrics.camus.anatomical.utils import check_segmentation_validity

import warnings
warnings.filterwarnings("ignore")

class Corrector:
    def correct_batch(self, b_img, b_act):
        """
        Correct a batch of images and their actions with AutoEncoder
        Args:
            b_img: batch of images
            b_act: batch of actions from policy

        Returns:
            Tuple (corrected actions, validity of these corrected versions, difference between original and corrected)
        """
        raise NotImplementedError


class AEMorphoCorrector(Corrector):
    def __init__(self, ae_ckpt_path):
        if ae_ckpt_path and Path(ae_ckpt_path).exists():
            self.ae_corrector = MorphologicalAndTemporalCorrectionAEApplicator(ae_ckpt_path)

    def correct_batch(self, b_img, b_act):
        corrected = np.empty_like(b_img.cpu().numpy())
        corrected_validity = np.empty(len(b_img))
        ae_comp = np.empty(len(b_img))
        for i, act in enumerate(b_act):
            c, _, _ = self.ae_corrector.fix_morphological_and_ae(act.unsqueeze(-1).cpu().numpy())
            corrected[i] = c.transpose((2, 0, 1))

            try:
                corrected_validity[i] = check_segmentation_validity(corrected[i, 0, ...].T, (1.0, 1.0), [0, 1, 2])
            except:
                corrected_validity[i] = False
            ae_comp[i] = compare_segmentation_with_ae(act.unsqueeze(0).cpu().numpy(), corrected[i])
        return corrected, corrected_validity, ae_comp

    def correct_single_seq(self, img, act, spacing):
        corrected, morph_fixed, _ = self.ae_corrector.fix_morphological_and_ae(act.cpu().numpy())

        corrected_validity = True
        for i in range(act.shape[-1]):
            try:
                corrected_validity = corrected_validity and check_segmentation_validity(corrected[..., i].T, spacing, [0, 1, 2])
            except:
                corrected_validity = False
                break
        ae_comp = compare_segmentation_with_ae(act.cpu().numpy(), corrected)
        return corrected, corrected_validity, ae_comp, morph_fixed.transpose((2, 1, 0))

