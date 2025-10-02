import torch

"""
Reward functions must each have pred, img, gt as input parameters
"""


class Reward:
    @torch.no_grad()
    def __call__(self, pred, imgs, gt, *args, **kwargs):
        raise NotImplementedError

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        pass

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt):
        return self(pred, imgs, gt)

    def get_reward_index(self, reward_name):
        print("No named rewards available, returned index 0")
        return 0