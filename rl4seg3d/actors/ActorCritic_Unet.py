from rl4seg3d.actors.Actors import Actor


class ActorCriticUnetCritic(Actor):
    """
        ActorCritic actor class, evaluates actor and value function approximate
        Value function is represented as a grid/matrix, unet is value function approximator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, imgs, actions):
        """
            Evaluate images with both actor and critic
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs, value function estimate from critic
        """
        logits, distribution, old_distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        if old_distribution:
            old_log_probs = old_distribution.log_prob(actions).detach()
        else:
            old_log_probs = log_probs.detach()

        sampled_actions = distribution.sample()
        entropy = distribution.entropy()

        if hasattr(self.critic.net, "deep_supervision") and self.critic.net.deep_supervision and self.critic.net.training:
            v = self.critic(imgs)
            v = [v_.squeeze(1) for v_ in v]
        else:
            v = self.critic(imgs).squeeze(1)

        return sampled_actions, logits, log_probs, entropy, v, old_log_probs


