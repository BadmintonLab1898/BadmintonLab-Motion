"""AMP checkpoint persistence for the installed RSL-RL runner API."""
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


class AMPOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        # Same standard fields as OnPolicyRunner.save, plus the full AMP reward
        # model and normalization. Upload only after the complete file is written.
        checkpoint = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            **self.alg.discriminator_checkpoint(),
        }
        if self.alg.rnd:
            checkpoint["rnd_state_dict"] = self.alg.rnd.state_dict()
            checkpoint["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        target = Path(path)
        temporary = target.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)
        if self.logger_type in ("neptune", "wandb") and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location=None, load_discriminator: bool = True):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if load_discriminator and "amp_discriminator_state_dict" in checkpoint:
            self.alg.load_discriminator_checkpoint(checkpoint, load_optimizer=load_optimizer)
            self.alg.set_discriminator_frozen(checkpoint.get("amp_freeze_discriminator", False))
        return infos

    def train_mode(self):
        super().train_mode()
        self.alg.discriminator.train(not self.alg.amp_freeze_discriminator)

    def eval_mode(self):
        super().eval_mode()
        self.alg.discriminator.eval()
