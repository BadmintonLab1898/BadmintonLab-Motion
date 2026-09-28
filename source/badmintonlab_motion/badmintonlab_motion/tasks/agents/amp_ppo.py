"""RSL-RL PPO with an adversarial motion-prior reward."""

from __future__ import annotations

from collections import deque
import hashlib

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from ..amp_motion import AMP_TRANSITION_STEPS, ReferenceMotionDataset


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.ELU()))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.network(transition)


class AMPPPO(PPO):
    """PPO fine-tuning with a least-squares AMP objective.

    The actor and critic remain compatible with ordinary RSL-RL PPO
    checkpoints. The task policy can therefore start from a converged hitting
    checkpoint while the discriminator starts from scratch.
    """

    def __init__(
        self,
        *args,
        amp_motion_dir: str,
        amp_motion_pattern: str = "Bwd*_step_*_FINAL.npz",
        amp_transition_steps: int = 4,
        amp_reward_dt: float = 0.02,
        amp_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        amp_learning_rate: float = 3.0e-5,
        amp_style_reward_weight: float = 0.05,
        amp_reward_clip: float = 1.0,
        amp_policy_samples_per_step: int = 256,
        amp_batch_size: int = 512,
        amp_discriminator_updates: int = 1,
        amp_discriminator_update_interval: int = 4,
        amp_replay_rollouts: int = 8,
        amp_expert_noise_std: float = 0.02,
        amp_gradient_penalty: float = 10.0,
        amp_logit_regularization: float = 0.001,
        amp_freeze_discriminator: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(amp_transition_steps) != AMP_TRANSITION_STEPS:
            raise ValueError(f"Badminton AMP currently supports amp_transition_steps={AMP_TRANSITION_STEPS}")
        self.reference_motions = ReferenceMotionDataset(amp_motion_dir, amp_motion_pattern, self.device)
        self.discriminator = AMPDiscriminator(self.reference_motions.transition_dim, tuple(amp_hidden_dims)).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=amp_learning_rate, weight_decay=1.0e-5
        )

        self.amp_style_reward_weight = float(amp_style_reward_weight)
        self.amp_reward_dt = float(amp_reward_dt)
        if self.amp_reward_dt <= 0.0:
            raise ValueError("amp_reward_dt must be positive")
        self.amp_reward_clip = float(amp_reward_clip)
        self.amp_policy_samples_per_step = int(amp_policy_samples_per_step)
        self.amp_batch_size = int(amp_batch_size)
        self.amp_discriminator_updates = int(amp_discriminator_updates)
        self.amp_discriminator_update_interval = int(amp_discriminator_update_interval)
        self.amp_expert_noise_std = float(amp_expert_noise_std)
        self.amp_gradient_penalty = float(amp_gradient_penalty)
        self.amp_logit_regularization = float(amp_logit_regularization)
        if self.amp_discriminator_update_interval <= 0:
            raise ValueError("amp_discriminator_update_interval must be positive")

        self.amp_update_counter = 0
        self._amp_policy_samples: list[torch.Tensor] = []
        self._amp_replay_transitions: deque[torch.Tensor] = deque(maxlen=int(amp_replay_rollouts))
        self._latest_style_reward = 0.0
        self._latest_policy_probability = 0.0
        self._latest_discriminator_metrics = (0.0, 0.0, 0.0, 0.0)
        self.discriminator_loaded = False
        self.set_discriminator_frozen(amp_freeze_discriminator)

        print(
            "[INFO] AMP discriminator enabled: "
            f"{len(self.reference_motions.files)} clips, "
            f"{self.reference_motions.transitions.shape[0]} expert transitions, "
            f"style_weight={self.amp_style_reward_weight}, "
            f"discriminator_interval={self.amp_discriminator_update_interval}"
        )

    @staticmethod
    def _expert_probability(discriminator_output: torch.Tensor) -> torch.Tensor:
        # LSGAN targets expert=+1 and policy=-1. This mapping is diagnostic only.
        return torch.sigmoid(2.0 * discriminator_output)

    def set_discriminator_frozen(self, frozen: bool) -> None:
        """Freeze only the reward model; PPO actor/critic remain trainable."""
        self.amp_freeze_discriminator = bool(frozen)
        self.discriminator.requires_grad_(not frozen)
        self.discriminator.train(not frozen)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        if frozen:
            self._amp_policy_samples.clear()
            self._amp_replay_transitions.clear()

    def discriminator_checkpoint(self) -> dict:
        return {
            "amp_discriminator_state_dict": self.discriminator.state_dict(),
            "amp_discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "amp_update_counter": self.amp_update_counter,
            "amp_freeze_discriminator": self.amp_freeze_discriminator,
            "amp_reference_signature": {
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in self.reference_motions.files],
                "feature_mean": self.reference_motions.feature_mean.detach().cpu(),
                "feature_std": self.reference_motions.feature_std.detach().cpu(),
            },
        }

    def load_discriminator_checkpoint(self, checkpoint: dict, load_optimizer: bool = True) -> None:
        """Load AMP fields only, never the policy, PPO optimizer or PPO iteration."""
        if "amp_discriminator_state_dict" not in checkpoint:
            raise ValueError("Checkpoint contains no amp_discriminator_state_dict")
        signature = checkpoint.get("amp_reference_signature")
        if signature is None:
            raise ValueError("Discriminator checkpoint lacks reference normalization; use a checkpoint from train_amp_discriminator.py")
        current = self.discriminator_checkpoint()["amp_reference_signature"]
        if signature["files"] != current["files"]:
            raise ValueError("Discriminator reference files differ from the current AMP motion dataset")
        for key in ("feature_mean", "feature_std"):
            saved = signature[key].to(self.device)
            expected = getattr(self.reference_motions, key)
            if (saved.shape != expected.shape or not torch.isfinite(saved).all()
                    or not torch.allclose(saved, expected, atol=1e-6, rtol=1e-5)):
                raise ValueError(f"Discriminator normalization mismatch: {key}")
        weights = checkpoint["amp_discriminator_state_dict"]
        if not all(torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("Non-finite discriminator weights")
        self.discriminator.load_state_dict(weights, strict=True)
        if load_optimizer:
            self.discriminator_optimizer.load_state_dict(checkpoint["amp_discriminator_optimizer_state_dict"])
        for key in ("feature_mean", "feature_std"):
            setattr(self.reference_motions, key, signature[key].to(self.device).detach().clone())
        self.amp_update_counter = int(checkpoint.get("amp_update_counter", 0))
        self.discriminator_loaded = True
        self._amp_policy_samples.clear()
        self._amp_replay_transitions.clear()
        self._latest_discriminator_metrics = (0.0, 0.0, 0.0, 0.0)

    def process_env_step(
        self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        amp_transition = extras.get("amp_transition")
        if amp_transition is None:
            raise KeyError("AMP training requires amp_transition from BadmintonLabMotionEnv")
        amp_transition = amp_transition.to(self.device)
        if amp_transition.shape[-1] != self.reference_motions.transition_dim:
            raise ValueError(
                f"Expected AMP transitions with {self.reference_motions.transition_dim} features, "
                f"got {amp_transition.shape[-1]}"
            )

        normalized = self.reference_motions.normalize(amp_transition)
        discriminator_output = self.discriminator(normalized).squeeze(-1)
        # Standard AMP reward for least-squares discriminator targets (+1/-1).
        style_reward = (1.0 - 0.25 * (discriminator_output - 1.0).square()).clamp(
            min=0.0, max=self.amp_reward_clip
        )
        # Treat style reward as a continuous-time density, matching the
        # environment's time-integrated task rewards.
        rewards_with_style = rewards + self.amp_reward_dt * self.amp_style_reward_weight * style_reward

        sample_count = min(self.amp_policy_samples_per_step, amp_transition.shape[0])
        if sample_count > 0 and not self.amp_freeze_discriminator:
            choice = torch.randperm(amp_transition.shape[0], device=self.device)[:sample_count]
            sample_indices = choice
            # OnPolicyRunner collects under inference_mode. Explicitly leave it
            # so discriminator training receives ordinary tensors during update.
            with torch.inference_mode(False):
                self._amp_policy_samples.append(amp_transition[sample_indices].detach().clone())
        self._latest_style_reward = float(style_reward.mean().item())
        self._latest_policy_probability = float(self._expert_probability(discriminator_output).mean().item())

        log = extras.setdefault("log", {})
        log["Metrics/AMP/style_reward_raw"] = torch.tensor(self._latest_style_reward, device=self.device)
        log["Metrics/AMP/discriminator_frozen"] = torch.tensor(float(self.amp_freeze_discriminator), device=self.device)
        log["Metrics/AMP/style_reward_weight"] = torch.tensor(self.amp_style_reward_weight, device=self.device)
        log["Metrics/AMP/policy_expert_probability"] = torch.tensor(
            self._latest_policy_probability, device=self.device
        )

        super().process_env_step(obs, rewards_with_style, dones, extras)

    def _update_discriminator(self) -> tuple[float, float, float, float]:
        if self.amp_freeze_discriminator:
            # No replay accumulation, backward pass or optimizer step in frozen mode.
            with torch.no_grad():
                expert = self.reference_motions.normalize(self.reference_motions.transitions)
                probability = float(self._expert_probability(self.discriminator(expert)).mean().item())
            return (0.0, probability, self._latest_policy_probability, 0.0)
        if self._amp_policy_samples:
            self._amp_replay_transitions.append(torch.cat(self._amp_policy_samples, dim=0))
            self._amp_policy_samples.clear()

        should_update = self.amp_update_counter % self.amp_discriminator_update_interval == 0
        if not self._amp_replay_transitions or not should_update:
            return self._latest_discriminator_metrics

        policy_transitions = torch.cat(tuple(self._amp_replay_transitions), dim=0)
        mean_loss = 0.0
        mean_expert_probability = 0.0
        mean_policy_probability = 0.0
        mean_gradient_penalty = 0.0
        for _ in range(self.amp_discriminator_updates):
            policy_indices = torch.randint(policy_transitions.shape[0], (self.amp_batch_size,), device=self.device)
            policy_batch = self.reference_motions.normalize(policy_transitions[policy_indices]).detach()
            expert_batch = self.reference_motions.normalize(self.reference_motions.sample(self.amp_batch_size)).detach()
            if self.amp_expert_noise_std > 0.0:
                expert_batch = expert_batch + self.amp_expert_noise_std * torch.randn_like(expert_batch)
            expert_batch.requires_grad_(True)

            policy_output = self.discriminator(policy_batch)
            expert_output = self.discriminator(expert_batch)
            classification_loss = 0.5 * (
                (expert_output - 1.0).square().mean() + (policy_output + 1.0).square().mean()
            )
            expert_gradient = torch.autograd.grad(
                expert_output.sum(), expert_batch, create_graph=True, retain_graph=True
            )[0]
            gradient_penalty = expert_gradient.square().sum(dim=-1).mean()
            logit_regularization = expert_output.square().mean() + policy_output.square().mean()
            loss = (
                classification_loss
                + self.amp_gradient_penalty * gradient_penalty
                + self.amp_logit_regularization * logit_regularization
            )

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.discriminator_optimizer.step()

            mean_loss += float(loss.item())
            mean_expert_probability += float(self._expert_probability(expert_output).mean().item())
            mean_policy_probability += float(self._expert_probability(policy_output).mean().item())
            mean_gradient_penalty += float(gradient_penalty.item())

        count = max(1, self.amp_discriminator_updates)
        self._latest_discriminator_metrics = (
            mean_loss / count,
            mean_expert_probability / count,
            mean_policy_probability / count,
            mean_gradient_penalty / count,
        )
        return self._latest_discriminator_metrics

    def update(self) -> dict[str, float]:
        discriminator_loss, expert_probability, policy_probability, gradient_penalty = (
            self._update_discriminator()
        )
        loss_dict = super().update()
        if not self.amp_freeze_discriminator:
            self.amp_update_counter += 1
        loss_dict.update(
            {
                "amp_discriminator": discriminator_loss,
                "amp_expert_probability": expert_probability,
                "amp_policy_probability": policy_probability,
                "amp_gradient_penalty": gradient_penalty,
                "amp_style_reward": self._latest_style_reward,
            }
        )
        return loss_dict

    def train_mode(self) -> None:
        self.policy.train()
        if self.rnd:
            self.rnd.train()
        self.discriminator.train(not self.amp_freeze_discriminator)

    def eval_mode(self) -> None:
        self.policy.eval()
        if self.rnd:
            self.rnd.eval()
        self.discriminator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict.update(
            {
                "amp_discriminator_state_dict": self.discriminator.state_dict(),
                "amp_discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                "amp_update_counter": self.amp_update_counter,
            }
        )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "amp_discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
            self.amp_update_counter = int(loaded_dict.get("amp_update_counter", 0))
            print(f"[INFO] Restored AMP discriminator at update {self.amp_update_counter}")
        else:
            print("[INFO] PPO checkpoint has no AMP state; initializing a new discriminator")
        return load_iteration
