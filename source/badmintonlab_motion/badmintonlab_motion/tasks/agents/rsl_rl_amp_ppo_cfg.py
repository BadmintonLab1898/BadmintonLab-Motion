"""RSL-RL configuration for task-preserving AMP fine-tuning."""

from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


_MOTION_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_T1_MOTION_DIR = str(_MOTION_DATA_DIR / "t1")
_T2_MOTION_DIR = str(_MOTION_DATA_DIR / "t2")


@configclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AMPPPO"

    # Reference clips are packaged with the reference-motion task.  The resolver also
    # accepts BADMINTONLAB_MOTION_DIR as an override for external runs.
    amp_motion_dir: str = _T1_MOTION_DIR
    # Rear-court footwork clips used to condition AMP for rear-court shots.
    # This matches Bwd_step, BwdL_step, and BwdR_step variants in ``t1``.
    amp_motion_pattern: str = "Bwd*_step_*_FINAL.npz"
    amp_transition_steps: int = 4
    amp_reward_dt: float = 0.02
    amp_hidden_dims: list[int] = [256, 128]
    amp_learning_rate: float = 3.0e-5
    amp_style_reward_weight: float = 1.0
    amp_reward_clip: float = 1.0
    amp_policy_samples_per_step: int = 256
    amp_batch_size: int = 512
    amp_discriminator_updates: int = 1
    amp_discriminator_update_interval: int = 16
    amp_replay_rollouts: int = 8
    amp_expert_noise_std: float = 0.02
    amp_gradient_penalty: float = 10.0
    amp_logit_regularization: float = 0.001
    amp_freeze_discriminator: bool = False


@configclass
class AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "badmintonlab_motion"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _task_amp_algorithm(pattern: str, motion_dir: str | None = None) -> AmpPpoAlgorithmCfg:
    """Build a task AMP config without dropping the shared PPO hyperparameters."""
    algorithm_kwargs = {
        "amp_motion_pattern": pattern,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.003,
        "num_learning_epochs": 5,
        "num_mini_batches": 16,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.995,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    }
    if motion_dir is not None:
        algorithm_kwargs["amp_motion_dir"] = motion_dir
    return AmpPpoAlgorithmCfg(**algorithm_kwargs)


@configclass
class T2AmpPPORunnerCfg(AmpPPORunnerCfg):
    experiment_name = "badmintonlab_motion_t2"
    algorithm = AmpPpoAlgorithmCfg(
        amp_motion_dir=_T2_MOTION_DIR,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ClearAmpPPORunnerCfg(AmpPPORunnerCfg):
    experiment_name = "badmintonlab_motion_t1_clear"
    algorithm = _task_amp_algorithm("Bwd*_step_*_FINAL.npz")


@configclass
class DropAmpPPORunnerCfg(AmpPPORunnerCfg):
    experiment_name = "badmintonlab_motion_t1_drop"
    algorithm = _task_amp_algorithm("Bwd*_step_*_FINAL.npz")


@configclass
class LiftAmpPPORunnerCfg(AmpPPORunnerCfg):
    experiment_name = "badmintonlab_motion_t1_lift"
    algorithm = _task_amp_algorithm("Fwd*_step_*_FINAL.npz")


@configclass
class T2ClearAmpPPORunnerCfg(T2AmpPPORunnerCfg):
    """AMP runner for T2 rear-court clear reference motions."""

    experiment_name = "badmintonlab_motion_t2_clear"
    algorithm = _task_amp_algorithm("Bwd*_step_*_FINAL.npz", motion_dir=_T2_MOTION_DIR)


@configclass
class T2SmashAmpPPORunnerCfg(T2AmpPPORunnerCfg):
    """AMP runner for T2 smash reference motions."""

    experiment_name = "badmintonlab_motion_t2_smash"
    algorithm = _task_amp_algorithm("Bwd*_step_*_FINAL.npz", motion_dir=_T2_MOTION_DIR)


@configclass
class T2DefenseAmpPPORunnerCfg(T2AmpPPORunnerCfg):
    """AMP runner for T2 defensive receive reference motions."""

    experiment_name = "badmintonlab_motion_t2_defense"
    algorithm = _task_amp_algorithm("Slide_receive*_success_*.npz", motion_dir=_T2_MOTION_DIR)
