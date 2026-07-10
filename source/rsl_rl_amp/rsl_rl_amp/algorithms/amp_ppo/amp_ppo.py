from __future__ import annotations

from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


from beyondAMP.motion.motion_dataset import MotionDataset
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl_amp.storage.replay_buffer import ReplayBuffer
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
import torch
import torch.nn as nn
import torch.optim as optim


class AMPPPO(PPO):
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size: float = 100000,
        min_std=None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device="cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Discriminator components
        self.discriminator: AMPDiscriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data: MotionDataset = amp_data
        self.amp_normalizer = amp_normalizer
        self.min_std = min_std

        # Optimizer for policy and discriminator.
        params = [
            {"params": self.actor.parameters(), "name": "actor"},
            {"params": self.critic.parameters(), "name": "critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.amp_linear.parameters(),
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]  # NOTE: ADDED
        self.optimizer = optim.Adam(params, lr=learning_rate)  # NOTE: ADDED
        self.transition = RolloutStorage.Transition()

    def act(self, obs, amp_obs=None):  # NOTE: ADDED
        actions = super().act(obs=obs)
        self.amp_transition.observations = amp_obs
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_obs=None,
    ) -> None:
        super().process_env_step(obs=obs, rewards=rewards, dones=dones, extras=extras)

        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.amp_transition.clear()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(
            generator,
            amp_policy_generator,
            amp_expert_generator,
        ):
            # (
            #     obs_batch,
            #     critic_obs_batch,  # NOTE: ADDED
            #     actions_batch,
            #     target_values_batch,
            #     advantages_batch,
            #     returns_batch,
            #     old_actions_log_prob_batch,
            #     old_mu_batch,
            #     old_sigma_batch,
            #     hid_states_batch,
            #     masks_batch,
            # ) = sample

            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(
                    num_aug, 1
                )
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            aug_obs_batch = obs_batch.detach()  # NOTE: ADDED

            self.actor(
                aug_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )

            actions_log_prob_batch = self.actor.get_actions_log_prob(
                actions_batch
            )  # NOTE: ADDED

            aug_critic_obs_batch = critic_obs_batch.detach()  # NOTE: ADDED

            value_batch = self.actor_critic.evaluate(
                aug_critic_obs_batch,  # NOTE: ADDED
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(
                        policy_state, self.device
                    )
                    policy_next_state = self.amp_normalizer.normalize_torch(
                        policy_next_state, self.device
                    )
                    expert_state = self.amp_normalizer.normalize_torch(
                        expert_state, self.device
                    )
                    expert_next_state = self.amp_normalizer.normalize_torch(
                        expert_next_state, self.device
                    )
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_next_state], dim=-1)
            )
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_next_state], dim=-1)
            )
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                *sample_amp_expert, lambda_=10
            )

            # Compute total loss.
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(
                    min=self.min_std
                )

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
        )
