import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        # 재현성을 위한 시드 설정 (POGO 스타일)
        seed = getattr(args, 'seed', None)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # W2 regularization parameters
        self.w2_reg_weight = getattr(args, 'w2_reg_weight', 0.0)
        self.old_policy_update_freq = getattr(args, 'old_policy_update_freq', 10)
        
        # Multi-step actor parameters
        self.num_actors = getattr(args, 'num_actors', 1)
        self.w2_weights = getattr(args, 'w2_weights', [0.0])
        if isinstance(self.w2_weights, (int, float)):
            # Single value: use for all actors except first
            self.w2_weights = [0.0] + [self.w2_weights] * (self.num_actors - 1)
        assert len(self.w2_weights) == self.num_actors, f"w2_weights length ({len(self.w2_weights)}) must match num_actors ({self.num_actors})"
        
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            # Multiple policies for multi-step actor update
            self.policies = []
            self.policy_targets = []
            self.policy_optimizers = []
            
            for i in range(self.num_actors):
                policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
                policy_target = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
                hard_update(policy_target, policy)
                policy_optim = Adam(policy.parameters(), lr=args.lr)
                
                self.policies.append(policy)
                self.policy_targets.append(policy_target)
                self.policy_optimizers.append(policy_optim)
            
            # For backward compatibility, keep self.policy as first policy
            self.policy = self.policies[0]
            self.policy_optim = self.policy_optimizers[0]
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.num_actors = 1
            self.policies = [self.policy]
            self.policy_targets = [copy.deepcopy(self.policy)]
            self.policy_optimizers = [self.policy_optim]

    def select_action(self, state, evaluate=False, actor_idx=None, seed=None):
        """
        Select action from policy.
        
        Args:
            state: state vector
            evaluate: if True, use deterministic action (mean)
            actor_idx: which actor to use (None → last actor)
            seed: 재현성을 위한 시드 (None이면 랜덤)
        """
        if actor_idx is None:
            actor_idx = self.num_actors - 1  # default: use last actor
        
        policy = self.policies[actor_idx]
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # 재현성을 위해 시드 설정
        if seed is not None and not evaluate:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None
        
        if evaluate is False:
            action, _, _ = policy.sample(state, generator=generator)
        else:
            _, _, action = policy.sample(state, generator=generator)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory (재현성을 위해 시드 고정)
        seed_base = updates * 1000 if hasattr(memory, 'seed') else None
        buffer_seed = seed_base if seed_base is not None else None
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size, seed=buffer_seed)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # ------------------------
        # Critic training (uses first policy for target)
        # ------------------------
        # 재현성을 위한 시드 설정
        seed_base = updates * 1000 if hasattr(memory, 'seed') else None
        
        with torch.no_grad():
            # Use first target policy for TD target (재현성을 위해 시드 고정)
            target_seed = seed_base + 1 if seed_base is not None else None
            target_generator = None
            if target_seed is not None:
                target_generator = torch.Generator(device=self.device).manual_seed(target_seed)
            first_policy_target = self.policy_targets[0]
            next_state_action, next_state_log_pi, _ = first_policy_target.sample(next_state_batch, generator=target_generator)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # ------------------------
        # Multi-step Actor training
        # ------------------------
        # 재현성을 위한 시드 설정 (이미 위에서 계산됨)
        
        policy_losses = []
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha)
        
        # 메트릭 저장용 딕셔너리 (POGO 스타일)
        metrics = {}
        
        # All policies in train mode
        for policy in self.policies:
            policy.train()
        
        for i in range(self.num_actors):
            policy_i = self.policies[i]
            w2_weight_i = self.w2_weights[i]
            
            # Sample action from policy_i (재현성을 위해 시드 고정)
            policy_seed = seed_base + 10 + i if seed_base is not None else None
            generator = None
            if policy_seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(policy_seed)
            pi_i, log_pi_i, _ = policy_i.sample(state_batch, generator=generator)
            qf1_pi, qf2_pi = self.critic(state_batch, pi_i)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            # Q 값 평균 저장
            q_mean = min_qf_pi.mean().item()
            metrics[f'Q_{i}_mean'] = q_mean
            
            # Base SAC loss: (alpha * log_pi) - min_qf_pi
            base_policy_loss = ((self.alpha * log_pi_i) - min_qf_pi).mean()
            
            # W2 distance
            if i == 0:
                # First policy: no W2 regularization
                w2_i = torch.tensor(0.0).to(self.device)
            else:
                base_policy_denom = base_policy_loss.abs().mean().detach().clamp_min(1e-6)
                base_policy_loss = base_policy_loss / base_policy_denom
                # Subsequent policies: W2 distance to previous policy (closed form for Gaussian)
                ref_policy = self.policies[i - 1]
                ref_policy.eval()
                
                # Get mean and std from both policies
                with torch.no_grad():
                    ref_mean, ref_log_std = ref_policy.forward(state_batch)
                    ref_std = ref_log_std.exp()
                
                current_mean, current_log_std = policy_i.forward(state_batch)
                current_std = current_log_std.exp()
                
                # W2 distance closed form: (dm)^2 + (dsigma)^2
                mean_diff = (current_mean - ref_mean).pow(2).sum(-1).mean()
                std_diff = (current_std - ref_std).pow(2).sum(-1).mean()
                w2_i = mean_diff + std_diff
                
                ref_policy.train()
                base_policy_loss = base_policy_loss + w2_weight_i * w2_i
            
            # W2 distance 저장
            metrics[f'w2_{i}_distance'] = w2_i.item()
            
            policy_loss_i = base_policy_loss
            policy_losses.append(policy_loss_i.item())
            
            # Actor loss 저장
            metrics[f'actor_{i}_loss'] = policy_loss_i.item()
            
            # Q gradient norm 계산 (POGO 스타일)
            opt = self.policy_optimizers[i]
            opt.zero_grad()
            
            # Q gradient만 계산하기 위해 Q 부분만 backward
            q_grad_loss = -min_qf_pi.mean()
            q_grad_loss.backward(retain_graph=True)
            q_grad_norm = 0.0
            for param in policy_i.parameters():
                if param.grad is not None:
                    q_grad_norm += param.grad.data.norm(2).item() ** 2
            q_grad_norm = q_grad_norm ** 0.5
            metrics[f'Q_{i}_grad_norm'] = q_grad_norm
            
            # Clear gradients and do full backward
            opt.zero_grad()
            policy_loss_i.backward()
            opt.step()
            
            # Alpha update (only for first policy)
            if i == 0 and self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi_i + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()
        
        # Unfreeze critic after all policy updates
        for p in self.critic.parameters():
            p.requires_grad_(True)
        
        # Soft-update targets
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            for policy, policy_target in zip(self.policies, self.policy_targets):
                soft_update(policy_target, policy, self.tau)
        
        # After update, copy current policy to old_policy if W2 reg enabled (with frequency control)
        if self.w2_reg_weight > 0 and hasattr(self, 'old_policy') and updates % self.old_policy_update_freq == 0:
            self.old_policy.load_state_dict(self.policies[0].state_dict())
        
        # Return unified metrics dictionary
        policy_loss = policy_losses[0] if len(policy_losses) > 0 else 0.0
        
        # 모든 메트릭을 하나의 딕셔너리로 통합
        metrics['critic_loss'] = qf1_loss.item() + qf2_loss.item()
        metrics['critic_1_loss'] = qf1_loss.item()
        metrics['critic_2_loss'] = qf2_loss.item()
        metrics['policy_loss'] = policy_loss
        metrics['entropy_loss'] = alpha_loss.item()
        metrics['alpha'] = alpha_tlogs.item()
        
        return metrics

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        
        checkpoint_dict = {
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
        }
        
        # Save all policies
        for i in range(self.num_actors):
            checkpoint_dict[f'policy_{i}_state_dict'] = self.policies[i].state_dict()
            checkpoint_dict[f'policy_{i}_target_state_dict'] = self.policy_targets[i].state_dict()
            checkpoint_dict[f'policy_{i}_optimizer_state_dict'] = self.policy_optimizers[i].state_dict()
        
        # For backward compatibility
        checkpoint_dict['policy_state_dict'] = self.policies[0].state_dict()
        checkpoint_dict['policy_optimizer_state_dict'] = self.policy_optimizers[0].state_dict()
        
        # Save alpha if exists
        if self.automatic_entropy_tuning:
            checkpoint_dict['log_alpha'] = self.log_alpha
            checkpoint_dict['alpha_optimizer_state_dict'] = self.alpha_optim.state_dict()
        
        torch.save(checkpoint_dict, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            
            # Load critic
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Load all policies
            for i in range(self.num_actors):
                if f'policy_{i}_state_dict' in checkpoint:
                    self.policies[i].load_state_dict(checkpoint[f'policy_{i}_state_dict'])
                    self.policy_targets[i].load_state_dict(checkpoint[f'policy_{i}_target_state_dict'])
                    self.policy_optimizers[i].load_state_dict(checkpoint[f'policy_{i}_optimizer_state_dict'])
                elif 'policy_state_dict' in checkpoint and i == 0:
                    # Backward compatibility: load first policy from old format
                    self.policies[0].load_state_dict(checkpoint['policy_state_dict'])
                    if 'policy_optimizer_state_dict' in checkpoint:
                        self.policy_optimizers[0].load_state_dict(checkpoint['policy_optimizer_state_dict'])
                    hard_update(self.policy_targets[0], self.policies[0])
            
            # Load alpha if exists
            if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
                self.log_alpha.data = checkpoint['log_alpha']
                if 'alpha_optimizer_state_dict' in checkpoint:
                    self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                self.alpha = self.log_alpha.exp()

            if evaluate:
                for policy in self.policies:
                    policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                for policy in self.policies:
                    policy.train()
                self.critic.train()
                self.critic_target.train()

