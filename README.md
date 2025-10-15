# SAC with W2 Regularization

This repository implements Soft Actor-Critic (SAC) with W2 regularization for continuous control tasks in MuJoCo environments.

## Features

- **Standard SAC**: Implementation of Soft Actor-Critic algorithm
- **W2 Regularization**: Wasserstein-2 distance regularization between current and previous policy
- **Configurable Parameters**: Flexible hyperparameter tuning for W2 regularization
- **TensorBoard Logging**: Comprehensive training metrics and visualization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SAC

# Install dependencies
pip install torch gymnasium[ mujoco ] tensorboard numpy
```

## Usage

### Basic SAC Training
```bash
python train_sac.py --env-name HalfCheetah-v4
```

### SAC with W2 Regularization
```bash
python train_sac.py --env-name HalfCheetah-v4 --w2_reg --w2_reg_weight 0.2 --old_policy_update_freq 10
```

### Humanoid Environment (Most Challenging)
```bash
# Without W2 regularization
python train_sac.py --env-name Humanoid-v4

# With W2 regularization
python train_sac.py --env-name Humanoid-v4 --w2_reg --w2_reg_weight 0.2 --old_policy_update_freq 10
```

## Key Parameters

### W2 Regularization Parameters
- `--w2_reg`: Enable W2 regularization (default: False)
- `--w2_reg_weight`: Weight for W2 regularization term (default: 0.2)
- `--old_policy_update_freq`: Frequency of old policy updates (default: 10)

### Standard SAC Parameters
- `--env-name`: MuJoCo environment name (default: HalfCheetah-v4)
- `--lr`: Learning rate (default: 0.0003)
- `--batch_size`: Batch size (default: 256)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)
- `--alpha`: Temperature parameter (default: 0.2)
- `--automatic_entropy_tuning`: Enable automatic entropy tuning (default: False)

## W2 Regularization Details

The W2 regularization term is added to the policy loss:

```
policy_loss = standard_sac_loss + w2_reg_weight * w2_reg_term
```

Where the W2 regularization term is:
```
w2_reg = mean_diff + std_diff
mean_diff = ||current_mean - prev_mean||²
std_diff = ||current_std - prev_std||²
```

## Results

Training logs are saved in the `runs/` directory and can be visualized using TensorBoard:

```bash
tensorboard --logdir runs/
```

## File Structure

```
SAC/
├── sac.py              # SAC algorithm implementation with W2 regularization
├── train_sac.py        # Training script
├── model.py            # Neural network models
├── replay_memory.py    # Experience replay buffer
├── utils.py            # Utility functions
└── runs/               # TensorBoard logs
```

## Environment Compatibility

This implementation is compatible with Gymnasium v4 environments:
- HalfCheetah-v4
- Humanoid-v4
- Ant-v4
- Walker2d-v4
- And other MuJoCo continuous control environments

## Citation

If you use this code in your research, please cite the original SAC paper and any relevant W2 regularization work.