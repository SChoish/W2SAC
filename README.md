# POSAC: POGO on Soft Actor-Critic in Online RL

POSAC은 POGO (Policy Optimization via Gradient flow in Offline RL)를 온라인 강화학습으로 확장한 알고리즘입니다. JKO (Jordan-Kinderlehrer-Otto) chain을 사용하여 여러 actor를 순차적으로 학습하며, Gaussian policy의 경우 closed form W2 거리를 활용한 정책 간 거리 측정을 구현합니다.

## 주요 특징

- **JKO Chain**: 여러 actor를 순차적으로 학습하는 gradient flow 기반 접근법
- **Gaussian Policy**: SAC 기반의 Gaussian policy를 사용한 정책 표현
- **Closed Form W2 Distance**: Gaussian policy 간 거리를 closed form으로 계산 $(d\mu)^2 + (d\sigma)^2$
- **온라인 학습**: 환경과 직접 상호작용하며 실시간으로 학습
- **재현성 보장**: 모든 랜덤 샘플링에 시드 고정 기능 포함
- **MuJoCo 지원**: MuJoCo 연속 제어 환경 지원
- **자동화된 실험**: `config_sac.yaml` 기반 대규모 실험 자동 실행

## 알고리즘 개요

### JKO Chain과 Gradient Flow

POSAC은 POGO의 JKO chain 구조를 온라인 RL에 적용합니다. 여러 actor를 순차적으로 연결하여 연속적인 gradient flow를 근사합니다:

- **Actor 0 ($\pi_0$)**: 순수 SAC loss로 학습 (W2 regularization 없음)
- **Actor i ($\pi_i$, $i \geq 1$)**: 이전 actor ($\pi_{i-1}$)에 대한 W2 거리로 학습

각 actor는 gradient flow의 한 단계를 나타내며, 전체 chain은 연속적인 정책 진화를 이산적으로 근사합니다.

**Actor 0가 W2 regularization을 사용하지 않는 이유**: 첫 번째 actor는 데이터셋이 아닌 환경과의 상호작용으로부터 학습하므로, 이전 정책에 대한 제약 없이 자유롭게 학습할 수 있습니다. 이후 actor들은 이전 actor라는 연속적인 분포를 reference로 하므로, 분포 간 거리를 측정하는 W2 거리가 필요합니다.

### 학습 목표

각 actor는 다음 JKO loss를 최소화합니다:

$$L_i = -\lambda \cdot \mathbb{E}[Q(s, \pi_i(s))] + w_i \cdot W_2(\pi_i, \pi_{i-1}) + \alpha \cdot H(\pi_i)$$

여기서:
- $Q(s, a)$: Critic 네트워크의 Q-value
- $W_2(\pi_i, \pi_{i-1})$: W2 거리 ($i=0$일 때는 0)
- $w_i$: W2 거리의 가중치
- $\alpha$: Entropy regularization 계수 (SAC의 temperature)
- $H(\pi_i)$: Policy의 entropy
- $\lambda$: Q-value의 정규화 계수

### Gaussian Policy

Actor는 Gaussian policy를 사용합니다:

$$\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$

여기서 $\mu_\theta(s)$와 $\sigma_\theta(s)$는 state를 입력으로 받아 action의 mean과 standard deviation을 출력하는 neural network입니다.

### Closed Form W2 Distance

Gaussian policy의 경우, 두 정책 간 W2 거리를 closed form으로 계산할 수 있습니다:

$$W_2^2(\pi_i, \pi_{i-1}) = ||\mu_i - \mu_{i-1}||^2 + ||\sigma_i - \sigma_{i-1}||^2$$

여기서:
- $\mu_i, \mu_{i-1}$: 두 정책의 mean
- $\sigma_i, \sigma_{i-1}$: 두 정책의 standard deviation

이는 Sinkhorn 알고리즘보다 훨씬 효율적이며, 정확한 거리 계산이 가능합니다.

### Critic 학습

Critic은 첫 번째 actor (online actor)를 behavior policy로 사용하여 TD target을 계산합니다:

$$Q_{\text{target}} = r + \gamma \cdot \min(Q_1(s', \pi_0(s')), Q_2(s', \pi_0(s'))) - \alpha \cdot \log \pi_0(a'|s')$$

Online policy를 사용함으로써 target policy의 지연 업데이트 문제를 완화하고, 더 빠른 학습을 가능하게 합니다.

## 설치

### 요구사항

- Python 3.8+
- PyTorch (CUDA 지원 권장)
- Gymnasium (MuJoCo 환경 지원)
- NumPy, Matplotlib, Pandas

### Conda 환경 설정

```bash
conda create -n onrl python=3.10
conda activate onrl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[mujoco] tensorboard numpy matplotlib pandas pyyaml
```

## 사용 방법

### 단일 실험 실행

```bash
python train_sac.py \
    --env-name Humanoid-v5 \
    --seed 123456 \
    --num-steps 1000000 \
    --eval-freq 10 \
    --num-actors 2 \
    --w2-weights "0.0,0.1" \
    --lr 0.0003 \
    --cuda
```

### 대규모 실험 자동 실행

`config_sac.yaml` 파일을 수정한 후:

```bash
python run_sac.py --config config_sac.yaml
```

### Config 파일 구조

`config_sac.yaml`에서 각 환경별로 `w2_weights` 또는 `w2_weights_list`를 설정할 수 있습니다:

```yaml
common:
  seeds: [123456, 789012]
  enabled_environments:
    - Humanoid-v5
    - HalfCheetah-v5
  num_actors: 2

environments:
  Humanoid-v5:
    w2_weights_list:
      - [0.0, 0.1]  # 첫 번째 actor: 0.0, 두 번째 actor: 0.1
      - [0.0, 0.2]
      - [0.0, 0.3]
  
  HalfCheetah-v5:
    w2_weights: [0.0, 0.1]  # 단일 설정
```

## 주요 하이퍼파라미터

### Agent 파라미터

- `num_actors`: Actor 개수 (기본값: `1`)
- `w2_weights`: 각 actor의 W2 거리 가중치 리스트 (예: `"0.0,0.1"`)
- `lr`: 학습률 (기본값: `0.0003`)
- `gamma`: 할인 계수 (기본값: `0.99`)
- `tau`: Target network soft-update 계수 (기본값: `0.005`)
- `alpha`: Temperature parameter (기본값: `0.2`)
- `automatic_entropy_tuning`: 자동 entropy 조정 (기본값: `False`)
- `batch_size`: 배치 크기 (기본값: `256`)

### 학습 파라미터

- `num_steps`: 최대 타임스텝 (기본값: `1000000`)
- `eval_freq`: 평가 주기 (episodes, 기본값: `10`)
- `start_steps`: 랜덤 액션 샘플링 기간 (기본값: `10000`)
- `updates_per_step`: 스텝당 업데이트 횟수 (기본값: `1`)

## 결과 확인

### 로그 파일

학습 로그는 다음 위치에 저장됩니다:
- **TensorBoard**: `runs/{file_name}/`
- **CSV Metrics**: `results/{file_name}/training/{file_name}_metrics.csv`
- **평가 결과**: `results/{file_name}/{file_name}_actor_{i}.npy`

### TensorBoard 시각화

```bash
tensorboard --logdir runs/
```

### 평가 결과

각 evaluation step마다 모든 actor의 성능이 출력됩니다:

```
Evaluation over 10 episodes:
  Actor 0 - Deterministic: 5234.56
  Actor 0 - Stochastic: 5123.45
  Actor 1 - Deterministic: 5345.67
  Actor 1 - Stochastic: 5212.34
```

### 체크포인트

체크포인트는 `checkpoints/{env_name}/{file_name}_final`에 저장되며, 학습 중단 시 재개할 수 있습니다.

## 재현성

모든 랜덤 샘플링에 시드 고정이 적용되어 있습니다:

- **데이터 샘플링**: ReplayBuffer에서 배치 샘플링 시 시드 고정
- **정책 샘플링**: Gaussian policy의 action 샘플링 시 시드 고정
- **환경 초기화**: 평가 시 환경 초기화 시드 고정

같은 시드로 실행하면 완전히 동일한 결과를 얻을 수 있습니다.

## 파일 구조

```
POSAC/
├── sac.py              # POSAC 알고리즘 구현 (Multi-step actor update)
├── train_sac.py        # 단일 실험 실행 스크립트
├── run_sac.py          # 대규모 실험 자동 실행 스크립트
├── config_sac.yaml     # 실험 설정 파일
├── model.py            # Neural network models (GaussianPolicy, QNetwork)
├── replay_memory.py    # Experience replay buffer
├── utils.py            # 유틸리티 함수
├── runs/               # TensorBoard 로그
├── results/             # 평가 결과 저장
└── checkpoints/        # 모델 체크포인트
```

## 지원 환경

### MuJoCo 연속 제어 환경
- `Humanoid-v5`
- `HalfCheetah-v5`
- `Walker2d-v5`
- `Ant-v5`
- 기타 Gymnasium MuJoCo 환경

## POGO와의 차이점

| 특징 | POGO (Offline) | POSAC (Online) |
|------|----------------|-----------------|
| 데이터 소스 | D4RL 데이터셋 | 환경 상호작용 |
| Policy 타입 | Transport Map | Gaussian Policy |
| W2 거리 계산 | Sinkhorn 알고리즘 | Closed form |
| Actor 0 학습 | L2 to dataset actions | 순수 SAC loss |
| Critic target | Online policy | Online policy |

## 참고 문헌

POSAC은 POGO (Policy Optimization via Gradient flow in Offline RL)를 온라인 RL로 확장한 알고리즘입니다. 자세한 알고리즘 설명은 POGO 논문을 참고하세요.

## 라이선스

MIT License

Copyright (c) 2025 POSAC Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
