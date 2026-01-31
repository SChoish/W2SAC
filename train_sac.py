#!/usr/bin/env python3
"""
SAC with Multi-step Actor Update Training Script
"""

import argparse
import csv
import datetime
import gymnasium as gym
import gymnasium_robotics  # Adroit 및 Kitchen 환경 등록을 위해 필요
import itertools
import json
import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# 환경 등록
gym.register_envs(gymnasium_robotics)

from sac import SAC
from replay_memory import ReplayMemory


# ---------------------------
# 유틸 함수
# ---------------------------
def set_global_seed(env, seed: int):
    """환경 및 랜덤 시드 설정 (POGO 스타일)"""
    import random
    try:
        env.reset(seed=seed)
    except TypeError:
        # old gym 버전
        env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# 평가 루틴
# ---------------------------
@torch.no_grad()
def eval_policy(agent, eval_env, base_seed, eval_episodes=10, deterministic=True, actor_idx=None):
    """정책 평가: deterministic과 stochastic 모두 평가 (POGO 스타일)
    
    Args:
        agent: SAC agent
        eval_env: 평가용 환경
        base_seed: 기본 시드
        eval_episodes: 평가 episode 수
        deterministic: True면 deterministic 평가, False면 stochastic 평가
        actor_idx: 평가할 actor 인덱스 (None이면 마지막 actor)
    
    Returns:
        tuple: (avg_reward, episode_rewards)
    """
    episode_rewards = []
    step_count = 0
    
    for ep in range(eval_episodes):
        # 재현성을 위해 각 episode마다 환경 시드 재설정 (POGO 스타일)
        ep_seed = base_seed + ep
        try:
            eval_env.action_space.seed(ep_seed)
            reset_result = eval_env.reset(seed=ep_seed)
        except (TypeError, AttributeError):
            try:
                reset_result = eval_env.reset(seed=ep_seed)
            except TypeError:
                reset_result = eval_env.reset()
        
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        # Dict observation space 처리
        if isinstance(state, dict):
            state = state['observation']
        
        episode_reward = 0
        done = False
        
        while not done:
            # 재현성을 위해 episode와 step 기반 시드 사용 (POGO 스타일)
            action_seed = base_seed * 10000 + ep * 1000 + step_count if not deterministic else None
            action = agent.select_action(state, evaluate=deterministic, actor_idx=actor_idx, seed=action_seed)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Dict observation space 처리
            if isinstance(next_state, dict):
                next_state = next_state['observation']
            
            state = next_state
            step_count += 1
        
        episode_rewards.append(episode_reward)
    
    avg_reward = np.mean(episode_rewards)
    return avg_reward, episode_rewards


def final_evaluation(agent, env_name, seed, runs=5, episodes=10, actor_idx=None):
    """최종 평가: 여러 run에 걸쳐 평가"""
    eval_env = gym.make(env_name)
    set_global_seed(eval_env, seed + 10_000)
    
    det_scores, stoch_scores = [], []
    for r in range(runs):
        det_avg, _ = eval_policy(
            agent, eval_env, base_seed=1000 + 100 * r, 
            eval_episodes=episodes, deterministic=True, actor_idx=actor_idx
        )
        stoch_avg, _ = eval_policy(
            agent, eval_env, base_seed=2000 + 100 * r, 
            eval_episodes=episodes, deterministic=False, actor_idx=actor_idx
        )
        det_scores.append(det_avg)
        stoch_scores.append(stoch_avg)
    
    det_scores = np.array(det_scores, dtype=np.float32)
    stoch_scores = np.array(stoch_scores, dtype=np.float32)
    
    print("======== Final Evaluation ========")
    print(f"[FINAL] Deterministic: mean={det_scores.mean():.3f}, std={det_scores.std():.3f} over {runs}x{episodes}")
    print(f"[FINAL] Stochastic:   mean={stoch_scores.mean():.3f}, std={stoch_scores.std():.3f} over {runs}x{episodes}")
    
    eval_env.close()
    return det_scores, stoch_scores


# ---------------------------
# 체크포인트 유틸
# ---------------------------
def save_checkpoint(agent, env_name: str, ckpt_dir: str, prefix: str, step: int, extra_meta=None):
    """체크포인트 저장"""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, prefix)
    agent.save_checkpoint(env_name, suffix=prefix, ckpt_path=path)
    
    meta = {
        "step": int(step),
        "checkpoint_name": prefix,
    }
    if extra_meta:
        meta.update(extra_meta)
    
    with open(path + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[CKPT] Saved: {path} (step={step})")


# ---------------------------
# 통합 학습
# ---------------------------
def train_unified(agent, env, env_name, seed, memory, max_steps, eval_freq, 
                  updates_per_step, start_steps, batch_size, 
                  file_name, ckpt_dir, results_dir, writer, start_step=0):
    """통합 학습: 모든 actor를 동시에 학습하고 평가"""
    
    eval_env = gym.make(env_name)
    # 평가 환경은 별도 시드 사용 (학습 환경과 분리)
    eval_env_seed = seed + 1234
    set_global_seed(eval_env, eval_env_seed)
    
    num_actors = getattr(agent, "num_actors", 1)
    w2_weights = getattr(agent, "w2_weights", [0.0] * num_actors)
    w2_str = ", ".join([f"{w:.3f}" for w in w2_weights])
    
    print(f"🚀 통합 학습 시작: {start_step} ~ {max_steps-1} steps (SAC, {num_actors}개 actor)")
    print(f"   Actor weights: [{w2_str}]")
    
    # 평가 결과 저장용
    eval_files = {}
    evaluations = {}
    for i in range(num_actors):
        eval_file = results_dir / f"{file_name}_actor_{i}.npy"
        eval_files[i] = eval_file
        if start_step > 0 and eval_file.exists():
            evaluations[i] = list(np.load(eval_file))
        else:
            evaluations[i] = []
    
    # Metrics CSV 파일
    log_dir = results_dir / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / f"{file_name}_metrics.csv"
    metrics_file_exists = metrics_file.exists()
    
    if start_step == 0 or not metrics_file_exists:
        metrics_file_handle = open(metrics_file, 'w', newline='', encoding='utf-8')
        metrics_writer = None
    else:
        metrics_file_handle = open(metrics_file, 'a', newline='', encoding='utf-8')
        metrics_writer = None
    
    # 평가 로그 파일
    eval_log_file = log_dir / f"{file_name}_evaluation.log"
    eval_log_handle = open(eval_log_file, 'a', encoding='utf-8') if eval_log_file.exists() else open(eval_log_file, 'w', encoding='utf-8')
    
    # 이전 step의 metrics 저장
    prev_metrics = {}
    
    # Training loop
    total_numsteps = start_step
    updates = 0
    episode_rewards = []
    
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        # 재현성을 위해 episode별로 다른 시드 사용 (POGO 스타일)
        episode_seed = seed + i_episode
        try:
            state, _ = env.reset(seed=episode_seed)
        except TypeError:
            state, _ = env.reset()
        try:
            env.action_space.seed(episode_seed)
        except Exception:
            pass
        
        # Dict observation space 처리
        if isinstance(state, dict):
            state = state['observation']
        
        while not done:
            if start_steps > total_numsteps:
                # 랜덤 액션 샘플링 시 시드 고정 (POGO 스타일)
                np.random.seed(episode_seed * 10000 + episode_steps)
                action = env.action_space.sample()
            else:
                # 학습 중 action 선택 시 시드 고정 (POGO 스타일: episode와 step 기반)
                action_seed = episode_seed * 10000 + episode_steps
                action = agent.select_action(state, evaluate=False, actor_idx=0, seed=action_seed)
            
            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    metrics = agent.update_parameters(memory, batch_size, updates)
                    
                    # TensorBoard 로깅 (모든 메트릭)
                    writer.add_scalar('loss/critic_1', metrics['critic_1_loss'], updates)
                    writer.add_scalar('loss/critic_2', metrics['critic_2_loss'], updates)
                    writer.add_scalar('loss/policy', metrics['policy_loss'], updates)
                    writer.add_scalar('loss/entropy_loss', metrics['entropy_loss'], updates)
                    writer.add_scalar('entropy_temprature/alpha', metrics['alpha'], updates)
                    
                    # TensorBoard 로깅 (상세 메트릭 - POGO 스타일)
                    for key, value in metrics.items():
                        if 'actor_' in key or 'Q_' in key or 'w2_' in key:
                            writer.add_scalar(f'metrics/{key}', value, updates)
                    
                    # 현재 metrics와 이전 metrics 병합
                    merged_metrics = prev_metrics.copy()
                    merged_metrics.update(metrics)
                    prev_metrics = merged_metrics.copy()
                    
                    # CSV 로깅은 평가 시점에만 수행 (아래 평가 블록에서)
                    
                    updates += 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
            # Dict observation space 처리
            if isinstance(next_state, dict):
                next_state_obs = next_state['observation']
            else:
                next_state_obs = next_state
            
            mask = 0.0 if done else 1.0
            memory.push(state, action, reward, next_state_obs, mask)
            state = next_state_obs
        
        if total_numsteps > max_steps:
            break
        
        episode_rewards.append(episode_reward)
        writer.add_scalar('reward/train', episode_reward, i_episode)
        print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward:.2f}")
        
        # 평가 (eval_freq 주기로 실행)
        if i_episode % eval_freq == 0:
            print(f"\n[Evaluation] Episode: {i_episode}, Time steps: {total_numsteps}")
            
            # 모든 actor 평가
            actor_results = []
            eval_metrics = {}  # 평가 메트릭 저장용
            for i in range(num_actors):
                det_avg, _ = eval_policy(
                    agent, eval_env, base_seed=100 + i * 100, 
                    eval_episodes=10, deterministic=True, actor_idx=i
                )
                stoch_avg, _ = eval_policy(
                    agent, eval_env, base_seed=200 + i * 100, 
                    eval_episodes=10, deterministic=False, actor_idx=i
                )
                
                actor_results.append({
                    'det_avg': det_avg,
                    'stoch_avg': stoch_avg
                })
                
                # 평가 메트릭 저장 (CSV에 포함될 메트릭)
                eval_metrics[f'actor_{i}_det_reward'] = det_avg
                eval_metrics[f'actor_{i}_sto_reward'] = stoch_avg
                
                # 평가 결과 저장
                evaluations[i].append(det_avg)
                np.save(eval_files[i], evaluations[i])
                
                # TensorBoard 로깅
                writer.add_scalar(f'avg_reward/actor_{i}_deterministic', det_avg, i_episode)
                writer.add_scalar(f'avg_reward/actor_{i}_stochastic', stoch_avg, i_episode)
                writer.add_scalar(f'avg_reward/actor_{i}_deterministic_steps', det_avg, total_numsteps)
                writer.add_scalar(f'avg_reward/actor_{i}_stochastic_steps', stoch_avg, total_numsteps)
            
            # 평가 메트릭을 prev_metrics에 추가 (CSV 저장용)
            prev_metrics.update(eval_metrics)
            
            # CSV 로깅 (평가 시점에 모든 메트릭 저장)
            row = {'step': total_numsteps, 'episode': i_episode}
            row.update(prev_metrics)  # 모든 메트릭 포함 (학습 메트릭 + 평가 메트릭)
            
            if metrics_writer is None:
                fieldnames = ['step', 'episode'] + sorted([k for k in row.keys() if k not in ['step', 'episode']])
                metrics_writer = csv.DictWriter(metrics_file_handle, fieldnames=fieldnames, extrasaction='ignore')
                if not metrics_file_exists:
                    metrics_writer.writeheader()
            else:
                new_fields = [k for k in row.keys() if k not in metrics_writer.fieldnames]
                if new_fields:
                    metrics_writer.fieldnames = list(metrics_writer.fieldnames) + sorted(new_fields)
            
            metrics_writer.writerow(row)
            metrics_file_handle.flush()
            
            # 결과 출력 (더 명확하게)
            print("=" * 60)
            print(f"Evaluation Results (Episode {i_episode}, Steps {total_numsteps}):")
            print("-" * 60)
            for i in range(num_actors):
                r = actor_results[i]
                print(f"  Actor {i} (w2_weight={w2_weights[i]:.3f}):")
                print(f"    Deterministic: {r['det_avg']:.2f}")
                print(f"    Stochastic:   {r['stoch_avg']:.2f}")
            print("=" * 60)
            print()
            
            # 로컬 로그 파일에 저장 (POGO 스타일)
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            eval_log_handle.write(f"[{timestamp_str}] Episode {i_episode}, Steps {total_numsteps}\n")
            eval_log_handle.write("-" * 60 + "\n")
            for i in range(num_actors):
                r = actor_results[i]
                eval_log_handle.write(f"Actor {i} (w2_weight={w2_weights[i]:.3f}):\n")
                eval_log_handle.write(f"  Deterministic: {r['det_avg']:.2f}\n")
                eval_log_handle.write(f"  Stochastic:   {r['stoch_avg']:.2f}\n")
            eval_log_handle.write("=" * 60 + "\n\n")
            eval_log_handle.flush()
    
    # Metrics 파일 닫기
    if metrics_file_handle is not None:
        metrics_file_handle.close()
        print(f"📊 Metrics saved to: {metrics_file}")
    
    # 평가 로그 파일 닫기
    if eval_log_handle is not None:
        eval_log_handle.close()
        print(f"📝 Evaluation log saved to: {eval_log_file}")
    
    eval_env.close()
    return agent, episode_rewards


# ---------------------------
# Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    
    # Environment
    parser.add_argument('--env-name', default="Humanoid-v5",
                        help='Mujoco Gym environment (default: Humanoid-v5)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--kitchen-tasks', type=str, default=None,
                        help='Kitchen tasks to complete (comma-separated, e.g., "microwave,kettle")')
    
    # Training
    parser.add_argument('--num-steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--eval-freq', type=int, default=10, metavar='N',
                        help='evaluation frequency in episodes (default: 10)')
    parser.add_argument('--start-steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    
    # Network
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    
    # SAC parameters
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=False, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    
    # Memory
    parser.add_argument('--replay-size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    
    # W2 regularization parameters
    parser.add_argument('--w2-reg-weight', type=float, default=0.1, metavar='G',
                        help='W2 regularization weight (default: 0.1)')
    parser.add_argument('--old-policy-update-freq', type=int, default=5, metavar='N',
                        help='old policy update frequency (default: 5)')
    
    # Multi-step actor parameters
    parser.add_argument('--num-actors', type=int, default=1, metavar='N',
                        help='number of actors for multi-step update (default: 1)')
    parser.add_argument('--w2-weights', type=str, default=None, metavar='G',
                        help='W2 weights for each actor (comma-separated, e.g., "0.0,0.1")')
    
    # System
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints",
                        help='checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--results-dir', type=str, default="./results",
                        help='results directory (default: ./results)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 실험 환경 정보 출력
    print("=" * 60)
    print("SAC 실험 설정")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Seed: {args.seed}")
    print(f"Max timesteps: {args.num_steps:,}")
    print(f"Evaluation frequency: {args.eval_freq} episodes")
    print(f"Number of actors: {args.num_actors}")
    print(f"W2 reg weight: {args.w2_reg_weight}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Discount: {args.gamma}")
    print(f"Tau: {args.tau}")
    print(f"Alpha: {args.alpha}")
    print(f"Automatic entropy tuning: {args.automatic_entropy_tuning}")
    print("=" * 60)
    print()
    
    # Parse w2_weights
    if args.w2_weights is not None:
        try:
            w2_weights_list = [float(x.strip()) for x in args.w2_weights.split(',')]
            args.w2_weights = w2_weights_list
        except ValueError:
            try:
                single_value = float(args.w2_weights)
                args.w2_weights = [0.0] + [single_value] * (args.num_actors - 1)
            except ValueError:
                print(f"Warning: Invalid w2_weights format '{args.w2_weights}', using default [0.0]")
                args.w2_weights = [0.0] * args.num_actors
    else:
        args.w2_weights = [0.0] + [args.w2_reg_weight] * (args.num_actors - 1)
    
    print(f"W2 weights: {args.w2_weights}")
    print()
    
    # 환경 설정 (시드 완전히 고정)
    # Kitchen 환경의 경우 tasks_to_complete 설정
    if 'Kitchen' in args.env_name and args.kitchen_tasks is not None:
        tasks = [t.strip() for t in args.kitchen_tasks.split(',')]
        env = gym.make(args.env_name, tasks_to_complete=tasks)
    else:
        env = gym.make(args.env_name)
    set_global_seed(env, args.seed)
    # 추가 시드 고정 (모든 랜덤 소스)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    # Observation space 처리 (Dict observation space 지원)
    if isinstance(env.observation_space, gym.spaces.Dict):
        # Dict observation space의 경우 'observation' 키 사용
        obs_space = env.observation_space['observation']
        obs_dim = obs_space.shape[0]
    else:
        obs_dim = env.observation_space.shape[0]
    
    # Agent 생성 (시드 전달하여 네트워크 초기화도 고정)
    agent = SAC(obs_dim, env.action_space, args)
    
    # 파일 이름 및 디렉토리 설정
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_name = "POSAC" if args.w2_reg_weight > 0 else "SAC"
    # w2_weights를 파일명에 포함하여 구분 (점을 p로 변경)
    w2_str = "_".join([str(w).replace(".", "p") for w in args.w2_weights[:3]])
    file_name = f"{agent_name}_{args.env_name}_{args.seed}_w2_{w2_str}_{timestamp}"
    
    # 디렉토리 생성
    ckpt_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir) / file_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(f'runs/{file_name}')
    
    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    
    # 학습
    agent, episode_rewards = train_unified(
        agent, env, args.env_name, args.seed, memory,
        max_steps=args.num_steps,
        eval_freq=args.eval_freq,
        updates_per_step=args.updates_per_step,
        start_steps=args.start_steps,
        batch_size=args.batch_size,
        file_name=file_name,
        ckpt_dir=str(ckpt_dir),
        results_dir=results_dir,
        writer=writer,
        start_step=0
    )
    
    # 체크포인트 저장
    save_checkpoint(agent, args.env_name, str(ckpt_dir), f"{file_name}_final", args.num_steps, {
        "file_name": file_name,
        "episode_rewards": episode_rewards[-100:] if len(episode_rewards) > 100 else episode_rewards
    })
    
    # 최종 평가
    num_actors = getattr(agent, "num_actors", 1)
    print("\n======== Final Evaluation (all actors) ========")
    for i in range(num_actors):
        print(f"\n======== Final Evaluation: Actor {i} ========")
        final_evaluation(
            agent, args.env_name, args.seed,
            runs=5, episodes=10, actor_idx=i
        )
    
    # 결과 저장
    final_data = {
        'agent': agent_name,
        'episode_rewards': episode_rewards,
        'config': {
            'env_name': args.env_name,
            'seed': args.seed,
            'num_actors': args.num_actors,
            'w2_weights': args.w2_weights,
            'w2_reg_weight': args.w2_reg_weight,
        }
    }
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(final_data, f, indent=2)
    
    np.save(results_dir / "episode_rewards.npy", episode_rewards)
    
    print(f"\n✅ 결과가 {results_dir}/에 저장되었습니다!")
    env.close()


if __name__ == "__main__":
    main()
