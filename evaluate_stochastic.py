#!/usr/bin/env python3
"""
체크포인트를 로드해서 stochastic evaluation을 수행하는 스크립트
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import os
from sac import SAC

def evaluate_stochastic(agent, env, num_episodes=100, seed=42):
    """Stochastic evaluation 수행"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    rewards = []
    episode_lengths = []
    
    print(f"Stochastic evaluation 시작: {num_episodes} episodes")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Stochastic sampling (evaluate=False)
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Safety check for infinite episodes
            if episode_length > 10000:
                print(f"Episode {episode} too long, breaking...")
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    return rewards, episode_lengths

def evaluate_deterministic(agent, env, num_episodes=100, seed=42):
    """Deterministic evaluation 수행 (비교용)"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    rewards = []
    episode_lengths = []
    
    print(f"Deterministic evaluation 시작: {num_episodes} episodes")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Deterministic sampling (evaluate=True)
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Safety check for infinite episodes
            if episode_length > 10000:
                print(f"Episode {episode} too long, breaking...")
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    return rewards, episode_lengths

def main():
    parser = argparse.ArgumentParser(description='Stochastic vs Deterministic Evaluation')
    parser.add_argument('--env-name', default='Humanoid-v5', help='Environment name')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    
    args = parser.parse_args()
    
    # Environment 생성
    env = gym.make(args.env_name)
    
    # Agent 생성
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    
    # Checkpoint 로드
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    agent.load_checkpoint(args.checkpoint, evaluate=True)
    print(f"Checkpoint 로드 완료: {args.checkpoint}")
    
    # Stochastic evaluation
    print("\n" + "="*50)
    print("STOCHASTIC EVALUATION")
    print("="*50)
    stochastic_rewards, stochastic_lengths = evaluate_stochastic(agent, env, args.num_episodes, args.seed)
    
    # Deterministic evaluation
    print("\n" + "="*50)
    print("DETERMINISTIC EVALUATION")
    print("="*50)
    deterministic_rewards, deterministic_lengths = evaluate_deterministic(agent, env, args.num_episodes, args.seed)
    
    # 결과 분석
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Stochastic Evaluation:")
    print(f"  Mean Reward: {np.mean(stochastic_rewards):.2f} ± {np.std(stochastic_rewards):.2f}")
    print(f"  Min Reward: {np.min(stochastic_rewards):.2f}")
    print(f"  Max Reward: {np.max(stochastic_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(stochastic_lengths):.2f} ± {np.std(stochastic_lengths):.2f}")
    
    print(f"\nDeterministic Evaluation:")
    print(f"  Mean Reward: {np.mean(deterministic_rewards):.2f} ± {np.std(deterministic_rewards):.2f}")
    print(f"  Min Reward: {np.min(deterministic_rewards):.2f}")
    print(f"  Max Reward: {np.max(deterministic_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(deterministic_lengths):.2f} ± {np.std(deterministic_lengths):.2f}")
    
    # 통계적 유의성 검정
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(stochastic_rewards, deterministic_rewards)
    print(f"\nStatistical Test (Paired t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    
    # 결과 저장
    results = {
        'stochastic_rewards': stochastic_rewards,
        'stochastic_lengths': stochastic_lengths,
        'deterministic_rewards': deterministic_rewards,
        'deterministic_lengths': deterministic_lengths,
        'statistics': {
            'stochastic_mean': np.mean(stochastic_rewards),
            'stochastic_std': np.std(stochastic_rewards),
            'deterministic_mean': np.mean(deterministic_rewards),
            'deterministic_std': np.std(deterministic_rewards),
            't_statistic': t_stat,
            'p_value': p_value
        }
    }
    
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과가 evaluation_results.json에 저장되었습니다.")
    
    env.close()

if __name__ == "__main__":
    main()
