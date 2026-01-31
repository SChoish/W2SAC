#!/usr/bin/env python3
"""
Config 기반 SAC 실험 순차 실행.
- common.enabled_environments, common.seeds 사용
- environments.<env_name>.w2_weights 또는 w2_weights_list 사용
  - w2_weights: [w1, w2, ...] → 해당 weight 1개만 실행
  - w2_weights_list: [[w1, w2, ...], ...] → 여러 weight를 순차 실행
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path

import yaml


def get_w2_weights_list(env_config):
    """
    env_config에서 실행할 w2_weights 목록 반환.
    w2_weights_list가 있으면 그대로, 없으면 w2_weights를 1개짜리 리스트로.
    """
    if "w2_weights_list" in env_config:
        return env_config["w2_weights_list"]
    w = env_config.get("w2_weights", [0.0, 0.1])
    if w and isinstance(w[0], (int, float)):
        return [list(w)]
    return [list(w) for w in w]


def run_one(root_dir, pyexec, env_name, seed, w2_weights, num_actors, common, env_config=None, dry_run=False):
    """train_sac.py 한 번 실행 (해당 env, seed, w2_weights)."""
    env_dir = env_name.replace("-", "_")
    w2_weights = list(w2_weights)
    
    # env_config 기본값 설정
    if env_config is None:
        env_config = {}
    
    # w2_weights를 문자열로 변환
    w_str = ",".join(str(x) for x in w2_weights)
    
    # 결과 디렉토리 설정
    w2_str = "_".join(str(x).replace(".", "p") for x in w2_weights[:3])
    results_dir = root_dir / "results" / env_dir / f"w2_{w2_str}" / f"seed_{seed}"
    checkpoint_dir = root_dir / "checkpoints" / env_dir / f"w2_{w2_str}" / f"seed_{seed}"
    
    cmd = [
        str(pyexec), "-u", "train_sac.py",
        "--env-name", env_name,
        "--seed", str(seed),
        "--num-actors", str(num_actors),
        "--w2-weights", w_str,
        "--num-steps", str(common.get("max_timesteps", 1000000)),
        "--eval-freq", str(common.get("eval_freq", 10)),
        "--checkpoint-dir", str(checkpoint_dir),
        "--results-dir", str(results_dir),
    ]
    
    # Optional parameters
    if "lr" in common:
        cmd.extend(["--lr", str(common["lr"])])
    if "batch-size" in common:
        cmd.extend(["--batch-size", str(common["batch-size"])])
    if "gamma" in common:
        cmd.extend(["--gamma", str(common["gamma"])])
    if "tau" in common:
        cmd.extend(["--tau", str(common["tau"])])
    if "alpha" in common:
        cmd.extend(["--alpha", str(common["alpha"])])
    if "automatic-entropy-tuning" in common and common["automatic-entropy-tuning"]:
        cmd.append("--automatic-entropy-tuning")
    if "cuda" in common and common["cuda"]:
        cmd.append("--cuda")
    
    # Kitchen 환경의 tasks_to_complete 설정
    if env_name == "FrankaKitchen-v1" and "tasks_to_complete" in env_config:
        tasks_str = ",".join(env_config["tasks_to_complete"])
        cmd.extend(["--kitchen-tasks", tasks_str])
    
    if dry_run:
        print("  ", " ".join(cmd))
        return type("R", (), {"returncode": 0})()
    return subprocess.run(cmd, cwd=str(root_dir))


def find_onrl_python():
    """onrl conda 환경의 python 경로 찾기"""
    import os
    # 일반적인 conda 경로들
    possible_paths = [
        os.path.expanduser("~/anaconda3/envs/onrl/bin/python"),
        os.path.expanduser("~/miniconda3/envs/onrl/bin/python"),
        os.path.expanduser("~/conda/envs/onrl/bin/python"),
        "/opt/conda/envs/onrl/bin/python",
    ]
    
    # CONDA_PREFIX 환경 변수 확인
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and "onrl" in conda_prefix:
        onrl_python = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(onrl_python):
            return onrl_python
    
    # 가능한 경로들 확인
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 찾지 못하면 기본값
    return "python"

def main():
    p = argparse.ArgumentParser(description="Config 기반 SAC 실험 순차 실행")
    p.add_argument("--config", required=True, help="YAML config 경로")
    p.add_argument("--root-dir", default=".", type=Path, help="프로젝트 루트 디렉토리")
    p.add_argument("--pyexec", default=None, type=str, help="Python 실행 파일 경로 (기본: onrl 환경 자동 탐지)")
    p.add_argument("--dry-run", action="store_true", help="실행하지 않고 명령만 출력")
    args = p.parse_args()
    
    # pyexec가 지정되지 않았으면 onrl 환경 자동 탐지
    if args.pyexec is None:
        args.pyexec = find_onrl_python()
        print(f"🐍 Using Python: {args.pyexec}")

    root_dir = Path(args.root_dir).resolve()
    config_path = root_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        print(f"❌ Config 없음: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    common = cfg.get("common", {})
    seeds = common.get("seeds", [123456])
    enabled = common.get("enabled_environments", [])
    envs_cfg = cfg.get("environments", {})
    num_actors = common.get("num_actors", 1)

    tasks = []
    for env_name in enabled:
        if env_name not in envs_cfg:
            print(f"⚠️  Warning: {env_name} 설정이 없습니다. 건너뜁니다.")
            continue
        
        env_config = envs_cfg[env_name]
        w2_list = get_w2_weights_list(env_config)
        
        for w2_weights in w2_list:
            for seed in seeds:
                tasks.append((env_name, seed, w2_weights, env_config))

    print(f"🔬 총 {len(tasks)}개 실험 예정")
    print(f"📋 순차 실행 모드 (num_actors={num_actors})\n")

    for i, task in enumerate(tasks, 1):
        if len(task) == 4:
            env_name, seed, w2_weights, env_config = task
        else:
            env_name, seed, w2_weights = task
            env_config = {}
        w_str = ",".join(str(x) for x in w2_weights)
        print(f"🔄 [{i}/{len(tasks)}] {env_name} | seed={seed} | w2_weights=[{w_str}]")
        ret = run_one(root_dir, args.pyexec, env_name, seed, w2_weights, num_actors, common, env_config=env_config, dry_run=args.dry_run)
        if ret.returncode != 0:
            print(f"❌ 실패: {env_name} seed={seed} w2=[{w_str}] (exit {ret.returncode})")
            sys.exit(ret.returncode)
        print(f"✅ 완료: {env_name} seed={seed} w2=[{w_str}]\n")

    print("🏁 모든 실험 완료.")


if __name__ == "__main__":
    main()
