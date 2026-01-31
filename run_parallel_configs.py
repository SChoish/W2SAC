#!/usr/bin/env python3
"""
두 개의 config 파일을 병렬로 실행하는 스크립트
onrl conda 환경을 기본으로 사용
"""
import subprocess
import sys
import os
from pathlib import Path

def find_onrl_python():
    """onrl conda 환경의 python 경로 찾기"""
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
    
    # 찾지 못하면 현재 python 사용 (경고 출력)
    print("⚠️  Warning: onrl conda 환경을 찾을 수 없습니다. 현재 Python을 사용합니다.")
    print("   onrl 환경을 활성화한 후 실행하거나, conda 환경 경로를 확인하세요.")
    return sys.executable

def run_config(config_file, log_file):
    """단일 config 파일 실행 (onrl 환경 사용)"""
    print(f"🚀 Starting: {config_file} (log: {log_file})")
    python_exec = find_onrl_python()
    cmd = [python_exec, "run_sac.py", "--config", config_file]
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            env=os.environ.copy()  # 환경 변수 유지
        )
    return process

def main():
    configs = [
        ("config_sac.yaml", "logs_config1.txt"),
        ("config2_sac.yaml", "logs_config2.txt")
    ]
    
    processes = []
    
    print("=" * 60)
    print("병렬 실험 시작")
    print("=" * 60)
    
    # 모든 config를 병렬로 시작
    for config_file, log_file in configs:
        if not os.path.exists(config_file):
            print(f"❌ Config 파일을 찾을 수 없습니다: {config_file}")
            continue
        
        process = run_config(config_file, log_file)
        processes.append((config_file, process, log_file))
        print(f"✅ {config_file} 시작됨 (PID: {process.pid})")
    
    print("\n" + "=" * 60)
    print("실험 진행 중...")
    print("로그 파일:")
    for _, _, log_file in processes:
        print(f"  - {log_file}")
    print("=" * 60)
    print("\n실험을 중지하려면 Ctrl+C를 누르세요.\n")
    
    # 모든 프로세스 완료 대기
    try:
        for config_file, process, log_file in processes:
            return_code = process.wait()
            if return_code == 0:
                print(f"✅ {config_file} 완료!")
            else:
                print(f"❌ {config_file} 실패 (exit code: {return_code})")
    except KeyboardInterrupt:
        print("\n\n⚠️  중단 신호 수신. 모든 프로세스를 종료합니다...")
        for config_file, process, _ in processes:
            process.terminate()
            process.wait()
        print("모든 프로세스가 종료되었습니다.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("모든 실험 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
