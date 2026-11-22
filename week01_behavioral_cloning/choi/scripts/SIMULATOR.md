# Udacity 시뮬레이터 실행

**터미널 1 · 주행 서버** (`model_b`, δ=0.8)

```bash
cd ~/E2E_study
source .venv/bin/activate
cd week01_behavioral_cloning/choi/scripts
python drive.py ../results/20261122_180415_110759/model_b/best.pt
```

다른 모델을 쓰려면 마지막 줄의 `best.pt` 경로를 바꾼다.

**터미널 2 · 시뮬레이터**

```bash
cd ~/E2E_study
./week01_behavioral_cloning/choi/scripts/start_simulator.sh
```

설정 창에서 **800×600 · Fastest · Windowed**를 선택하고 **OK**를 누른다. 터미널 1에 `127.0.0.1:4567` 대기 메시지가 뜨면 시뮬레이터에서 트랙을 선택한 뒤 **Autonomous Mode**를 누른다.
