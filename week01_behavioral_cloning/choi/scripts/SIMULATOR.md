# Udacity 시뮬레이터 실행

게임 파일은 `choi/beta_simulator_linux/`에 있다. 아래 명령으로 주행 서버와 시뮬레이터를 함께 실행한다.

```bash
cd ~/E2E_study
source .venv/bin/activate
cd week01_behavioral_cloning/choi/scripts
python simulator.py ../results/20261122_180415_110759/model_b/best.pt
```

설정 창에서 **800×600 · Fastest · Windowed**를 선택하고 **OK**를 누른다. 메뉴에서 트랙을 고른 뒤 **Autonomous Mode**를 누른다.

- 다른 모델로 주행하려면 `best.pt` 경로를 바꾼다. 목표 속도는 `--target-speed`로 지정한다(기본값 8.0).
- 수동 주행은 `python simulator.py`로 창을 열고 **Training Mode**를 선택한다.
- 창이 이미 열려 있으면 `python simulator.py --server-only <체크포인트 경로>`로 주행 서버만 시작한다.
- 종료하려면 실행한 터미널에서 `Ctrl+C`를 누른다.
