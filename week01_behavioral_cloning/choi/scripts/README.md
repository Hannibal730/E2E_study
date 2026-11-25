# Choi · 1주차 학습 및 평가

아래 명령은 프로젝트 가상환경을 활성화한 뒤 `scripts/`에서 실행한다.

```bash
cd ~/E2E_study
source .venv/bin/activate
cd week01_behavioral_cloning/choi/scripts
```

| 작업 | 명령 | [config.yaml](config.yaml)에서 지정할 항목 |
|---|---|---|
| 학습 | `python train.py` | `train_models`에서 모델 선택. Model B의 좌우 보정값은 `training.left_delta`, `training.right_delta`로 지정 |
| 기존 모델 평가 | `python evaluate.py` | `evaluation.run_dir`에 시각 폴더 이름, `evaluation.checkpoints`에 해당 폴더 기준 체크포인트 경로 지정 |
| 결과 시각화 | `python plot_results.py` | `plot.input_csvs`에 비교할 실행의 `histogram.csv` 경로 지정 |

학습 결과는 `../results/<시각>/<모델>/`에 저장된다. `best.pt`는 최적 체크포인트, `history.csv`와 `learning_curve.png`는 에폭별 학습 기록과 그래프다. `evaluation.after_training: true`이면 학습 직후 해당 실행을 평가해 시각 폴더에 `metrics.csv`와 `histogram.csv`를 생성한다. 이 경우 수동 평가 단계는 생략할 수 있다.

시각화는 전체 조향 분포를 `plot.output_png`에 저장하고, 같은 폴더에 모델별 분포(`distributions_by_delta.png`)와 델타별 성능(`metrics_by_delta.png`)을 저장한다. 델타별 성능 그래프에는 각 실행의 평가 지표 CSV도 필요하다.

[experiment_config.py](experiment_config.py)는 위 스크립트가 가져다 쓰는 공통 모듈이다. YAML 읽기, 좌우 델타 확인, 실행 폴더와 출력 경로 처리를 담당하므로 별도 실행 명령은 없다. 시뮬레이터 주행 방법은 [SIMULATOR.md](SIMULATOR.md)에 정리되어 있다.
