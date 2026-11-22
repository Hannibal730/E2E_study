# 1주차 · 행동 복제와 오프라인 조향 보정

같은 주행 구간에서 정면 영상으로 학습한 모델 A와 측면 복귀 보정을 추가해 학습한 모델 B를 비교한다. 원본 이미지와 주행 기록은 [Udacity 데이터 폴더](../data/Udacity/)에 있다.

## 원본 데이터 배치

저장소 루트를 기준으로 아래와 같이 파일을 배치한다. 각 트랙의 `IMG/`에는 세 카메라의 JPEG 파일이 함께 들어 있다.

```text
data/
└── Udacity/
    ├── track1data/          # 학습 및 검증
    │   ├── driving_log.csv
    │   └── IMG/
    │       ├── center_*.jpg
    │       ├── left_*.jpg
    │       └── right_*.jpg
    ├── track2data/          # OOD 테스트
    │   ├── driving_log.csv
    │   └── IMG/
    └── dataset/             # 두 트랙의 통합본 (1주차 실험에서는 생략 가능)
        ├── driving_log.csv
        └── IMG/
```

분할 CSV의 `image_path`는 `data/Udacity/`를 기준으로 한 상대 경로다.

## 데이터 분할

분할 단위는 헤더 없는 원본 `driving_log.csv`의 한 행이다. `row_index`는 0부터 시작한다. Track 1의 처음 80%를 학습용으로, 뒤 20%를 검증용으로 배정한 뒤 학습 구간에서만 카메라별 샘플을 구성했다.

| 목록 | 트랙 | 원본 행 인덱스 (양 끝 포함) | 카메라 | 샘플 수 |
|---|---|---|---|---:|
| [Train A](splits/train_a.csv) | track1data | 0–8491 | center | 8,492 |
| [Train B](splits/train_b.csv) | track1data | 0–8491 | center, left, right | 25,476 |
| [공통 Val](splits/val.csv) | track1data | 8492–10614 | center | 2,123 |
| [공통 OOD Test](splits/test.csv) | track2data | 0–21827 | center | 21,828 |

## 학습 CSV 구성

두 파일은 Track 1의 동일한 8,492개 주행 시점을 사용한다. `track`과 `row_index`가 같으면 같은 시점이고, `camera`는 촬영 위치를 나타낸다.

- **`train_a.csv`**는 모델 A의 학습 목록이다. 시점마다 정면 이미지 한 장과 해당 시점의 원본 조향값을 한 행에 담는다.
- **`train_b.csv`**는 모델 B의 학습 목록이다. 시점마다 정면·왼쪽·오른쪽 이미지를 한 행씩 담아 총 25,476행이다. **CSV에 저장된 측면 보정량은 δ = 0**이므로 세 행 모두 같은 원본값을 가지며, `steering_target`과 `steering_recorded`가 일치한다.

## 조향값과 평가 기준

모델 B를 학습할 때 각 팀원은 실험별로 δ를 정하고, 원본 조향값 `s = steering_recorded`에서 카메라별 학습 목표를 계산한다.

| 카메라 | CSV에 저장된 목표값 | 모델 B 학습 시 목표값 |
|---|---|---|
| center | `s` | `s` |
| left | `s` | `s + δ` |
| right | `s` | `s - δ` |

모든 실험은 공통 Val과 OOD Test에서 정면 이미지와 원본 조향값을 사용한다. 체크포인트는 Val MSE로 선택하고, OOD Test에서 모델별 MSE, MAE와 조향 분포를 비교한다.
