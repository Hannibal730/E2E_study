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

예를 들어 분할 CSV의 `image_path`가 `track1data/IMG/center_2019_04_02_19_25_33_671.jpg`이면 실제 파일 위치는 `data/Udacity/track1data/IMG/center_2019_04_02_19_25_33_671.jpg`이다.

## 데이터 분할

분할 단위는 헤더 없는 원본 `driving_log.csv`의 한 행이다. `row_index`는 0부터 시작한다. Track 1의 처음 80%를 학습용으로, 뒤 20%를 검증용으로 배정한 뒤 학습 구간에서만 카메라별 샘플을 구성했다.

| 목록 | 트랙 | 원본 행 인덱스 (양 끝 포함) | 카메라 | 샘플 수 |
|---|---|---|---|---:|
| [Train A](splits/train_a.csv) | track1data | 0–8491 | center | 8,492 |
| [Train B](splits/train_b.csv) | track1data | 0–8491 | center, left, right | 25,476 |
| [공통 Val](splits/val.csv) | track1data | 8492–10614 | center | 2,123 |
| [공통 OOD Test](splits/test.csv) | track2data | 0–21827 | center | 21,828 |

## 사용 순서

1. 각 모델에 해당하는 Train CSV를 읽는다. 각 행은 이미지 한 장과 조향 목표값 하나를 나타낸다. 원본 기록은 `track`과 `row_index`로 식별하고 시점은 `camera` 열로 구분한다.
2. 학습 시 CSV의 `steering_target`을 그대로 사용한다. Train B에서는 주행 로그의 값 `s = steering_recorded`에 카메라별 고정 보정을 한 번 적용해 저장했다. 아래 예시는 첫 주행 기록(`row_index = 0`)에 해당한다.

   | 카메라 | 목표값 계산 | 기록값이 0일 때 목표값 |
   |---|---|---:|
   | center | `s` | 0 |
   | left | `s + 0.2` | 0.2 |
   | right | `s - 0.2` | −0.2 |

   0.2는 측면에서 중앙으로 복귀하는 동작을 근사하기 위해 원본 조향값 척도에서 정한 실험 상수다. 보정값은 클리핑하지 않아 Train B의 측면 샘플 45개가 [−1, 1] 범위를 벗어난다.
3. 각 모델의 체크포인트는 공통 Val의 MSE로 선택한다. 선택한 두 모델을 공통 OOD Test에서 평가한다.
