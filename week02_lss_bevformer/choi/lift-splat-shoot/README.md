# Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D

PyTorch code for Lift-Splat-Shoot (ECCV 2020).

```

### Preparation
Download nuscenes data from [https://www.nuscenes.org/](https://www.nuscenes.org/). Install dependencies.

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

### Pre-trained Model
Download a pre-trained BEV vehicle segmentation model from here: [https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing](https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing)

| Vehicle IOU (reported in paper)        | Vehicle IOU (this repository)         |
|:-------------:|:-------------:| 
| 32.07      | 33.03 |

### Evaluate a model
Evaluate the IOU of a model on the nuScenes validation set. To evaluate on the "mini" split, pass `mini`. To evaluate on the "trainval" split, pass `trainval`.

```
python main.py eval_model_iou mini --modelf=./weights/model525000.pt --dataroot=../../../data/nuScenes --gpuid=0
```

### Visualize Predictions
From the `lift-splat-shoot/` directory, run the following commands to save the BEV predictions in `results/`.

```
cd results
python ../main.py viz_model_preds mini --modelf=../weights/model525000.pt --dataroot=../../../../data/nuScenes --map_folder=../../../../data/nuScenes/mini --gpuid=0
```
<img src="./imgs/eval.gif">

### 다른 카메라 데이터로 BEV 차량 점유 영역 추론하기

`weights/model525000.pt`를 사용하면 한 시점에 촬영한 카메라 영상으로 자차 주변의 어느 격자 칸에 차량이 있는지 예측할 수 있다. 위의 `eval_model_iou`와 `viz_model_preds`는 nuScenes 메타데이터에서 영상과 보정값을 조회한다. 일반 이미지 폴더에 적용하려면 **프레임별 영상과 해당 카메라의 보정값을 읽어 모델 입력을 만드는 로더**를 작성해야 한다. 단일 프레임 추론에 필요한 센서 데이터는 카메라 영상과 보정값이다. LiDAR, 지도, 차량 정답 라벨은 모델 입력에 포함되지 않는다.

#### 1. 카메라별로 준비할 데이터

| 항목 | 준비 방법 |
|---|---|
| RGB 영상 | 한 시점의 영상을 카메라 ID로 묶는다. 여러 대라면 타임스탬프를 맞춰 움직이는 차량이 각 영상에서 같은 시점에 관측되도록 한다. 전방 한 대만 있으면 카메라 수를 1로 두고, 결과를 해석할 때 전방 시야 범위를 고려한다. |
| 내부 파라미터 `K` | 카메라마다 원본 영상 해상도에서 측정한 초점거리 `fx, fy`와 주점 `cx, cy`를 픽셀 단위로 준비한다. 일반적인 핀홀 카메라의 행렬은 `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`이다. 어안 등 렌즈 왜곡이 크면 먼저 핀홀 영상으로 보정하고, 그 영상에 대응하는 내부 파라미터를 사용한다. |
| 외부 파라미터 `R, t` | 각 카메라에서 본 3차원 점을 자차 좌표로 바꾸는 회전행렬과 이동벡터를 준비한다. 모델의 식은 `p_ego = R @ p_camera + t`이며, `t`는 자차 기준 카메라 위치(m)다. 카메라 광학 좌표는 오른쪽·아래·전방이 각각 +x·+y·+z, 자차 좌표는 전방·왼쪽·위쪽이 각각 +x·+y·+z다. |

데이터가 반대 방향인 `p_camera = R_ec @ p_ego + t_ec`를 제공하면 `R = R_ec.T`, `t = -R_ec.T @ t_ec`로 변환한다. 데이터셋과 모델의 축 정의도 먼저 맞춘다. 예를 들어 전방을 수평으로 바라보는 카메라라면 카메라 광축 `[0, 0, 1]`에 `R`을 곱한 결과가 자차의 전방 `[1, 0, 0]`을 향해야 한다. 실제 장착 각도가 있으면 그만큼 기울어진다.

#### 2. 영상 전처리와 모델 입력

1. 각 영상을 RGB로 읽고 **해당 영상의 실제 높이와 너비**로 검증 전처리를 계산한다. 기존 코드는 `s = max(128 / H, 352 / W)`로 비율을 유지해 리사이즈한 뒤 가로 중앙을 기준으로 `128 × 352`(높이 × 너비)를 자른다. 세로 시작점은 `int(0.89 * resized_height) - 128`이며, 검증 시 좌우 반전과 회전은 적용하지 않는다. nuScenes 원본 크기인 `900 × 1600`을 다른 카메라에 그대로 대입하면 픽셀 위치가 어긋난다.
2. 리사이즈와 크롭을 영상과 함께 기록한다. 원본 픽셀을 `(u, v)`, 크롭의 왼쪽 위를 `(left, top)`이라 하면 입력 픽셀은 `(s*u - left, s*v - top)`이다. 이때 `post_rots = diag(s, s, 1)`, `post_trans = [-left, -top, 0]`이고 `intrins`에는 원본 `K`를 넣는다. 같은 변환을 `K`에도 적용하면 보정이 중복된다. 기존 `img_transform`과 `get_image_data`를 따르면 영상과 변환값을 함께 얻을 수 있다.
3. `normalize_img`로 RGB 값을 [0, 1]로 바꾼 뒤 채널별 평균 `[0.485, 0.456, 0.406]`과 표준편차 `[0.229, 0.224, 0.225]`로 정규화한다. OpenCV로 읽은 컬러 영상은 먼저 BGR에서 RGB로 바꾼다.

아래는 `model(...)`에 전달하는 순서다. `B`는 배치의 시점 수, `N`은 시점당 카메라 수다. 여섯 텐서는 모두 `float32`로 만들고 같은 카메라 순서로 쌓는다. 영상 한 장도 `B=1, N=1` 두 차원을 유지한다.

| 인자 | 텐서 크기 | 내용 |
|---|---|---|
| `imgs` | `(B, N, 3, 128, 352)` | 정규화한 RGB 영상 |
| `rots` | `(B, N, 3, 3)` | 카메라 → 자차 회전 |
| `trans` | `(B, N, 3)` | 카메라 → 자차 이동(m) |
| `intrins` | `(B, N, 3, 3)` | 전처리 전 영상에 대응하는 내부 파라미터 |
| `post_rots` | `(B, N, 3, 3)` | 영상 전처리의 선형 변환 |
| `post_trans` | `(B, N, 3)` | 영상 전처리의 픽셀 이동 |

전처리 구현은 [src/data.py](./src/data.py)의 `NuscData.sample_augmentation`, `NuscData.get_image_data`와 [src/tools.py](./src/tools.py)의 `img_transform`, `normalize_img`에서 확인할 수 있다.

#### 3. 체크포인트 로드와 추론

아래는 새 로더에서 위 여섯 텐서를 준비한 뒤 호출하는 예시다. 경로는 `lift-splat-shoot/` 기준이다. 다른 PC에서도 이 저장소의 모델 코드와 의존성, `weights/model525000.pt` 파일을 준비한다. 현재 모델 생성자는 EfficientNet-B0 초기 가중치도 읽으므로 첫 실행 시 다운로드 연결 또는 해당 가중치의 로컬 캐시가 필요하다.

```python
import torch
from src.models import compile_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}
model = compile_model(grid_conf, {"final_dim": (128, 352)}, outC=1)
model.load_state_dict(torch.load("weights/model525000.pt", map_location="cpu"))
model.to(device).eval()

inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
with torch.inference_mode():
    logits = model(*(tensor.to(device) for tensor in inputs))
    vehicle_prob = logits.sigmoid().cpu()
```

#### 4. 출력 해석과 첫 프레임 검증

`vehicle_prob[b, 0]`은 `200 × 200` 차량 점유 확률 지도다. 두 축 모두 자차 기준 −50 m부터 +50 m 직전까지를 0.5 m 간격으로 나눈다. 배열의 첫 인덱스는 자차 x(전방), 두 번째는 y(왼쪽)다. 예를 들어 정면 10 m, 좌우 0 m 지점은 `[120, 100]` 칸에 해당한다. 이미지를 그릴 때에는 전방과 왼쪽이 화면의 어느 방향인지 축에 표시한다.

처음 적용할 때는 **위치를 아는 차량이 보이는 프레임**으로 확인한다. 차량이 정면에 있는데 점유 영역이 옆이나 뒤에 생기면 외부 파라미터의 방향과 축 정의를 먼저 점검한다. 여러 카메라의 공통 시야에서 한 차량이 두 위치로 갈라지면 영상과 보정값의 카메라 ID, 동기화, 리사이즈·크롭 보정 순서로 확인한다.

새 카메라의 시야각, 설치 높이, 촬영 환경에 따라 예측 품질은 달라진다. 정확도를 IoU로 비교하려면 동일 시점의 차량 바닥면 정답을 같은 자차 좌표와 격자에 표시하고 예측 마스크(`vehicle_prob > 0.5`)와 비교한다. 전방 한 대만 사용하는 실험에서는 전체 격자와 카메라 시야 내부 중 어느 영역에서 IoU를 계산했는지도 기록한다. 기존 정답 생성 방식은 [src/data.py](./src/data.py)의 `get_binimg`에서 확인할 수 있다.

### Visualize Input/Output Data (optional)
Run a visual check to make sure extrinsics/intrinsics are being parsed correctly. Left: input images with LiDAR scans projected using the extrinsics and intrinsics. Middle: the LiDAR scan that is projected. Right: X-Y projection of the point cloud generated by the lift-splat model. Pass `--viz_train=True` to view data augmentation.

```
python main.py lidar_check mini/trainval --dataroot=NUSCENES_ROOT --viz_train=False
```
<img src="./imgs/check.gif">

### Train a model (optional)
Train a model. Monitor with tensorboard.

```
python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
tensorboard --logdir=./runs --bind_all
```
