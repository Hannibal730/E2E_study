## 0. E2E 자율주행 패러다임의 이해 (Overview)

> **목표:** 기존 모듈형(Modular) 파이프라인에서 End-to-End로 넘어가는 학계의 거시적인 흐름과 주요 챌린지들을 파악하기.

- [ ]  `[ 논문 읽기 ]`
    - **E2E Survey:** https://arxiv.org/abs/2306.16927

- [ ]  `[ 흐름 정리 ]` **모듈형 vs E2E 아키텍처 비교 요약:**
    - 인지, 판단, 제어 모듈이 분리된 기존 Rule-based/모듈형 방식과 통합된 E2E 방식의 장단점(해석 가능성, 누적 오차 측면 등) 비교표 작성.
    - Open-loop 평가(데이터셋 기반)와 Closed-loop 평가(시뮬레이터 기반)의 차이점 및 Closed-loop가 E2E에서 필수적인 이유 요약.

## 1. 모방 학습의 한계와 극복 (Behavioral Cloning & DAgger)

> **목표:** 순수 BC 모델과 오프라인 복구 데이터(`left/right` 보정)를 적용한 BC 모델을 구축하여 OOD(Out-of-Distribution) 환경에서의 성능을 정량적으로 비교하고, 오프라인 보정의 한계를 통해 DAgger의 필요성을 도출합니다.

- [ ]  **`[ ]` [ 논문 읽기 ] DAgger (Ross et al., 2011)**
    - 링크: https://arxiv.org/abs/1011.0686
    - 핵심 포인트: 순수 BC에서 오차가  $O(\epsilon T^2)$로 누적되는 이유(Covariate Shift)와 데이터 병합(Aggregation) 개념 파악.

- [ ]  **`[ ]` [ 데이터 파이프라인 ] 시간순 분할 및 대조군 데이터셋 구축**
    - Udacity Self-Driving Simulator 데이터셋 다운로드

        https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset

    - **분할 전략:** `track1data`를 시간(인덱스) 순서대로 80%(Train) / 20%(Val) 분할 (랜덤 셔플 분할 금지), `track2data`(또는 급커브 구간)를 OOD Test Set으로 설정.
    - **Train Set 2종 구성:**
        - **Dataset A (Vanilla BC):** `center` 이미지 및 원본 `steering`만 사용.
        - **Dataset B (Augmented BC):** `center` + `left(steering + 0.2)` + `right(steering - 0.2)` 사용.
    - **Val / Test Set:** 평가 기준 왜곡 방지를 위해 두 실험 모두 **`center` 이미지와 원본 `steering`만** 사용.

- [ ]  **`[ ]` [ 모델 학습 ] PyTorch ResNet-18 기반 대조 실험**
    - Model: `torchvision.models.resnet18(weights='DEFAULT')` 로드 후 `model.fc = nn.Linear(model.fc.in_features, 1)` 수정.
    - Loss: `nn.MSELoss()`, Optimizer: `Adam` 등 하이퍼파라미터를 동일하게 통제.
    - Dataset A와 B로 각각 **Model A(Vanilla)**, **Model B(Augmented)** 학습 후 Validation MSE 기준 Best Checkpoint 저장.

- [ ]  **`[ ]` [ 시각화 및 정량 평가 ] OOD Covariate Shift 비교 분석**
    - 동일한 OOD Test Set(`track2data` `center` 전용)에서 Model A와 Model B를 각각 추론.
    - `plt.hist(alpha=0.5)`로 **Ground Truth(전문가) vs Model A vs Model B**의 조향각 분포를 겹쳐 시각화 (서브플롯 2개로 비교하면 가독성 우수).
    - 두 모델 각각의 **MSE / MAE** 및 **KL Divergence** 수치 비교 테이블 작성.

- [ ]  **`[ ]` [ 수식 및 결론 정리 ] 오프라인 보정의 한계와 DAgger의 필연성**
    - 수식 정리:  $\pi_i = \arg\min_{\pi} \sum \mathbb{E}_{s \sim d_{\pi_{i-1}}} [L(\pi(s), \pi^*(s))]$
    - **비교 고찰:** Model B가 Model A보다 OOD에서 개선된 이유(오프라인 복구 행동 학습)와, 그럼에도 여전히 실패하는 이유(임의의 `±0.2` 휴리스틱 라벨의 부정확성, 새로운 트랙/급커브 분포 미포함)를 정리하고, 이를 모델이 직접 겪은 상태 $s \sim d_{\pi_{i-1}}$에 전문가 $\pi^*$가 재라벨링하는 DAgger 방식과 대비시켜 마무리.

## 2. 2D to 3D 투영과 BEV 기초 (LSS & BEVFormer)

> **목표:** 다중 카메라 이미지를 자율주행 차량 중심의 BEV(Bird's Eye View) 공간으로 투영하는 최신 메커니즘 2가지를 수학적으로 분해합니다.

- [ ]  `[ 논문 읽기 ]`
    - **LSS (Lift, Splat, Shoot):** https://arxiv.org/abs/2008.05711
    - **BEVFormer:** https://arxiv.org/abs/2203.17270

- [ ]  `[ 차원 추적 ]` **LSS (Lift, Splat, Shoot) Tensor 변환 리포트:**
    - **Lift:** 입력 `[B, C, H, W]` → 카메라 Extrinsic $E$, Intrinsic $K$ 행렬 적용 → Depth 예측 네트워크 통과 → `[B, D (Depth bins), C, H, W]` 생성. (이때 $D=41$ 등 하이퍼파라미터 명시).
    - **Splat & Shoot:** Depth 정보를 3D Point Cloud 공간으로 던진 후(Shoot), Voxel Pooling을 통해 높이(Z축)를 압축하여(Splat) 최종 BEV Feature `[B, C, X, Y]`가 되는 과정을 텐서 Shape 변화와 함께 작성.

- [ ]  `[ 알고리즘 해부 ]` **BEVFormer Spatial Cross-Attention 다이어그램:**
    - Deformable DETR의 Reference Point 기반 샘플링 개념 요약.
    - BEV Grid 상의 한 Query $(x,y)$가 3D 상의 기둥(Pillar)으로 올라가고, 이 기둥이 다중 카메라 화면(2D)으로 역투영(Reprojection)되어 특정 픽셀 위치의 Feature만 쏙쏙 뽑아오는 과정(Spatial Cross-Attention)을 Draw.io로 그리고 첨부.

## 3. Transformer 기반 이종 센서 퓨전 (TransFuser)

> **목표:** 카메라(RGB)와 LiDAR(Point Cloud) 데이터를 융합할 때, Cross-Attention이 휴리스틱(Concat 등) 방식을 어떻게 압도하는지 코드 단에서 확인합니다.

- [ ]  `[ 논문 읽기 ]`
    - **TransFuser:** https://arxiv.org/abs/2205.15997

- [ ]  `[ 코드 뜯어보기 ]` **TransFuser Fusion 구조 분석:**
    - GitHub `transfuser` 레포지토리의 `models/transfuser.py` 확인.
    - `TransFuser(nn.Module)` 클래스 내의 `forward` 함수를 분석하여, Image와 LiDAR 피처맵이 ResNet의 어떤 해상도 단계(e.g., 64x64, 32x32, 16x16)에서 추출되어 `TransformerBlock`의 $Q,K,V$ 연산으로 들어가는지 코드 라인을 복사하여 주석 작성.

- [ ]  `[ 스크립트 작성 ]` **Attention Map 시각화 로직 구축:**
    - PyTorch의 `hook` (ex: `register_forward_hook`) 기능을 이용하여 Transformer Block의 Attention 가중치(Weight) 텐서를 추출하는 코드 스니펫 작성.
    - 이미지 위에 Heatmap을 덧씌우는 `cv2.addWeighted` 함수 활용 예제 작성.

## 4. E2E 네트워크의 Output(Head) 설계 (TCP)

> **목표:** E2E 네트워크 출력을 '궤적(Waypoint)'으로 할지, '직접 제어(Action)'로 할지 결정하는 TCP 프레임워크의 Loss 설계를 분해합니다.

- [ ]  `[ 논문 읽기 ]`
    - **TCP:** https://arxiv.org/abs/2206.08129

- [ ]  `[ 수식 분석 ]` **TCP Loss Function 정리:**
    - 궤적 브랜치 $\mathcal{L}_{traj} = \sum \Vert{} \hat{W} - W_{GT} \Vert{}_1$ (L1 Loss)
    - 제어 브랜치  $\mathcal{L}_{ctrl} = \Vert{} \hat{a} - a_{GT} \Vert{}_1 + \Vert{} \hat{s} - s_{GT} \Vert{}_1$
    - 두 출력을 상황에 따라 스위칭하는 Gating 확률 변수 $\alpha$에 따른 Total Loss 통합 수식 정리.

- [ ]  `[ 비교 문서 ]` **제어 방식 장단점 비교표 작성:**
    - **Waypoint 방식:** PID 제어기 사용에 따른 Jerk(승차감) 이점, 동역학 한계 극복 여부, 제어 지연(Latency).
    - **Direct Control 방식:** End-to-End 최적화 가능성, 하드웨어 이식성.

## 5. Unified 파이프라인 해부 (UniAD)

> **목표:** 인지-예측-계획이 분절되지 않고 'Query'를 통해 한 번에 최적화되는 SOTA 논문(UniAD)의 아키텍처를 해부합니다.

- [ ]  `[ 논문 읽기 ]`
    - **UniAD:** https://arxiv.org/abs/2212.10156

- [ ]  `[ 아키텍처 매핑 ]` **Query Data Flow Chart (Draw.io):**
    - BEV Feature 생성 이후의 흐름 세분화.
    - `Track Query` (동적 객체) → `Map Query` (차선, 횡단보도 등 정적 객체) → `Motion Query` (미래 궤적) → `Ego Vehicle Query` (자차 계획).
    - 각 모듈이 다음 모듈로 어떤 텐서를 넘겨주는지 명시.

- [ ]  `[ 알고리즘 분석 ]` **Bipartite Matching 기반 Loss 구성:**
    - Hungarian Algorithm을 통해 예측한 Query와 Ground Truth 객체를 1:1로 매칭할 때 사용되는 Cost Matrix 수식 (Focal Loss + L1 Bounding Box Loss + GIoU Loss) 정리.

## 6. Vector 기반 인지와 프레임워크 한계점 (VAD)

> **목표:** Raster(픽셀 격자) 기반 인지의 연산 병목을 극복하는 Vector 방식을 이해하고, E2E가 넘어야 할 상용화 한계점을 분석합니다.

- [ ]  `[ 논문 읽기 ]`
    - **VAD (Vectorized Autonomous Driving):** https://arxiv.org/abs/2303.12077

- [ ]  `[ 데이터 분석 ]` **Raster vs Vector 연산량 비교표:**
    - `VAD` 논문을 바탕으로 차선/객체를 `Occupancy Grid (H x W)`로 예측할 때와 `Polyline (N개의 점 좌표)`로 예측할 때의 파라미터 수(M), GFLOPs, 추론 시간(FPS) 표 정리.

- [ ]  `[ 리서치 리포트 ]` **E2E 자율주행 3대 맹점 기술 리포트:**
    - **해석성(Interpretability):** 사고 시 왜 스티어링을 틀었는지 역추적할 방안.
    - **롱테일(Long-tail) 대응:** 데이터셋에 없는 기형적인 장애물 인식 실패 문제.
    - **온디바이스 최적화:** 차량용 AP(NVIDIA Orin 등) 탑재를 위한 모델 경량화(Pruning/Quantization) 방안.

## 7. BEV 기반 E2E 모델의 Planning 모듈 비판적 분석 (BEV-Planner)

> **목표:** UniAD와 VAD 등 최신 BEV 기반 모델들이 궤적(Trajectory)을 생성할 때 겪는 근본적인 한계를 파악하고, 복잡한 인지 모듈과 실제 제어 간의 인과성(Causality) 문제를 심층적으로 분석합니다.

- [ ]  `[ 논문 읽기 ]`
    - **BEV-Planner:** https://arxiv.org/abs/2312.03031

- [ ]  `[ 비판적 분석 ]`  **인지(Perception)와 계획(Planning)의 괴리 현상 요약:**
    - 논문에서 지적하는 문제점 요약: "복잡한 3D 인지 피처나 맵 정보보다 단순한 자차 상태(Ego-status, ex: 현재 속도/조향각)가 Planning 성능에 더 지배적인 영향을 미치고 있다"는 'Illusion(착각)' 현상 정리.

- [ ]  `[ 기술 진단 리포트 ]`  **진정한 E2E Planning을 위한 개선 방안 도출:**
    - 단순히 Open-loop(데이터셋 벤치마크) 상에서의 L2 오차(L2 Error)를 줄이는 것을 넘어, Closed-loop(실제 시뮬레이터 주행)에서 인지 결과물이 제어에 유의미하게 반영되도록 강제하는 Loss 설계 아이디어 또는 향후 연구 방향 요약.

## 8. CARLA Server & Client Docker 환경 세팅

> **목표:** 스터디원 전원이 '로컬 종속성 충돌 없이' 완벽히 동일한 시뮬레이터와 학습 환경을 구축합니다.

- [ ]  `[ 도커 세팅 ]` **CARLA Server Docker Run 스크립트 작성:**
    - 아래 스크립트 작성 및 구동 테스트 (`docker ps` 및 포트 2000 확인).

        ```
        docker run -d --name carla-server \
          -p 2000-2002:2000-2002 --gpus all \
          carlasim/carla:0.9.10.1 \
          ./CarlaUE4.sh -RenderOffScreen
        ```

- [ ]  `[ 도커 세팅 ]` **Client 모델 구동용 `Dockerfile` 작성:**
    - `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04` 베이스.
    - `requirements.txt` (torch, torchvision, carla, opencv-python 등) 설치 명령어 추가.

- [ ]  `[ 도커 세팅 ]` **통합 `docker-compose.yml` 작성:**
    - 네트워크 모드를 `host`로 설정하거나 브릿지 네트워크로 두 컨테이너를 연결하고, 로컬의 E2E 모델 코드를 Client 컨테이너의 Volume으로 마운트하는 설정 파일 작성.

## 9. 모델 검증 및 응용 실험 (Mini Project)

> **목표:** CARLA 벤치마크 상에서 모델을 구동하고, 코드를 직접 수정하여 날씨 변화나 모델 경량화에 따른 성능 변화를 기록합니다.

- [ ]  `[ 시뮬레이션 ]` **Closed-loop 추론 및 지표 로깅:**
    - TransFuser 또는 TCP 모델 로드 후 CARLA `Town01` 주행 스크립트 실행.
    - 최종 산출 지표인 **Driving Score (DS)** = Route Completion (RC) $\times$ Infraction Penalty (IP) 로그 파일 확보.
