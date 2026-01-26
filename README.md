# AICV03 Missing Line Segmentation Ensemble Model (앙차생)
**Occlusion-Aware Drivable Area Segmentation Pipeline**  
YOLO 기반 가림(occlusion) 탐지 → Inpainting 복구 → Segmentation → (Mask-aware) Ensemble로 **가려진 도로 영역 인식 성능을 개선**하는 프로젝트입니다.

> 핵심 목표: 자율주행 환경에서 차량/보행자 등으로 가려진 도로 영역을 감지·복구하여 더 정확한 주행 가능 영역(Drivable Area) 인식을 달성

---

## 1. 프로젝트 소개 (문제 정의 포함)

### 문제 정의
- Cityscapes의 `gtFine` 라벨은 **보이는 픽셀 중심으로 라벨링**되어, 실제 주행에서 자주 발생하는 **가림(occlusion)** 구간의 도로 정보가 충분히 반영되지 않습니다.
- 그 결과, 모델이 occlusion 구간에서 **도로 영역을 과소 예측**하고 주행 가능 영역 판단이 불안정해질 수 있습니다.

### 접근 방법 (Solution)
1) **Occluder 탐지**: YOLOv8로 차량/사람/자전거 등 가림 유발 객체를 탐지  
2) **Occlusion Mask 생성**: 탐지 결과를 픽셀 단위 마스크로 변환  
3) **Inpainting 복구**: Simple-LaMa로 가려진 배경(도로 텍스처)을 복구  
4) **Segmentation**: 원본 이미지와 복구 이미지 각각에서 도로 확률맵/마스크 생성  
5) **Mask-aware Ensemble**: occlusion 영역에서는 복구 기반 추정을 더 반영하여 결합

---

## 2. 주요 기능
- **Cityscapes Binary Road GT 생성**
  - `gtFine_labelIds.png`로부터 도로 계열(road/sidewalk/parking/rail track 등) 라벨을 **binary mask**로 변환
- **Baseline Segmentation (DeepLabV3+)**
  - RGB 이미지 → Road Probability Map 추정
- **Occlusion Detection (YOLOv8m)**
  - 타겟 클래스: `car, truck, person, bicycle, motorcycle, bus`
  - 출력: `{H, W}` occlusion mask (1.0=가려짐, 0.0=드러남)
- **Inpainting (Simple-LaMa)**
  - Original + Occlusion Mask → 가려진 영역 자연 복구
- **Mask-aware Ensemble**
  - 원본 기반 추정 `P_A`와 복구 기반 추정 `P_B`를 occlusion mask로 결합  
  - 예: `P_final = (1 - M_occ) * P_A + M_occ * P_B`

---

## 3. 기술 스택
- **Language**: Python
- **Framework**: PyTorch
- **Segmentation**
  - `segmentation_models_pytorch` (DeepLabV3+)
  - HuggingFace `transformers` (SegFormer)
- **Detection**: `ultralytics` (YOLOv8)
- **Inpainting**: `simple-lama-inpainting` (Simple-LaMa)
- **ETC**: OpenCV, NumPy, Pandas, Matplotlib, tqdm  
- **Execution**: Google Colab (CUDA) 중심

---

## 4. 시스템 아키텍처 (간단히)

mermaid
flowchart LR
  A[Original Image] --> B[YOLOv8 Occlusion Detection]
  B --> C[Occlusion Mask M_occ]
  A --> D[Baseline Seg P_A<br/>DeepLabV3+]
  A --> E[Inpainting<br/>Simple-LaMa]
  C --> E
  E --> F[Segmentation P_B<br/>SegFormer]
  D --> G[Mask-aware Ensemble]
  F --> G
  C --> G
  G --> H[Final Drivable Area Mask]

---

## 5. 성과 및 개선사항 (수치 포함)

### 정량 성과 (mIoU)
| Model | mIoU |
|---|---:|
| DeepLabV3+ (Baseline) | 0.4653 |
| SegFormer | 0.7899 |
| Ensemble | **0.7907** |

- Baseline → SegFormer: **+69.76%** (상대 개선)
- SegFormer → Ensemble: **+0.08%p** (소폭 추가 개선)

### 추론 속도 (평균, Stage-wise)
| Stage | Avg Latency |
|---|---:|
| Baseline Seg (DeepLabV3+) | ~44 ms |
| YOLO Occlusion Mask | ~23.4 ms (min 19.2 / max 43.4) |
| Inpainting (Simple-LaMa) | ~1138.6 ms |
| Segmentation (SegFormer) | ~67.1 ms |

- **병목(Bottleneck)**: Inpainting이 end-to-end latency 대부분을 차지 → 최적화 여지 큼

### 데이터/상황 분석
- 평균 Occlusion 비율: **약 13.29% ~ 14.56%**
- 관찰된 실패 케이스:
  - 조명/그림자 변화가 큰 환경에서 occlusion mask가 불안정한 경우
  - 복구 텍스처가 실제 도로 패턴과 어긋날 때 segmentation 오류가 발생하는 경우

---


## 6. 설치 및 실행 방법

> 본 레포는 **Notebook 기반 실행**을 기본으로 합니다. (Colab 권장)

### 6.1. Repository Clone
```bash
git clone https://github.com/choiy4432/AICV03_Missing_line_segmentation_ensemble_model.git
cd AICV03_Missing_line_segmentation_ensemble_model
```

### 6.2. Dependencies 설치
```bash
pip install -r requirements.txt
```

### 6.3. 실행 (Notebook)
`/notebook` 폴더에서 아래 순서로 실행을 권장합니다.

1. `00_setup.ipynb` : 환경 설정/경로/라이브러리 설치  
2. `01_cityscapes_seg_baseline.ipynb` : Baseline 학습/추론 및 평가  
3. `1_Final_pipeline.ipynb` : YOLO → Inpainting → Segmentation 파이프라인  
4. `1_Final_pipeline_ensemble.ipynb` : 최종 앙상블 및 성능 평가  

> Colab 사용 시, 노트북 내 데이터 경로(`PROJECT_ROOT` 등)를 본인 환경에 맞게 수정하세요.

---

## 7. 향후 계획

- Inpainting 병목 개선: ONNX/TensorRT, mixed precision, batch inference 적용 (**목표: 2×**)  
- Rule-based ensemble → uncertainty 기반 weighting / 학습 기반 gating으로 확장  
- 야간/악천후/저조도 환경에서 occlusion 강건성 강화  
- 엣지 디바이스(예: Jetson) 실시간 추론 목표(20+ FPS)

---

## Demo
<img width="880" height="544" alt="image" src="https://github.com/user-attachments/assets/10d40583-ac15-43d0-9a9d-2a04bf9de391" />


---

## Contributors
- Team 앙~상블
