# AICV03_Missing_line_segmentation_ensemble_model
Cityscapes에서 LOID(논문) 스타일 occlusion-inpainting → segmentation ensemble 구현
목표: 앞차에 가려진 도로 영역 복원 IoU 개선 증명 (4일 MVP)

🎯 Project Goal
text
입력: Cityscapes 이미지 (1024×2048)
출력: 
1. baseline road seg (P_A) 
2. occlusion-inpainting → seg (P_B)
3. ensemble (occlusion 영역 B 우선) → IoU_occ↑ 증명
핵심 아이디어: LOID 논문(BDD100K/CULane) → Cityscapes 이식 [arXiv:2408.09117]


📋 Pipeline
[Image] → YOLOv8(occlusion mask) → LaMa inpaint → Seg(DeepLabV3+) → Ensemble
  ↓              ↓                    ↓                ↓             ↓
P_A ← 원본이미지                   P_B ← inpainted    M_occ → rule-based
                                                 ↓
                                            P_final = (1-M)*P_A + M*P_B

📁 Folder Structure
```
loid_cityscapes/
├── datasets/cityscapes/           # gtFine + leftImg8bit_trainval
├── experiments/
│   ├── ckpts/                    # *.pth
│   ├── figs/                     # debug + report 이미지
│   └── results/                  # metrics.json
├── notebooks/                    # Colab *.ipynb
├── src/                          # *.py 모듈
└── reports/                      # final report.md
```

🚀 Quick Start (Colab)
```
# 1. Drive 마운트 + Clone
%cd /content/drive/MyDrive
!git clone https://github.com/choiy4432/AICV03_Missing_line_segmentation_ensemble_model.git
%cd AICV03_Missing_line_segmentation_ensemble_model

# 2. Requirements
!pip install -r requirements.txt

# 3. Cityscapes 경로 설정
DATA_ROOT = ""
```
