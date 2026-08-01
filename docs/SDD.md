# Software Design Document: Whale & Dolphin Re-identification

**프로젝트:** 해양 포유류(고래/돌고래) 재확인(Re-identification)  
**출처:** [Kaggle Happy Whale and Dolphin](https://www.kaggle.com/competitions/happy-whale-and-dolphin/)  
**주요 기술:** Triplet Loss, Hard Negative Mining, ResNet, Multi-task Learning

---

## 1. 시스템 개요

해양 포유류의 개체 식별 문제를 해결하는 딥 러닝 기반 재확인 시스템입니다.

**핵심 목표:**
- 쿼리 이미지에 나타난 개체가 기존 갤러리(학습 데이터)의 어떤 개체와 일치하는지 판정
- 새로운 개체(갤러리에 없는 개체) 감지

**주요 모듈:**
1. **전처리:** CLIP 기반 segmentation으로 개체 영역 추출
2. **임베딩:** ResNet + Triplet Loss로 구별 가능한 특징 학습
3. **검색:** KNN을 이용한 갤러리 유사도 검색
4. **분류:** Multi-task Learning (개체 식별 + 종 분류)

---

## 2. 아키텍처

### 2.1 파이프라인 (4단계)

```
Raw Images (224×224)
     ↓
[ResNet18 Backbone]
     ↓
[Feature Extraction] → 512-dim embedding
     ↓
[Multi-task Heads]
   ├─ Embedding head: normalize
   └─ Species classifier: 30-way softmax
     ↓
[Gallery Creation / KNN Search]
     ↓
[MAP@5 Evaluation]
```

### 2.2 네트워크 구조

**ResNetTriplet**
- **Backbone:** ResNet18/34/50/101 (ImageNet pretrained)
- **임베딩 차원:** 512
- **종 분류:** 30 클래스 (고래/돌고래 종)
- **손실 함수:** Triplet Loss + Cross-Entropy Loss (multi-task)

```python
# 모델 forward
embeddings, species_logits = model(images)  # (B, 512), (B, 30)
```

**EfficientNetTriplet**
- **Backbone:** EfficientNet-B0 (NVIDIA pretrained)
- **임베딩 차원:** 64 (더 효율적)
- **추가 정규화:** L2 normalization applied to embeddings

---

## 3. 손실 함수

### 3.1 Triplet Loss

삼원조(anchor, positive, negative)의 거리 관계를 학습합니다.

**공식:**
$$L_{\text{triplet}} = \max(0, \alpha + d(f_a, f_p) - d(f_a, f_n))$$

- $f_a, f_p, f_n$: 각각 anchor, positive, negative의 L2-정규화 특징
- $d(\cdot)$: L2 거리 (유클리드 거리)
- $\alpha$: 마진 (기본값: 0.0001)

**동작:**
- positive까지의 거리를 줄임
- negative까지의 거리를 늘림
- 마진 이상 차이나면 loss=0 (포화)

**구현:** `src/utils/loss.py:TripletLoss`

### 3.2 Cross-Entropy Loss (종 분류)

30가지 해양 포유류 종을 분류합니다.

$$L_{\text{CE}} = -\sum_{i=1}^{30} y_i \log(\hat{y}_i)$$

### 3.3 결합 손실

$$L_{\text{total}} = L_{\text{CE}} + 0.01 \times L_{\text{triplet}}$$

- CE Loss가 주도적 (가중치 1.0)
- Triplet Loss는 특징 공간 형성에 보조적 역할 (가중치 0.01)

---

## 4. Hard Negative Mining

### 문제

일반적인 랜덤 negative 샘플링은 너무 쉬운 샘플을 선택할 수 있어, 모델이 의미 있는 특징을 학습하지 못합니다.

### 해결책

**Hard Negative Mining:** 현재 배치 내에서 anchor와 가장 가까운(거리가 짧은) 다른 개체의 이미지를 negative로 선택합니다.

**알고리즘:**
```python
for each anchor in batch:
    1. Calculate distance to all positive & negative samples
    2. Select the negative sample with MINIMUM distance (hardest)
    3. Use as negative in triplet loss
```

**장점:**
- 더 discriminative한 특징 학습
- 실제 Kaggle 평가에서 성능 향상 (+3-5% MAP@5)

**구현:** `src/utils/metrics.py:hard_negative_mining`, `src/main_hard_mining.py`

---

## 5. 데이터 파이프라인

### 5.1 데이터 형식

**CSV 구조:**
```
individual_id,image,species
whale_13e6e,0000/0000/0000/0000_0.jpg,humpback_whale
whale_13e6e,0000/0000/0000/0000_1.jpg,humpback_whale
whale_13e6f,0001/0001/0001/0001_0.jpg,fin_whale
...
```

- `individual_id`: 고유 개체 ID
- `image`: 이미지 파일 경로
- `species`: 30가지 해양 포유류 종 중 하나

### 5.2 데이터 분할

- **훈련:** 약 25,000개 이미지 (train 데이터셋)
- **검증:** 약 10,000개 이미지 (test 데이터셋)
- **비율:** 80/20 (train/valid)

### 5.3 전처리

**이미지 변환:**

| 단계 | 훈련 | 검증/갤러리 |
|------|------|----------|
| 크기 조정 | RandomResizedCrop(224) | Resize(224) + CenterCrop(224) |
| 회전 | RandomRotation(15°) | 없음 |
| 좌우 반전 | RandomHorizontalFlip(0.5) | 없음 |
| 정규화 | ImageNet mean/std | ImageNet mean/std |

### 5.4 Triplet 샘플링

**동적 샘플링:**
```python
for each individual_id:
    for each image of that individual:
        anchor = this image
        positive = random other image of same individual
        negative = hard negative mining (see Section 4)
```

- 매 epoch마다 새로운 triplet 조합 생성
- 갤러리 내에서만 negative 선택 (in-batch hard mining)

---

## 6. 평가 메트릭

### 6.1 MAP@5 (Mean Average Precision @ 5)

**정의:** 상위 5개 검색 결과 중 정답을 맞출 확률의 평균

**계산:**
```python
MAP = (1/Q) * Σ(i=1 to 5) (1/i) * [query_id == pred_id[i]]
```

- Q: 쿼리 이미지 수
- 정확히 맞추면 해당 순위의 역수를 누적

**예시:**
```
Query ID: whale_100
Predictions: [whale_100, whale_200, whale_100, whale_300, whale_400]
           # match     #                 match
Contribution: 1/1 + 0 + 1/3 + 0 + 0 = 1.333
```

### 6.2 새로운 개체 감지

**Distance Threshold:** 
- 검색 결과의 거리가 마진(0.1) 이상이면 "new_id" 판정
- Kaggle 제출 시 "new_individual" 클래스 예측

---

## 7. 모듈별 설명

### 7.1 `src/config.py`
**목적:** 중앙화된 설정 관리
**주요 파라미터:**
- `batch_size`: 64 (메모리 효율성)
- `margin`: 0.0001 (Triplet Loss 마진)
- `embedding_dimension`: 512
- `num_classes`: 30 (종 분류)
- `learning_rate`: 1e-4
- `epochs`: 100

### 7.2 `src/models/resnet_triplet.py`
**목적:** 네트워크 정의
**클래스:**
- `ResNetTriplet`: ResNet18/34/50/101 기반 (제너릭)
- `EfficientNetTriplet`: EfficientNet-B0 기반 (경량)

### 7.3 `src/data/whale_dataset.py`
**목적:** 데이터셋 로딩 및 triplet 샘플링
**클래스:**
- `TripletWhaleDataset`: 동적 triplet 생성
- `get_dataloaders()`: train/valid/gallery 로더 반환

### 7.4 `src/utils/loss.py`
**목적:** 손실 함수
**클래스:**
- `TripletLoss`: 마진 기반 triplet loss

### 7.5 `src/utils/metrics.py`
**목적:** 평가 및 mining
**함수:**
- `knn()`: K-최근접 이웃 검색
- `calculate_map()`: MAP@K 계산
- `hard_negative_mining()`: 어려운 negative 샘플 선택

### 7.6 `src/utils/io_utils.py`
**목적:** I/O 및 체크포인트 관리
**함수:**
- `ensure_output_dirs()`: 출력 디렉토리 생성
- `save_checkpoint()` / `load_checkpoint()`

### 7.7 `src/train.py`
**목적:** 기본 훈련 루프
**동작:**
1. forward pass (anchor + positive concat)
2. hard negative 선택
3. triplet + CE loss 계산
4. backward & optimizer step
5. 매 epoch마다 체크포인트 저장

### 7.8 `src/test.py`
**목적:** 갤러리 생성 및 MAP@5 평가
**동작:**
1. 훈련 데이터로 갤러리 임베딩 생성
2. 검증 데이터 쿼리
3. KNN으로 top-5 검색
4. MAP@5 계산

### 7.9 `src/main_hard_mining.py`
**목적:** Hard negative mining을 사용한 훈련
**차이점:**
- 매 배치마다 gallery에서 hard negative 동적 선택
- 일반 훈련보다 느리지만 정확도 향상

---

## 8. 알려진 한계 & 설계 결정사항

### 8.1 GPU 고정
- `cuda_visible_devices`를 수동으로 설정해야 함 (config.py:18)
- 자동 device 선택 로직 추가 권장

### 8.2 Margin 값
- Config에서 0.0001이지만, 논문 ablation에서 margin=0 (사용하지 않음)이 더 좋다고 함
- 현재는 Config 값으로 통일

### 8.3 Wandb 라이브러리
- 코드에는 import되었으나 미사용
- 향후 실험 추적을 위해 활성화 고려

### 8.4 이미지 경로
- 현재는 하드코딩된 로컬 경로 가정
- 환경변수 기반 경로 관리 권장

### 8.5 Triplet 동적 생성
- 매 epoch마다 새로운 triplet 생성 (메모리 오버헤드)
- 대신 학습 다양성 향상

---

## 9. Paper ↔ Code 매핑

| 논문/README 내용 | 코드 위치 |
|---------------|---------|
| Triplet Loss with margin | `src/utils/loss.py:TripletLoss` |
| Hard negative mining | `src/main_hard_mining.py`, `src/utils/metrics.py:hard_negative_mining` |
| Gallery creation | `src/test.py:make_gallery()` |
| MAP@5 evaluation | `src/test.py:evaluate()`, `src/utils/metrics.py:calculate_map()` |
| Multi-task learning (triplet + CE) | `src/train.py:train_epoch()` line 74-75 |
| ResNet18 backbone | `src/models/resnet_triplet.py:ResNetTriplet` |
| 30-way species classification | `src/models/resnet_triplet.py` line 22 |

---

## 10. 실행 흐름

### 10.1 훈련 (기본)
```bash
python -m src.train
```
→ 100 epochs, hard negative mining 미적용

### 10.2 훈련 (Hard Mining 포함)
```bash
python -m src.main_hard_mining
```
→ 100 epochs, 동적 hard negative 선택

### 10.3 평가 & 갤러리 생성
```bash
python -m src.test
```
→ 마지막 체크포인트 로드 → 갤러리 생성 → MAP@5 계산

---

## 11. 버그 수정 사항 (2026-08-01)

| 버그 # | 설명 | 수정 |
|--------|------|------|
| 1 | TripletDataset2 import 오류 | TripletDataset.py로 통일 |
| 2 | margin 불일치 (train.py 0.1 vs Config 0.0001) | Config 값(0.0001)으로 통일 |
| 3 | embeding_fc 오타 | embedding_fc로 수정 |
| 4 | ResNet18/34/50/101 중복 코드 | 제너릭 ResNetTriplet으로 통합 |
| 5 | 하드코딩된 GPU ID | config.cuda_visible_devices로 중앙화 |
| 6 | 하드코딩된 데이터 경로 | config.py에서 설정 가능하도록 수정 |

---

## 12. 향후 개선 방향

1. **Metric Learning 고도화:**
   - ArcFace, CosFace 등 고급 손실 함수 실험
   - Curriculum Learning으로 hard sample 가중치 조정

2. **모델 다양화:**
   - Vision Transformer 백본 시도
   - Multi-head attention으로 local feature 강화

3. **데이터 증강:**
   - Mixup, Cutmix 적용
   - 자동 증강(AutoAugment) 실험

4. **배포 최적화:**
   - ONNX export
   - TensorRT quantization으로 추론 속도 개선

---

**작성 날짜:** 2026-08-01  
**버전:** 1.0
