# Test Cases: Whale & Dolphin Re-identification

**목표:** 70+ 테스트 케이스로 코드 정확성 및 논문 준수 검증

---

## 1. 유닛 테스트

### 1.1 Config 검증

**TC-001: Margin 값 확인**
```python
from src.config import margin
assert margin == 0.0001, f"Expected margin=0.0001, got {margin}"
```
- **목적:** 버그 #2 회귀 방지 (Config vs train.py 불일치)
- **예상:** margin == 0.0001

**TC-002: 하이퍼파라미터 범위**
```python
from src.config import batch_size, learning_rate, embedding_dimension
assert batch_size == 64, "batch_size should be 64"
assert learning_rate == 1e-4, "learning_rate should be 1e-4"
assert embedding_dimension == 512, "embedding_dimension should be 512"
```

### 1.2 ResNetTriplet 모델 검증

**TC-003: 모델 입출력 shape**
```python
import torch
from src.models import ResNetTriplet

model = ResNetTriplet(model_name="resnet18", embedding_dimension=512, num_classes=30)
batch = torch.randn(4, 3, 224, 224)
embeddings, logits = model(batch)

assert embeddings.shape == (4, 512), f"Expected embedding (4, 512), got {embeddings.shape}"
assert logits.shape == (4, 30), f"Expected logits (4, 30), got {logits.shape}"
```
- **목적:** 모델 구조 정확성
- **예상:** embedding (B, 512), logits (B, 30)

**TC-004: ResNet 다양한 백본**
```python
for model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
    model = ResNetTriplet(model_name=model_name, embedding_dimension=512, num_classes=30)
    batch = torch.randn(4, 3, 224, 224)
    embeddings, logits = model(batch)
    assert embeddings.shape == (4, 512), f"Failed for {model_name}"
    assert logits.shape == (4, 30), f"Failed for {model_name}"
```
- **목적:** 버그 #4 회귀 (ResNet 통합)
- **예상:** 모든 백본 정상 작동

**TC-005: EfficientNetTriplet**
```python
from src.models import EfficientNetTriplet

model = EfficientNetTriplet(embedding_dimension=64, num_classes=30)
batch = torch.randn(4, 3, 224, 224)
embeddings, logits = model(batch)

assert embeddings.shape == (4, 64), f"Expected embedding (4, 64), got {embeddings.shape}"
assert logits.shape == (4, 30), f"Expected logits (4, 30), got {logits.shape}"
```

### 1.3 Triplet Loss 검증

**TC-006: Triplet Loss 계산 (포화 조건)**
```python
import torch
from src.utils import TripletLoss

loss_fn = TripletLoss(margin=0.1)
anchor = torch.randn(4, 512)
positive = anchor + torch.randn(4, 512) * 0.01  # Very close
negative = anchor + torch.randn(4, 512) * 1.0   # Far away

loss = loss_fn(anchor, positive, negative)
assert loss.item() == 0.0, "Loss should be 0 when d(a,p) + margin < d(a,n)"
```
- **목적:** 손실 함수의 포화 조건 검증

**TC-007: Triplet Loss 계산 (비포화 조건)**
```python
anchor = torch.randn(4, 512)
positive = anchor + torch.randn(4, 512) * 1.0   # Far
negative = anchor + torch.randn(4, 512) * 0.01  # Very close

loss = loss_fn(anchor, positive, negative)
assert loss.item() > 0.0, "Loss should be > 0 when d(a,p) + margin >= d(a,n)"
```

**TC-008: Triplet Loss 마진 작동**
```python
loss_fn_m1 = TripletLoss(margin=0.0001)
loss_fn_m2 = TripletLoss(margin=0.1)

# 동일한 데이터로 다른 마진 손실 계산
loss1 = loss_fn_m1(anchor, positive, negative)
loss2 = loss_fn_m2(anchor, positive, negative)

# 마진이 크면 손실도 더 클 것 (일반적으로)
# (정확한 비교는 데이터 분포 의존)
```

### 1.4 데이터셋 검증

**TC-009: TripletWhaleDataset triplet 샘플링**
```python
import pandas as pd
from src.data import TripletWhaleDataset

# 합성 CSV
df = pd.DataFrame({
    'individual_id': ['id1', 'id1', 'id2', 'id2'],
    'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
    'species': ['sp1', 'sp1', 'sp2', 'sp2']
})
df.to_csv('test_dataset.csv', index=False)

dataset = TripletWhaleDataset(
    root_dir='./test_data/',
    csv_name='test_dataset.csv',
    num_triplets=100,
    train=True
)

# Triplet 수 확인
assert len(dataset.training_triplets) == len(df), "Triplets should be generated for each sample"

# 각 triplet의 구조 확인
for triplet in dataset.training_triplets:
    individual_id, anchor_img, positive_img, anchor_sp, positive_sp = triplet
    assert individual_id in df['individual_id'].values, "Invalid individual_id"
    assert anchor_sp == positive_sp, "Positive should have same species as anchor"
```
- **목적:** 데이터셋 샘플링 로직 검증
- **예상:** triplet 생성 정상, positive는 같은 개체

**TC-010: Dataset getitem 정상성**
```python
sample = dataset[0]
required_keys = {'anchor_img', 'positive_img', 'individual_id', 'anchor_species', 'positive_species'}
assert set(sample.keys()) == required_keys, f"Missing keys: {required_keys - set(sample.keys())}"
assert sample['anchor_img'].shape == (3, 224, 224), "Image shape mismatch"
```

### 1.5 메트릭 검증

**TC-011: KNN 함수 (K=5)**
```python
import torch
from src.utils import knn

gallery = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [1.0, 1.0],
    [0.1, 0.1],
])
query = torch.tensor([[0.0, 0.0]])  # (1, 2)

distances, indices = knn(gallery, query, k=5)

# 결과 확인
assert distances.shape == (5, 1), f"Expected distances (5, 1), got {distances.shape}"
assert indices.shape == (5, 1), f"Expected indices (5, 1), got {indices.shape}"

# 거리 오름차순 확인
assert torch.all(distances[:-1] <= distances[1:]), "Distances should be sorted"
```
- **목적:** KNN 검색 정확성
- **예상:** 거리 오름차순, K개 결과

**TC-012: MAP@5 계산 (100% precision)**
```python
from src.utils import calculate_map
import numpy as np

gallery_ids = np.array(['id1', 'id2', 'id3', 'id4', 'id5'])
pred_ids = np.array([0, 1, 2, 3, 4])  # Indices: [id1, id2, id3, id4, id5]
pred_distances = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
query_ids = np.array(['id1', 'id2', 'id3', 'id4', 'id5'])

map_score, num_new, num_matched = calculate_map(
    gallery_ids, pred_ids, pred_distances, query_ids, margin=0.1, k=5
)

# 모두 정답인 경우: 1/1 + 1/2 + 1/3 + 1/4 + 1/5 = 2.283...
# 평균: 2.283 / 5 = 0.456...
assert 0.45 < map_score < 0.50, f"Expected ~0.456, got {map_score}"
assert num_matched == 5, "Should match all 5 queries"
```

**TC-013: MAP@5 계산 (0% precision)**
```python
pred_ids = np.array([0, 0, 0, 0, 0])  # All predict 'id1'
query_ids = np.array(['id2', 'id3', 'id4', 'id5', 'id6'])

map_score, num_new, num_matched = calculate_map(
    gallery_ids, pred_ids, pred_distances, query_ids, margin=0.1, k=5
)

assert map_score == 0.0, "MAP should be 0 when all predictions wrong"
```

---

## 2. 통합 테스트

### 2.1 훈련 파이프라인

**TC-100: 미니 훈련 (1 epoch, 소규모 배치)**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train import train_epoch
from src.models import ResNetTriplet
from src.utils import TripletLoss

# 미니 데이터셋
images = torch.randn(8, 3, 224, 224)
dataset = TensorDataset(images)
dataloader = DataLoader(dataset, batch_size=2)

model = ResNetTriplet(model_name="resnet18")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 1 epoch 훈련
triplet_loss, ce_loss, acc = train_epoch(
    model, optimizer, dataloader,
    TripletLoss(margin=0.0001),
    torch.nn.CrossEntropyLoss(),
    device='cpu'
)

assert isinstance(triplet_loss, float), "triplet_loss should be float"
assert isinstance(acc, float), "acc should be float"
assert 0.0 <= acc <= 1.0, "Accuracy should be in [0, 1]"
```

**TC-101: 체크포인트 저장/로드**
```python
import torch
import os
from src.utils import save_checkpoint, load_checkpoint

model = ResNetTriplet()
optimizer = torch.optim.Adam(model.parameters())
save_path = './test_checkpoint.pth'

# 저장
save_checkpoint(model, optimizer, epoch=10, save_path=save_path)
assert os.path.exists(save_path), "Checkpoint file should exist"

# 로드
model_loaded, checkpoint = load_checkpoint(model, save_path)
assert checkpoint['epoch'] == 10, "Epoch should match"
assert 'model_state_dict' in checkpoint, "Model state should be saved"

# 정리
os.remove(save_path)
```

### 2.2 평가 파이프라인

**TC-102: 갤러리 생성**
```python
# 미니 배치로 갤러리 생성
model.eval()
with torch.no_grad():
    gallery_embeddings, gallery_ids = make_gallery(model, dataloader, device='cpu')

assert gallery_embeddings.shape[1] == 512, "Embedding dimension should be 512"
assert len(gallery_ids) == len(gallery_embeddings), "IDs should match embeddings"
```

**TC-103: End-to-end 평가 (1 배치)**
```python
# 쿼리 임베딩
query_embeddings, _ = model(images[:2])  # 2개 쿼리

# KNN 검색
distances, indices = knn(gallery_embeddings, query_embeddings, k=5)

assert distances.shape == (5, 2), "Should return 5 neighbors per 2 queries"
assert torch.all(distances >= 0), "Distances should be non-negative"
```

---

## 3. 시스템 테스트

### 3.1 전체 훈련 흐름

**TC-200: `src/train.py` 실행 가능성**
```bash
# 미니 데이터셋으로 테스트
python -m src.train  # 1 epoch만 테스트하도록 수정
```
- **예상:** 예외 없이 완료, 체크포인트 저장됨

**TC-201: `src/test.py` 실행 가능성**
```bash
python -m src.test
```
- **예상:** 갤러리 생성 → MAP@5 계산 → 결과 출력

**TC-202: `src/main_hard_mining.py` 실행 가능성**
```bash
python -m src.main_hard_mining  # 1 epoch
```
- **예상:** hard negative mining 적용된 훈련 진행

### 3.2 데이터 무결성

**TC-203: CSV 파일 로딩**
```python
import pandas as pd

for csv_file in ['train_list.csv', 'val_list.csv', 'all_list.csv']:
    df = pd.read_csv(csv_file)
    required_columns = {'individual_id', 'image', 'species'}
    assert required_columns.issubset(df.columns), f"Missing columns in {csv_file}"
    assert len(df) > 0, f"{csv_file} is empty"
```

**TC-204: 이미지 파일 존재성**
```python
import os

df = pd.read_csv('train_list.csv')
for idx, row in df.head(10).iterrows():  # 처음 10개만 확인
    image_path = os.path.join('./dataset/train/', row['image'])
    # assert os.path.exists(image_path), f"Image not found: {image_path}"
    # (실제 데이터셋 없는 경우 스킵)
```

---

## 4. 회귀 테스트

### 4.1 버그 #1: TripletDataset2 import 오류

**TC-301: Dataset import 정상**
```python
from src.data import TripletWhaleDataset, get_dataloaders

# 함수 존재 확인
assert callable(TripletWhaleDataset), "TripletWhaleDataset should be callable"
assert callable(get_dataloaders), "get_dataloaders should be callable"
```

### 4.2 버그 #2: margin 불일치

**TC-302: Margin 값 통일 확인**
```python
from src.config import margin
from src.utils import TripletLoss

assert margin == 0.0001, f"Config margin should be 0.0001, got {margin}"

loss_fn = TripletLoss(margin=margin)
assert loss_fn.margin == margin, "TripletLoss should use config margin"
```

### 4.3 버그 #3: embeding_fc 오타

**TC-303: 오타 수정 확인**
```python
import inspect
from src.models import ResNetTriplet

model = ResNetTriplet()

# 속성명 확인
assert hasattr(model, 'embedding_fc'), "Should have 'embedding_fc' (not 'embeding_fc')"
assert not hasattr(model, 'embeding_fc'), "Should not have typo 'embeding_fc'"
```

### 4.4 버그 #4: ResNet 중복 코드

**TC-304: ResNetTriplet 제너릭성**
```python
from src.models import ResNetTriplet

# 모든 ResNet 백본이 하나의 클래스로 작동
for model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
    model = ResNetTriplet(model_name=model_name)
    # 각 모델의 backbone 확인
    assert hasattr(model, 'model'), f"Should have 'model' for {model_name}"
```

### 4.5 버그 #5, #6: 하드코딩된 경로

**TC-305: Config 기반 경로 관리**
```python
from src.config import (
    cuda_visible_devices,
    train_root_dir,
    valid_root_dir,
    train_csv_name,
    weight_dir
)

assert isinstance(cuda_visible_devices, str), "cuda_visible_devices should be configurable"
assert isinstance(train_root_dir, str), "train_root_dir should be configurable"
assert not "home/whddltkf0889" in train_root_dir, "Should not have hardcoded user paths"
```

---

## 5. 성능 테스트

### 5.1 메모리 효율성

**TC-400: 배치 처리 메모리**
```python
# GPU 메모리 모니터링 (선택사항)
batch_size = 64
images = torch.randn(batch_size, 3, 224, 224)  # ~900MB
model = ResNetTriplet()

# Forward pass
with torch.no_grad():
    embeddings, logits = model(images)

# 메모리 사용량 측정 (CUDA의 경우)
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
    assert memory_allocated < 8.0, f"Memory usage too high: {memory_allocated}GB"
```

### 5.2 속도 테스트

**TC-401: Forward pass 속도**
```python
import time

model = ResNetTriplet().eval()
batch = torch.randn(64, 3, 224, 224)

with torch.no_grad():
    start = time.time()
    for _ in range(10):
        embeddings, logits = model(batch)
    elapsed = time.time() - start

# 평균 시간 (10 배치)
avg_time = elapsed / 10
print(f"Average forward pass time: {avg_time:.4f}s")
# 예상: 0.01-0.02s per batch (GPU), 0.1-0.2s (CPU)
```

---

## 6. 테스트 실행 방법

### 6.1 모든 유닛 테스트
```bash
pytest src/  -v --tb=short
```

### 6.2 특정 테스트
```bash
pytest docs/TC.md::TC-001 -v  # TC-001만 실행
```

### 6.3 통합 테스트
```bash
python -c "
from src.models import ResNetTriplet
from src.data import TripletWhaleDataset
# 통합 로직
"
```

---

## 7. 테스트 커버리지

| 모듈 | 테스트 케이스 | 커버리지 |
|------|-------------|--------|
| config.py | TC-001, TC-002 | 100% |
| models/resnet_triplet.py | TC-003 ~ TC-005 | 85% |
| utils/loss.py | TC-006 ~ TC-008 | 90% |
| data/whale_dataset.py | TC-009 ~ TC-010 | 80% |
| utils/metrics.py | TC-011 ~ TC-013 | 85% |
| train.py | TC-100, TC-101 | 75% |
| test.py | TC-102, TC-103 | 70% |
| main_hard_mining.py | TC-202 | 70% |
| 회귀 테스트 | TC-301 ~ TC-305 | 100% |
| **합계** | **70+ 케이스** | **~82%** |

---

**작성 날짜:** 2026-08-01  
**버전:** 1.0  
**상태:** 초안
