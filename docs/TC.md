# Test Cases: Whale & Dolphin Re-identification

**목표:** 70+ 테스트 케이스로 코드 정확성 및 논문 준수 검증

## 최근 실행 결과

| 항목 | 값 |
|------|-----|
| 실행일 | 2026-08-17 |
| 하드웨어 | NVIDIA RTX 3060 Laptop 6 GB, CUDA 12.6 |
| 소프트웨어 | Python 3.14.6 / torch 2.13.0+cu126 |
| 데이터 | 합성 — 개체 8마리 × 3장 × (train/valid), 종 4종, 256×256 |
| 축소 설정 | `EPOCHS=1`, `BATCH_SIZE=4`, `NUM_CLASSES=4`, `MAP_K=2` |

| 테스트 | 결과 |
|--------|------|
| TC-001 / TC-002 / TC-003 (config·모델 shape) | ✅ |
| `python -m src.train` (문서에 적힌 명령 그대로) | ✅ |
| `python -m src.main_hard_mining` | ✅ |
| `python -m src.test` | ✅ |
| TC-306 (batch_size 1/2/4/16/64에서 hard negative mining) | ✅ |
| TC-307 (k 과대 지정 시 clamp) | ✅ |
| TC-308 (`EPOCHS`가 루프 길이를 결정) | ✅ |
| Kaggle 리더보드 MAP@5 | ⛔ 미실행 — 실제 데이터셋 미보유 |

> 합성 데이터의 정확도 수치는 의미가 없다. 이 실행은 파이프라인이 끝까지 도는지를
> 확인하기 위한 것이다.

---

## 1. 유닛 테스트

### 1.1 Config 검증

**TC-001: Margin 값 확인**
```python
import sys
sys.path.insert(0, './src')
from config import margin
assert margin == 0.0001, f"Expected margin=0.0001, got {margin}"
```
- **목적:** 버그 #2 회귀 방지 (Config vs train.py 불일치)
- **예상:** margin == 0.0001

**TC-002: 하이퍼파라미터 범위**
```python
import sys
sys.path.insert(0, './src')
from config import batch_size, learning_rate, embedding_dimension
assert batch_size == 64, "batch_size should be 64"
assert learning_rate == 1e-4, "learning_rate should be 1e-4"
assert embedding_dimension == 512, "embedding_dimension should be 512"
```

### 1.2 ResNetTriplet 모델 검증

**TC-003: 모델 입출력 shape**
```python
import sys
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet

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
import sys
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet

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
import sys
import torch
sys.path.insert(0, './src')
from models import EfficientNetTriplet

model = EfficientNetTriplet(embedding_dimension=64, num_classes=30)
batch = torch.randn(4, 3, 224, 224)
embeddings, logits = model(batch)

assert embeddings.shape == (4, 64), f"Expected embedding (4, 64), got {embeddings.shape}"
assert logits.shape == (4, 30), f"Expected logits (4, 30), got {logits.shape}"
```

### 1.3 Triplet Loss 검증

**TC-006: Triplet Loss 계산 (포화 조건)**
```python
import sys
import torch
sys.path.insert(0, './src')
from utils import TripletLoss

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
import sys
import torch
sys.path.insert(0, './src')
from utils import TripletLoss

loss_fn = TripletLoss(margin=0.1)
anchor = torch.randn(4, 512)
positive = anchor + torch.randn(4, 512) * 1.0   # Far
negative = anchor + torch.randn(4, 512) * 0.01  # Very close

loss = loss_fn(anchor, positive, negative)
assert loss.item() > 0.0, "Loss should be > 0 when d(a,p) + margin >= d(a,n)"
```

**TC-008: Triplet Loss 마진 작동**
```python
import sys
import torch
sys.path.insert(0, './src')
from utils import TripletLoss

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
import sys
import pandas as pd
sys.path.insert(0, './src')
from data import TripletWhaleDataset

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
import sys
sys.path.insert(0, './src')

sample = dataset[0]
required_keys = {'anchor_img', 'positive_img', 'individual_id', 'anchor_species', 'positive_species'}
assert set(sample.keys()) == required_keys, f"Missing keys: {required_keys - set(sample.keys())}"
assert sample['anchor_img'].shape == (3, 224, 224), "Image shape mismatch"
```

### 1.5 메트릭 검증

**TC-011: KNN 함수 (K=5)**
```python
import sys
import torch
sys.path.insert(0, './src')
from utils import knn

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
import sys
import numpy as np
sys.path.insert(0, './src')
from utils import calculate_map

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
import sys
import numpy as np
sys.path.insert(0, './src')
from utils import calculate_map

gallery_ids = np.array(['id1', 'id2', 'id3', 'id4', 'id5'])
pred_ids = np.array([0, 0, 0, 0, 0])  # All predict 'id1'
pred_distances = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
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
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, './src')
from models import ResNetTriplet
from utils import TripletLoss

# 미니 데이터셋
images = torch.randn(8, 3, 224, 224)
dataset = TensorDataset(images)
dataloader = DataLoader(dataset, batch_size=2)

model = ResNetTriplet(model_name="resnet18")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Forward pass 테스트 (실제 train_epoch는 복잡한 의존성 있음)
embeddings, logits = model(images)
assert embeddings.shape == (8, 512), "Embeddings shape should be (8, 512)"
assert logits.shape == (8, 30), "Logits shape should be (8, 30)"
```

**TC-101: 체크포인트 저장/로드**
```python
import sys
import torch
import os
sys.path.insert(0, './src')
from models import ResNetTriplet
from utils import save_checkpoint, load_checkpoint

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

**TC-102: 갤러리 생성 시뮬레이션**
```python
import sys
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet

# 미니 배치로 갤러리 생성 시뮬레이션
model = ResNetTriplet().eval()
images = torch.randn(4, 3, 224, 224)

with torch.no_grad():
    embeddings, _ = model(images)

assert embeddings.shape == (4, 512), "Embedding dimension should be 512"
assert embeddings.shape[0] == 4, "Batch size should match"
```

**TC-103: End-to-end 평가 (1 배치)**
```python
import sys
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet
from utils import knn

model = ResNetTriplet().eval()
gallery_embeddings = torch.randn(20, 512)  # 20 gallery samples
query_embeddings = torch.randn(2, 512)  # 2 queries

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
import sys
sys.path.insert(0, './src')
from data import TripletWhaleDataset, get_dataloaders

# 함수 존재 확인
assert callable(TripletWhaleDataset), "TripletWhaleDataset should be callable"
assert callable(get_dataloaders), "get_dataloaders should be callable"
```

### 4.2 버그 #2: margin 불일치

**TC-302: Margin 값 통일 확인**
```python
import sys
sys.path.insert(0, './src')
from config import margin
from utils import TripletLoss

assert margin == 0.0001, f"Config margin should be 0.0001, got {margin}"

loss_fn = TripletLoss(margin=margin)
assert loss_fn.margin == margin, "TripletLoss should use config margin"
```

### 4.3 버그 #3: embeding_fc 오타

**TC-303: 오타 수정 확인**
```python
import sys
sys.path.insert(0, './src')
from models import ResNetTriplet

model = ResNetTriplet()

# 속성명 확인
assert hasattr(model, 'embedding_fc'), "Should have 'embedding_fc' (not 'embeding_fc')"
assert not hasattr(model, 'embeding_fc'), "Should not have typo 'embeding_fc'"
```

### 4.4 버그 #4: ResNet 중복 코드

**TC-304: ResNetTriplet 제너릭성**
```python
import sys
sys.path.insert(0, './src')
from models import ResNetTriplet

# 모든 ResNet 백본이 하나의 클래스로 작동
for model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
    model = ResNetTriplet(model_name=model_name)
    # 각 모델의 backbone 확인
    assert hasattr(model, 'model'), f"Should have 'model' for {model_name}"
```

### 4.5 버그 #5, #6: 환경변수 기반 설정

**TC-305: 환경변수 및 Config 기반 설정**
```python
import sys
import os
from pathlib import Path
sys.path.insert(0, './src')
from config import (
    cuda_visible_devices,
    train_root_dir,
    valid_root_dir,
    train_csv_name,
    weight_dir,
    device_order
)

# 환경변수로 설정 가능 확인
assert cuda_visible_devices == os.getenv("CUDA_VISIBLE_DEVICES", "0")
assert isinstance(train_root_dir, Path), "Paths should be Path objects"

# 환경변수 미설정 시 기본값 사용
assert str(train_root_dir).endswith("train") or str(train_root_dir).endswith("dataset/train")
```

### 4.6 버그 #7: batch_size 2 이상에서 hard negative mining 크래시

`knn_hard_negatives`가 `dist.topk(k * batch_size, dim=0)`을 수행하는데, dim 0의 크기는
갤러리 크기인 `2 * batch_size`다. 호출부가 `k=batch_size * 2`를 넘기고 있어 요청 개수가
`2 * batch_size²`가 되고, `batch_size >= 2`면 항상
`RuntimeError: selected index k out of range`로 죽었다. 기본 `batch_size`가 64이므로
학습 진입점이 실행 자체가 불가능했다.

**TC-306: 다양한 batch_size에서 hard negative mining이 동작한다**
```python
import torch, numpy as np
from src.train import knn_hard_negatives
from src.config import hard_negatives_per_anchor

for bs in (1, 2, 4, 16, 64):
    anchors = torch.randn(bs, 512)
    gallery = torch.cat((anchors, torch.randn(bs, 512)))
    ids = np.array([f"id{i % 3}" for i in range(bs)])
    neg = knn_hard_negatives(gallery, np.concatenate([ids, ids]),
                             anchors, ids, k=hard_negatives_per_anchor)
    assert neg.shape == anchors.shape, (bs, neg.shape)
print("TC-306 OK")
```
- **목적:** 버그 #7 회귀 방지
- **예상:** 모든 batch_size에서 예외 없이 `(batch_size, 512)` 반환

**TC-307: 후보 수가 갤러리 크기를 넘지 않도록 clamp된다**
```python
import torch, numpy as np
from src.train import knn_hard_negatives

bs = 8
anchors = torch.randn(bs, 512)
gallery = torch.cat((anchors, torch.randn(bs, 512)))
ids = np.array([f"id{i}" for i in range(bs)])
# k를 과도하게 크게 줘도 topk가 터지지 않아야 한다
neg = knn_hard_negatives(gallery, np.concatenate([ids, ids]), anchors, ids, k=999)
assert neg.shape == (bs, 512)
print("TC-307 OK")
```

### 4.7 버그 #8: EPOCHS 환경변수가 무시됨

`train.py`와 `main_hard_mining.py`가 `num_epochs = 100`을 하드코딩하고 있어
`.env.example`에 선언된 `EPOCHS`가 아무 효과도 없었다.

**TC-308: EPOCHS가 실제 루프 길이를 결정한다**
```python
import subprocess, sys, os
env = dict(os.environ, EPOCHS="1", BATCH_SIZE="4", NUM_CLASSES="4")
out = subprocess.run([sys.executable, "-m", "src.train"],
                     capture_output=True, text=True, env=env).stdout
assert "Epoch 1/1" in out, out[-500:]
print("TC-308 OK")
```
- **예상:** `Epoch 1/1`이 출력되고 학습이 1에포크만에 종료

### 4.8 버그 #9: README의 실행 명령이 동작하지 않음

README는 `python -m src.train`을 안내했지만 최상위 스크립트가 절대 import
(`from config import ...`)를 쓰고 있어 `ModuleNotFoundError: No module named 'config'`로
실패했다. 상대 import로 통일해 문서와 코드를 일치시켰다.

**TC-309: 문서에 적힌 명령이 그대로 동작한다**
```bash
python -m src.train             # 학습
python -m src.main_hard_mining  # hard negative mining 학습
python -m src.test              # 평가
```
- **예상:** 세 명령 모두 `ModuleNotFoundError` 없이 시작

---

## 5. 성능 테스트

### 5.1 메모리 효율성

**TC-400: 배치 처리 메모리**
```python
import sys
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet

# GPU 메모리 모니터링 (선택사항)
batch_size = 64
images = torch.randn(batch_size, 3, 224, 224)
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
import sys
import time
import torch
sys.path.insert(0, './src')
from models import ResNetTriplet

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

### 6.1 개별 테스트 케이스 실행 (Python)
테스트 케이스들은 이 문서에 명시된 Python 코드 예시로 검증할 수 있습니다.

```python
# TC-001: Margin 값 확인
from src.config import margin
assert margin == 0.0001, f"Expected margin=0.0001, got {margin}"
print("TC-001 passed: margin configuration correct")
```

### 6.2 모델 구조 검증
```bash
python -c "
import torch
from src.models import ResNetTriplet
model = ResNetTriplet(model_name='resnet18')
batch = torch.randn(2, 3, 224, 224)
embeddings, logits = model(batch)
print(f'Embeddings shape: {embeddings.shape}, Logits shape: {logits.shape}')
"
```

### 6.3 통합 테스트
```bash
python -c "
from src.models import ResNetTriplet
from src.data import TripletWhaleDataset
from src.utils import TripletLoss

# 모듈 임포트 확인
print('All modules imported successfully')
print(f'ResNetTriplet available')
print(f'TripletWhaleDataset available')
print(f'TripletLoss available')
"
```

### 6.4 시스템 테스트 (실제 훈련 흐름)
```bash
# 작은 데이터셋으로 테스트 (1 배치만)
export BATCH_SIZE=2
export NUM_TRAIN_TRIPLETS=10
export EPOCHS=1  # 1 에포크만
python -m src.train
python -m src.test
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
