# 📈 Multimodal AutoEncoder 기반 시장 이상치 탐지 (Market Anomaly Detection)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C)
![KoBERT](https://img.shields.io/badge/Model-KoBERT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

<b>본 프로젝트는 KOSPI 가격 데이터(정량)와 뉴스 텍스트 데이터(정성)를 결합한  
멀티모달(Multimodal) AutoEncoder 기반 <b>시장 이상치 탐지 시스템<b>입니다.

AutoEncoder의 재구성 오차(Reconstruction Error)를 기반으로  
비정상적인 시장 움직임을 탐지하고, 해당 시점의 주요 뉴스(첫 문장)를 추출하여  
시장 변동 원인을 자동으로 리포팅합니다.

---

#  1. 프로젝트 개요

##  목표
- 주가와 뉴스 데이터를 통합하여 **시장 이상치(Anomaly)를 자동 탐지**
- 뉴스 텍스트 분석을 활용하여 **시장 변동 원인을 설명**
- 이상치 Top 20 날짜에 대한 **가격 그래프 + 요약 리포트 자동 생성**
- 멀티모달 AutoEncoder 기반 Reconstruction 기반 이상 탐지 모델 구축

---

# 2. 입력 데이터(Inputs)

##  2-1. 정량 데이터 (Quantitative)
| Feature | 설명 |
|--------|------|
| close | KOSPI 종가 |
| return | 수익률 |
| log_return | 로그 수익률 |
| volatility_10d | 10일 이동 변동성 |
| volume | 거래량 |
[![img_1.png](img_1.png)](https://github.com/siyeoncoding/PytorchProject/issues/new)
➡ 총 5개 수치형 Feature

##  2-2. 크롤링 데이터
- 네이버 뉴스 API 수집 
- 하루 전체 뉴스 문장을 하나의 텍스트로 병합
- → KoBERT CLS Embedding (768차원)
<img width="1115" height="399" alt="Image" src="https://github.com/user-attachments/assets/67648d67-2e37-44f1-9283-bcc9abd349f3" />

### 최종 입력 구조
```
5 numerical features  
+ 768 KoBERT embedding  
= 총 773차원 Feature Vector
```

---

# 3. 모델 구조 (AutoEncoder)

```
Encoder: 773 → 512 → 256 → 128 → 64  
Decoder: 64 → 128 → 256 → 512 → 773  
Loss: MSE (Reconstruction Error)
```

정상 데이터를 잘 복원하고,  
**복원이 어려운 날(오차 ↑)** 을 **이상치(Anomaly)** 로 간주.

---

# 🔍 4. 데이터 분할 (6:2:2 비율)

성능 향상 및 안정적 검증을 위해 6:2:2로 데이터 분할.

| 구분 | 비율 | 설명 |
|------|------|------|
| Train | 60% | 모델 학습 |
| Validation | 20% | 하이퍼파라미터 튜닝 |
| Test | 20% | 최종 이상치 탐지 |

> 랜덤 셔플 기반 분할 (`train_test_split` 사용)

---

# 5. 실험 결과 (Hyperparameter Tuning)

Batch Size에 따른 학습 곡선 비교.

## 🔹 Batch Size = 16  
![ae_training_curve_16batch.png](models%2Fautoencoder%2Fae_training_curve_16batch.png)

## 🔹 Batch Size = 32 (Baseline)
![ae_training_curve_32batch.png](models%2Fautoencoder%2Fae_training_curve_32batch.png)

## 🔹 Batch Size = 64  
![ae_training_curve_64batch.png](models%2Fautoencoder%2Fae_training_curve_64batch.png)

## 🔹 6:2:2 분할 실험
![ae_training_curve_622.png](models%2Fautoencoder%2Fae_training_curve_622.png)

### Observations
- Batch Size 32 → 가장 안정적, 최적 성능  
- Batch Size 16 → 수렴 속도 느림  
- Batch Size 64 → Validation Loss 진동 증가  
- 6:2:2 → Train/Val 균형 좋아 과적합 감소

---

# 6. 이상치 탐지 결과 (Top 20)

##  6-1. KOSPI 가격 + 이상치 위치  
![A1_price_with_anomalies_top20.png](src%2Fanalysis%2Fanalysis%2Ffigures%2FA1_price_with_anomalies_top20.png)

🔴 붉은 점 = Reconstruction Error 상위 20개 날짜

---

## 6-2. 이상치 Top 20 상세 뉴스 요약

### Page 1
![A2_top20_anomaly_summaries_part1.png](src%2Fanalysis%2Fanalysis%2Ffigures%2FA2_top20_anomaly_summaries_part1.png)

### Page 2
![A2_top20_anomaly_summaries_part2.png](src%2Fanalysis%2Fanalysis%2Ffigures%2FA2_top20_anomaly_summaries_part2.png)

### 내용 요약
- 이상치 날짜는 **대규모 수주/전쟁/환율 변동/정책 충격** 등 주요 이벤트 집중
- 뉴스 첫 문장으로 해당 시장 반응의 원인 유추하기
---

#  7. 결론 (Conclusion)

- **정량 + 정성 결합 멀티모달 AutoEncoder**는 시장 이상 탐지에 효과적
- KoBERT 임베딩은 이상치 구분력 향상에 기여
- KoBART 요약 대신 “뉴스 첫 문장 추출 방식” 사용하여 속도 최적화
- 6:2:2 분할에서 모델 안정적 수렴 & 과적합 감소
- Batch Size 32 + LR=1e-3 조합이 최적

---

# 📁 8. 프로젝트 디렉토리 구조(요약)

```
PytorchProject/
│
├── data/
│   ├── raw/                # 원천 데이터
│   ├── interim/            # 중간 처리 결과
│   ├── processed/          # 최종 병합 + anomaly 결과
│
├── src/
│   ├── data/               # 로딩/전처리
│   ├── features/           # Feature 생성 + KoBERT 임베딩
│   ├── models/             # AutoEncoder 모델 정의
│   ├── analysis/           # 이상치 분석/리포트 생성
│
├── models/
│   ├── autoencoder/        # Best 모델 저장
│
└── README.md
```

---

# 🔗 9. 향후 발전 방향

- Transformer 기반 시계열 AutoEncoder 적용  
- 뉴스 요약에 KoBART-Large or LLM Summarizer 추가  
- 섹터별 이상치 분석 모델 확장  
- 실시간 스트리밍 기반 Online Anomaly Detection 구축  

---

