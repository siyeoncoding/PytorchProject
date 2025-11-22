# KOSPI Insight Analyzer  
**PyTorch + KoBERT 기반 한국 주식시장 이상 변동 탐지 시스템**

---

## 프로젝트 개요

**KOSPI Insight Analyzer**는 한국 주식시장의  
**뉴스 텍스트 + 주가 시계열 데이터를 동시에 분석**하여  
급등·급락·비정상 변동(Anomaly)을 **딥러닝 AutoEncoder**로 조기에 탐지하는 프로젝트입니다.

뉴스에 나타난 ‘시장 심리’와  
시계열 데이터가 보여주는 ‘가격 구조 변화’를 함께 읽어  
더 높은 신뢰도를 가진 변동성 탐지를 목표로 합니다.

---

## 핵심 목표

### • 시장 심리 기반 변동성 탐지  
뉴스 텍스트를 KoBERT로 임베딩하여 시장의 감정·정책·리스크 신호를 반영

### • 주가 시계열 기반 위험 구간 탐지  
KOSPI 종가·수익률·변동성 등의 변화를 기반으로 패턴 분석

### • AutoEncoder로 정상 상태 학습 → 이상 탐지  
비지도 학습 기반으로 정상 시장 패턴을 학습해  
재구성 오차(Reconstruction Error)를 통해 급격한 시장 변동을 포착

---

## 구현 기능

### 1️⃣ 네이버 뉴스 API 기반 1년치 뉴스 수집  
- 약 **10만 건 뉴스 데이터 자동 수집**
- pubDate 직접 파싱 → 날짜 조건 필터링
- ‘한국 증시’ 관련 기사만 저장

### 2️⃣ KOSPI 데이터 전처리  
- pykrx 기반 시계열 데이터 로드
- Return, Volatility 생성
- 날짜 정규화 및 결측 처리

### 3️⃣ 뉴스 텍스트 처리  
- HTML 태그 제거
- 날짜별 다수 기사 → 단일 문장으로 요약 병합
- 데이터 프레임화

### 4️⃣ 뉴스 + KOSPI 멀티모달 병합  
- KOSPI 거래일 기준 Left Join  
- 뉴스가 없는 날짜는 빈 문자열("")  
→ KoBERT("", 중립 벡터)으로 임베딩 처리

### 5️⃣ 뉴스 임베딩 (KoBERT)  
- 각 뉴스 문장을 KoBERT로 768차원 벡터화  
- 임베딩 캐싱 기능 제공 (kobert_emb.pt)

### 6️⃣ PyTorch AutoEncoder 모델 학습  
- 입력 차원: 771 (수치 3 + 텍스트 임베딩 768)
- Reconstruction Error 기반 anomaly score 산출

---

 ##  기술 스택

- Python 3.9+
- PyTorch
- HuggingFace Transformers (KoBERT)
- python-dotenv
- pykrx
- Pandas / NumPy


---

