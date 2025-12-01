# 광진구 빅데이터 공모전 - 외식업 폐업률 요인분석

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)

광진구 외식업 예비 창업자를 위한 폐업률 요인 분석 및 예측 모델

## 🎯 Overview

이 프로젝트는 코로나 이후에도 계속되는 외식업 폐업 현상에 대해 광진구를 중심으로 유의미한 요인을 분석하고, 예비 창업자에게 인사이트를 제공합니다.

- **Target Area**: 서울시 광진구
- **Analysis Period**: 2019년 ~ 2023년 (5년간)
- **Best Model**: Random Forest
- **Performance**: RMSE 5.9395, MAE 4.0214

## 📄 Research Background

### Problem Statement

**"음식점 10개 창업 때 8개 이상 폐업...폐업률, 타업종보다 높아"**

- 스타 식당도 줄폐업...빛 바랜 '미쉐린 별'
- "코로나 때보다 힘들어요"...음식점 줄 폐업

코로나 이후에도 계속되는 외식업의 폐업 현상을 분석하여 광진구 음식점 폐업률에 대한 유의미한 요인을 도출하고자 합니다.

### Research Objective

광진구 외식업 예비 창업자를 위한 데이터 기반 의사결정 지원

## 🔬 Method

### Data Collection

**서울시 상권분석 서비스 데이터 활용**

#### 업체 자료
- 서비스 업종명
- 주중/주말 매출 금액
- 시간대별 매출금액
- 연령대별 매출금액

#### 상권 자료
- 유사업종 점포 수
- 개업률/폐업률
- 프랜차이즈 점포수

### Analysis Pipeline
```
Data Collection (2019-2023)
    ↓
Hierarchical Linear Models (HLM)
    ├─ Level 1: 시간
    ├─ Level 2-1: 음식점
    └─ Level 2-2: 행정동
    ↓
Feature Engineering (Lasso Regression)
    ↓
Model Training & Comparison
    ├─ Linear Regression
    ├─ Random Forest ✓
    └─ XGBoost
    ↓
Feature Importance Analysis
```

## 📊 Results

### Model Performance Comparison

| Model | RMSE | MAE | Status |
|-------|------|-----|--------|
| Linear Regression | 5.9658 | 4.1190 | ✓ |
| **Random Forest** | **5.9395** | **4.0214** | ⭐ **Selected** |
| XGBoost | 5.9396 | 4.1144 | ✓ |

**Random Forest가 가장 좋은 성능을 보임**

### Key Findings - Feature Importance

#### Top 3 중요 변수 (Random Forest 기준)

1. **개업률 (0.30)**
   - 개업이 활발하면 경쟁이 더욱 치열해질 수 있음
   - 기존 가게들의 시장 점유율 경쟁 심화

2. **21시~24시 영업매출 (0.05)**
   - 24시까지 영업하는 점포는 야간 고객층 타겟 가능
   - 매출 증대로 이어질 수 있어 폐업률 감소

3. **유사업종 점포수 (0.04)**
   - 유사업종 밀집 지역에서 경쟁 치열
   - 가격/서비스 경쟁으로 수익성 감소
   - 폐업 가능성 증가

### HLM Analysis Results

#### 음식점 수준 분석
```
Level 1: 폐업률ij = γ00 + γ01*Timeij + γ0i + γ1i*Timeij + eij

Random Effect:
- τ00 = 1.195 (2019년 1분기 업종별 폐업률 차이 존재)
- τ01 = 0.007 (음식점 간 시간에 따른 차이 거의 없음)

Fixed Effect:
- γ00 = 4.500 (절편)
- γ01 = -0.046 (시간 효과 - 유의하지 않음)

ICC = 0.005
```

#### 행정동 수준 분석
```
Level 1: 폐업률ij = γ00 + γ01*Timeij + γ0i + γ1i*Timeij + eij

Random Effect:
- τ00 = 0.048 (행정동 간 차이 매우 작음)
- τ01 = 0.003 (시간 효과 차이 거의 없음)

Fixed Effect:
- γ00 = 4.441 (절편)
- γ01 = -0.041 (시간 효과 - 유의하지 않음)

ICC = 0.019
```

**→ ICC가 매우 낮아(0.005, 0.019) 종단적 특성을 반영하지 않는 분석 방법도 가능**

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gwangjin-restaurant-closure.git
cd gwangjin-restaurant-closure

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. 서울시 상권분석 서비스에서 데이터 다운로드
2. `data/raw/` 디렉토리에 저장
```
data/
├── raw/
│   ├── 점포수_2019-2023.csv
│   ├── 개폐업수_2019-2023.csv
│   └── 매출데이터_2019-2023.csv
```

### Quick Start
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# 데이터 로드
df = pd.load_csv('data/processed/gwangjin_restaurant_data.csv')

# 특징 선택 (Lasso)
X = df.drop(['폐업률'], axis=1)
y = df['폐업률']

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso로 특징 선택
lasso = Lasso(alpha=0.359)
lasso.fit(X_scaled, y)

# 중요 특징 추출
important_features = X.columns[lasso.coef_ != 0]

# Random Forest 학습
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=8,
    min_samples_split=8,
    random_state=2024
)

rf.fit(X_scaled[:, important_features], y)

# 예측
predictions = rf.predict(X_test_scaled)
```

## 📁 Repository Structure
```
gwangjin-restaurant-closure/
│
├── data/
│   ├── raw/                        # 원본 데이터
│   │   ├── 점포수_2019-2023.csv
│   │   ├── 개폐업수_2019-2023.csv
│   │   └── 매출데이터_2019-2023.csv
│   └── processed/                  # 전처리된 데이터
│
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py       # 데이터 정제
│   │   └── feature_engineering.py # 특징 공학
│   │
│   ├── models/
│   │   ├── hlm_analysis.py        # 위계적 선형 모델
│   │   ├── linear_regression.py
│   │   ├── random_forest.py       # Random Forest (최종 모델)
│   │   └── xgboost_model.py
│   │
│   ├── feature_selection/
│   │   └── lasso_selection.py     # Lasso 특징 선택
│   │
│   └── visualization/
│       └── importance_plot.py      # 변수 중요도 시각화
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_HLM_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   └── 04_model_comparison.ipynb
│
├── results/
│   ├── models/                     # 학습된 모델
│   └── figures/                    # 시각화 결과
│
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

### Statistical Analysis
- **Hierarchical Linear Models (HLM)** - 종단 데이터 분석
- **ICC (Intraclass Correlation)** - 집단 간 상관 분석

### Machine Learning
- **Lasso Regression** - 특징 선택
- **Random Forest** - 최종 예측 모델
- **XGBoost** - 모델 비교
- **Grid Search CV** - 하이퍼파라미터 최적화

### Data Processing
- **Pandas** - 데이터 처리
- **NumPy** - 수치 연산
- **StandardScaler** - 데이터 표준화

### Visualization
- **Matplotlib** - 기본 시각화
- **Seaborn** - 통계 시각화

## 💡 Key Insights

### 1. 개업률 (Opening Rate)

**가장 중요한 변수 (중요도: 0.30)**

- 개업이 활발하면 경쟁이 더욱 치열해짐
- 기존 가게들의 시장 점유율 경쟁 → 가격/마케팅 경쟁 심화
- 폐업 가능성 증가

**창업자 가이드:**
- 개업률이 높은 지역은 피할 것
- 차별화된 경쟁력 확보 필수

### 2. 21시~24시 영업매출

**야간 영업의 중요성 (중요도: 0.05)**

- 야간 노동 인구 타겟 가능
- 늦은 시간 서비스 수요층 확보
- 매출 증대 → 폐업률 감소

**창업자 가이드:**
- 야간 영업 가능한 업종 고려
- 주변 유동 인구 패턴 분석

### 3. 유사업종 점포수

**경쟁 강도 지표 (중요도: 0.04)**

- 유사업종 밀집 → 치열한 경쟁
- 가격/서비스 경쟁 → 수익성 감소
- 경쟁력 및 생존 가능성 감소

**창업자 가이드:**
- 유사업종이 적은 지역 선택
- 또는 확실한 차별화 전략 수립

## 📈 Model Details

### Lasso Feature Selection
```python
최적의 alpha: 0.3593813663804626

# 선택된 특징
선택된 특성 Index(['개업_률'], dtype='object')
```

### Random Forest Hyperparameters

**Grid Search 최적 파라미터:**
```python
{
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_leaf': 8,
    'min_samples_split': 8
}
```

### XGBoost Feature Importance

**Top 5 중요 변수:**
1. 시간대_06~11_매출_금액
2. 주요일_매출_건수
3. 개업_율
4. 시간대_건수~17_매출_건수
5. 남성_매출_금액

## 🚧 Limitations

### 1. 데이터 부정확성

데이터의 부정확성으로 인해 명확한 분석과 해석에 어려움이 존재합니다.

### 2. 메타데이터 부재

각 컬럼의 형성 과정을 알 수 없어 데이터를 분석하고 해석하는 것에 한계점이 존재합니다.

## 🎓 Use Cases

### 예비 창업자를 위한 활용 방안

1. **입지 선정**
   - 개업률이 낮은 지역 우선 고려
   - 유사업종 밀집도 확인

2. **영업 전략**
   - 야간 영업 가능성 검토
   - 시간대별 매출 패턴 분석

3. **경쟁 분석**
   - 주변 유사업종 점포수 조사
   - 차별화 전략 수립

4. **리스크 평가**
   - 해당 지역/업종 폐업률 확인
   - 데이터 기반 의사결정

## 📚 References

### 데이터 출처
- [서울시 상권분석 서비스](https://golmok.seoul.go.kr/) - 지역분석 데이터 (2019-2023)
  - 점포수
  - 개폐업수
  - 인구수
  - 신생기업 생존율

### 언론 기사
- 서울경제(2024), "스타 식당도 줄폐업…빛 바랜 '미쉐린 별'"
- 조선경제(2024), "작년 문닫은 식당, 코로나 때보다 많았다"
- KBS뉴스(2024), "코로나 때보다 힘들어요"…음식점 줄 폐업

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

⭐ **Key Findings**: Random Forest 모델을 통해 개업률, 야간 영업매출, 유사업종 점포수가 광진구 외식업 폐업률에 가장 큰 영향을 미치는 요인임을 확인하였습니다. 이는 예비 창업자의 입지 선정 및 경영 전략 수립에 유용한 인사이트를 제공합니다.
