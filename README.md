
# 연기 감지 및 화재 경보 예측

## 프로젝트 목적

본 프로젝트의 주요 목적은 **딥러닝 기술을 활용한 고성능 연기 감지 시스템을 개발**하여 화재를 조기에 발견하고 대응 능력을 향상시키는 것입니다. 이를 통해 인명 및 재산 피해를 최소화하고, 공공 안전을 증진시키고자 합니다. 기존 연기 감지기의 한계를 극복하고, 다양한 환경에서도 신뢰할 수 있는 화재 감지 시스템을 구현하는 것을 목표로 합니다.

<img width="768" alt="스크린샷 2024-08-21 오후 10 51 38" src="https://github.com/user-attachments/assets/d8d05369-cc6e-45c6-a1e3-f681727fc138">


## 현 상황 분석

1. **기존 연기 감지기의 한계**:
   - 오작동으로 인한 불필요한 경보 발생
   - 늦은 반응 시간으로 인한 화재 대응 지연
   - 특정 환경(예: 고습도, 먼지가 많은 환경)에서 낮은 신뢰성

2. **대형 화재 사고의 지속적 발생**:
   - 대형 화재로 인한 인명 및 재산 피해가 계속해서 발생하고 있으며, 이는 사회적 우려를 증가시키고 있습니다.
   - 이러한 상황에서 더욱 정교하고 신뢰할 수 있는 화재 감지 시스템의 필요성이 대두되고 있습니다.

## 기대 효과

1. **정확도 및 반응 시간 향상**:
   - 딥러닝을 활용하여 높은 정확도와 빠른 반응 시간을 갖춘 연기 감지 시스템을 구현합니다.
   - 신속한 화재 감지를 통해 초기 대응이 가능해집니다.

2. **다양한 환경에서의 신뢰성 강화**:
   - 다양한 환경과 상황에서 화재를 조기에 발견할 수 있도록 시스템의 감지 능력을 향상시킵니다.
   - 기존 감지기의 한계를 극복하여, 환경 변화에도 안정적인 성능을 발휘합니다.

3. **오작동 감소**:
   - 불필요한 경보 발생을 줄여 소방력의 낭비를 방지하고, 실제 화재 상황에서의 대응 효율성을 높입니다.

4. **화재 피해 예방**:
   - 조기 화재 감지를 통해 인명 및 재산 피해를 예방하고, 안전한 생활 환경을 조성합니다.

**결론**: 본 프로젝트를 통해 개발될 딥러닝 기반 연기 감지 시스템은 기존 화재 감지 기술의 한계를 극복하고, 더욱 안전한 사회 구현에 기여할 것으로 기대됩니다.

## 데이터 설명

- **데이터 출처**: IoT 기반 연기 감지기 데이터셋
- **데이터 포맷**: CSV 파일
- **총 샘플 수**: 60,000개 (오버샘플링을 통해 데이터 확장)
- **주요 변수**:
  - `Temperature[C]`: 온도 (섭씨)
  - `Humidity[%]`: 습도 (백분율)
  - `TVOC[ppb]`: 총 휘발성 유기 화합물 농도 (ppb)
  - `eCO2[ppm]`: 이산화탄소 농도 (ppm)
  - `Raw H2`: 수소 농도
  - `Raw Ethanol`: 에탄올 농도
  - `Pressure[hPa]`: 대기압 (hPa)
  - `PM1.0`, `PM2.5`: 미세먼지 농도 (μg/m³)
  - `NC0.5`, `NC1.0`, `NC2.5`: 입자 개수 농도
  - `Hour`, `Day`, `Month`, `Year`: 시간 관련 파생 변수
  - `Fire Alarm`: 화재 경보 여부 (타겟 변수, 0: 경보 없음, 1: 경보 있음)

## 분석 및 모델링 과정

1. **데이터 전처리**:
   - **불필요한 열 제거**: 분석에 필요하지 않은 `Unnamed: 0` 열을 제거.
   - **시간 데이터 처리**: UTC 시간을 사람이 읽기 쉬운 `Hour`, `Day`, `Month`, `Year`로 변환하여 시간 관련 패턴 분석 가능.
   - **오버샘플링**: 원본 데이터의 균형 문제를 해결하기 위해 데이터를 오버샘플링하여 60,000개의 샘플을 확보.

2. **탐색적 데이터 분석 (EDA)**:
   - **상관관계 분석**: 변수들 간의 상관관계를 시각화하여 중요한 변수들을 파악.
   - **데이터 분포 확인**: 각 변수의 분포를 분석하여 데이터의 특성을 파악하고, 잠재적인 이상치를 식별.

3. **모델링 및 평가**:
   - **모델 선택**: 화재 경보를 예측하기 위해 **랜덤 포레스트(Random Forest)** 모델을 사용.
   - **훈련 및 테스트 데이터 분할**: 데이터를 80%는 훈련용, 20%는 테스트용으로 분할.
   - **모델 학습**: 훈련 데이터를 사용해 랜덤 포레스트 모델을 학습.
   - **모델 평가**: 테스트 데이터를 사용해 모델의 성능을 평가하고, 정확도(Accuracy)와 분류 보고서(Classification Report)를 통해 결과를 분석.

4. **특성 중요도 분석**:
   - **특성 중요도 시각화**: 모델이 예측 시 가장 중요하게 사용하는 변수를 시각적으로 분석. 이를 통해 모델이 어떤 변수에 의존하는지 파악 가능.
   - **결과 해석**: 중요도가 높은 변수들이 실제로 화재 발생과 어떤 관계가 있는지 해석하여 모델의 신뢰성을 높임.

## 주요 결과 및 결론

이 프로젝트를 통해 다음과 같은 결론을 도출했습니다:
- **모델 정확도**: 랜덤 포레스트 모델은 높은 정확도로 화재 경보를 예측할 수 있음을 확인했습니다.
- **중요한 변수**: `TVOC`, `eCO2`, `Raw H2`, `Raw Ethanol` 등이 화재 예측에서 중요한 역할을 하는 변수로 확인되었습니다.
- **오버샘플링의 효과**: 데이터를 오버샘플링함으로써 모델의 예측 성능이 향상되었으며, 특히 소수 클래스에 대한 예측력이 개선되었습니다.

## 사용 방법

1. **환경 설정**:
   - 이 리포지토리를 클론하고, 필요한 Python 라이브러리(`pandas`, `matplotlib`, `seaborn`, `scikit-learn`)를 설치합니다.
   - `requirements.txt` 파일이 제공된 경우, 다음 명령어로 필요한 패키지를 설치할 수 있습니다:
     ```
     <img width="638" alt="스크린샷 2024-08-21 오후 10 47 23" src="https://github.com/user-attachments/assets/cd50110e-b9ea-4a9a-a7f8-51f10120a2ce">

     ```

2. **데이터 분석 및 모델링**:
   - Python 스크립트를 실행하여 데이터 분석 및 모델 학습을 진행합니다.
   - 분석 결과와 모델 성능을 시각화된 그래프로 확인할 수 있습니다.

3. **결과 해석 및 응용**:
   - 생성된 모델을 실제 화재 감지 시스템에 적용하여 화재 발생 가능성을 예측하고, 대응 전략을 세울 수 있습니다.
   - 결과를 바탕으로 추가적인 데이터 수집 및 모델 개선을 위한 피드백을 얻을 수 있습니다.

