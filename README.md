# 연기 감지 및 화재 경보 예측
<img width="768" alt="스크린샷 2024-08-21 오후 10 51 38" src="https://github.com/user-attachments/assets/4c18fa78-79a6-4a1b-bec0-34cfd2fdfee5">


# **프로젝트 이름: IOT 기반 화재 예측 시스템**

## **프로젝트 개요**
본 프로젝트의 주요 목적은 IOT(사물인터넷) 장치를 통해 수집된 데이터를 사용하여 고성능 연기 감지 시스템을 개발하는 것입니다. 이를 통해 화재를 조기에 발견하고 대응함으로써 인명과 재산 피해를 최소화하고, 공공 안전을 증진시키고자 합니다.

## **현 상황 분석**
- 기존 연기 감지기의 한계: 오작동이 잦고, 반응 시간이 늦으며, 특정 환경에서는 신뢰성이 낮음
- 지속적인 대형 화재 사고 발생으로 인한 사회적 우려 증가

## **기대 효과**
- 높은 정확도와 빠른 반응 시간을 갖춘 연기 감지 시스템 구현
- 다양한 환경 및 상황에서의 화재 조기 발견 능력 향상
- 오작동 감소로 인한 소방력 낭비 방지
- 화재 관련 인명 및 재산 피해 예방

본 프로젝트를 통해 개발될 딥러닝 기반 연기 감지 시스템은 기존 화재 감지 기술의 한계를 극복하고, 더욱 안전한 사회 구현에 기여할 것으로 기대됩니다.

## **데이터 설명**
- 데이터는 약 60,000개의 측정값으로 구성되었으며, 모든 센서의 샘플링 속도는 1Hz입니다.
- 주요 변수:
  - **Temperature[C]**: 온도 (섭씨)
  - **Humidity[%]**: 습도 (백분율)
  - **TVOC[ppb]**: 총 휘발성 유기 화합물 농도 (ppb)
  - **eCO2[ppm]**: 이산화탄소 농도 (ppm)
  - **Raw H2**: 수소 농도
  - **Raw Ethanol**: 에탄올 농도
  - **Pressure[hPa]**: 대기압 (hPa)
  - **PM1.0, PM2.5**: 미세먼지 농도 (μg/m³)
  - **Fire Alarm**: 화재 경보 (0: 경보 없음, 1: 경보 발생)

## **모델링 및 분석**
- **데이터 전처리**:
  - 불필요한 열을 제거하고, `UTC` 시간을 연도, 월, 일, 시간 등으로 분리하여 분석에 용이하게 변환했습니다.
- **탐색적 데이터 분석 (EDA)**:
  - 상관관계 행렬을 통해 각 변수들 간의 관계를 분석했습니다. 예를 들어, 특정 가스 농도와 온도 사이의 강한 상관관계를 확인할 수 있었습니다.
  - 시간에 따른 주요 환경 변수들의 변화를 시각화하여, 시간이 지남에 따라 변수들이 어떻게 변동하는지 파악했습니다.
    
- **모델링**:
  - **랜덤 포레스트 모델**을 사용하여 `Fire Alarm`을 예측했습니다. 모델은 **100%의 정확도**로 테스트 데이터를 예측했습니다.
  - **정확도(Accuracy)**: 테스트 데이터에서 예측된 값 중 실제 값과 일치하는 비율로, 1.00(100%)의 정확도를 기록했습니다. 이는 모델이 테스트 데이터에 대해 모든 예측을 정확히 수행했다는 것을 의미합니다.
  - **정밀도(Precision)**: 모델이 경보를 예측한 경우, 그 예측이 실제로 맞는 비율을 나타냅니다. 이 모델의 정밀도는 1.00으로, 모델이 예측한 경보는 모두 실제로 화재였음을 의미합니다.
  - **재현율(Recall)**: 실제 화재 경보 중에서 모델이 정확하게 예측한 비율을 나타내며, 이 모델의 재현율은 1.00입니다. 이는 실제 화재 경보를 모두 정확히 예측했다는 것을 의미합니다.
  - **F1 점수(F1-Score)**: 정밀도와 재현율의 조화 평균으로, 이 모델의 F1 점수도 1.00으로 매우 높은 성능을 나타냅니다.
  - 이러한 결과는 모델이 매우 높은 성능으로 화재 경보를 예측할 수 있음을 보여줍니다.
    
- **특성 중요도 분석**:
  - 랜덤 포레스트 모델을 통해 어떤 변수가 예측에 가장 중요한 역할을 하는지 시각적으로 확인했습니다. 예를 들어, `TVOC`와 `eCO2`가 중요한 변수로 나타났습니다. 이는 화재 예측에서 이 두 변수가 중요한 역할을 한다는 것을 시사합니다.

## **결론**
이 프로젝트는 IOT 기반 화재 예측 시스템의 가능성을 성공적으로 시연했습니다. 특히, 랜덤 포레스트 모델을 통해 **100%의 예측 정확도**를 달성했으며, 각 변수들의 중요성을 분석하여 화재 예측에 중요한 인사이트를 제공했습니다.


## 사용 방법

1. **환경 설정**:
   - 이 리포지토리를 클론하고, 필요한 Python 라이브러리(`pandas`, `matplotlib`, `seaborn`, `scikit-learn`)를 설치합니다.
   - `requirements.txt` 파일이 제공된 경우, 다음 명령어로 필요한 패키지를 설치할 수 있습니다:
     ```
   <img width="638" alt="스크린샷 2024-08-21 오후 10 47 23" src="https://github.com/user-attachments/assets/60683b94-03e3-4e13-8437-c3e726d94146">


     ```

2. **데이터 분석 및 모델링**:
   - Python 스크립트를 실행하여 데이터 분석 및 모델 학습을 진행합니다.
   - 분석 결과와 모델 성능을 시각화된 그래프로 확인할 수 있습니다.

3. **결과 해석 및 응용**:
   - 생성된 모델을 실제 화재 감지 시스템에 적용하여 화재 발생 가능성을 예측하고, 대응 전략을 세울 수 있습니다.
   - 결과를 바탕으로 추가적인 데이터 수집 및 모델 개선을 위한 피드백을 얻을 수 있습니다.

