# 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 불러오기 및 전처리
file_path = '/Users/joon/Documents/GitHub/Final_Project/data/1.smoke_detection_iot.csv'
df = pd.read_csv(file_path)

# 불필요한 열 제거 및 UTC 시간 변환
df = df.drop(columns=['Unnamed: 0'])
df['UTC'] = pd.to_datetime(df['UTC'], unit='s')
df['Hour'] = df['UTC'].dt.hour
df['Day'] = df['UTC'].dt.day
df['Month'] = df['UTC'].dt.month
df['Year'] = df['UTC'].dt.year

# 2. 탐색적 데이터 분석 (EDA)
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Feature Correlation Matrix", fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()

# 3. 시간에 따른 모든 변수의 데이터 시각화
variables = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5']

plt.figure(figsize=(12, 8))
for var in variables:
    plt.plot(df['UTC'], df[var], label=var)

# X축 레이블 및 눈금 설정
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Environmental Variables Over Time')
plt.legend(loc='upper right')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))

plt.xticks(rotation=45)
plt.show()

# 4. 모델링 및 평가
# 'UTC' 열을 모델에서 제외하고, 독립 변수와 종속 변수 설정
X = df.drop(columns=['Fire Alarm', 'UTC'])
y = df['Fire Alarm']

# 데이터셋을 학습용과 테스트용으로 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 테스트 세트에 대한 예측 및 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# 5. 특성 중요도 시각화
importances = model.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90, fontsize=12)
plt.ylabel('Importance', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xlim([-1, X.shape[1]])
plt.show()
