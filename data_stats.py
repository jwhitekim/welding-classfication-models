import pandas as pd

# 1. CSV 파일 불러오기
# 'your_data.csv' 부분을 실제 파일명으로 바꿔주세요.
try:
    df = pd.read_csv("./resource/Data_RSW.csv")
    print("파일이 성공적으로 로드되었습니다.\n")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 파일 경로와 이름을 다시 확인해주세요.")
    exit()

# 2. 데이터 기본 정보 확인 (데이터 타입, 누락값 등)
print("--- 데이터 기본 정보 ---")
print(df.info())
print("\n")

# 3. 데이터의 상위 5개 행 미리보기
print("--- 데이터 상위 5개 행 ---")
print(df.head())
print("\n")

# 4. 숫자 및 범주형 데이터의 통계 요약 (평균, 표준편차, 최빈값 등)
# 숫자형 데이터: count, mean, std, min, max 등
# 범주형 데이터: count, unique, top, freq 등
print("--- 데이터 통계 요약 ---")
statistical_summary = df.describe(include='all')
print(statistical_summary)
print("\n")

# 5. 통계 요약 결과를 새로운 CSV 파일로 저장
# 'statistical_summary.csv'라는 이름으로 결과가 저장됩니다.
statistical_summary.to_csv('data/statistical_summary.csv')
print("통계 요약 결과가 'statistical_summary.csv' 파일로 저장되었습니다.")