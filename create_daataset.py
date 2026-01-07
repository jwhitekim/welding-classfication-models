import os
import shutil
import pandas as pd

# 경로 설정
csv_path = "./resource/Data_RSW.csv"   # CSV 파일 경로
img_folders = ["./resource/Images_IR", "./resource/Images_RGB"]            # 원본 이미지 폴더
output_folder = "datasets"       # 분류된 폴더 생성

# CSV 읽기
df = pd.read_csv(csv_path)

# Good / Bad 폴더 생성
for category in df['Category'].unique():
    os.makedirs(os.path.join(output_folder, "IR", category), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "RGB", category), exist_ok=True)

# 이미지 분류
for idx, row in df.iterrows():
    sample_id = str(row['Sample ID'])       # 이미지 파일명과 매칭
    category = row['Category']
    
    # 이미지 파일 확장자 예시: .jpg, 필요하면 png로 변경
    img_name_IR = f"IR_{sample_id}.jpg"
    src_path = os.path.join(img_folders[0], img_name_IR)
    dst_path = os.path.join(output_folder, "IR", category, img_name_IR)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"이미지 없음: {src_path}")

    img_name_RGB = f"RGB_{sample_id}F.jpg"
    src_path = os.path.join(img_folders[1], img_name_RGB)
    dst_path = os.path.join(output_folder, "RGB", category, img_name_RGB)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"이미지 없음: {src_path}")

print("이미지 분류 완료!")
