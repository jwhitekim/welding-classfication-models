from train import *
from plot import *

import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

NUM_CLASSES = 3

# 클래스 이름 가져오기
TRAIN_DATASET = datasets.ImageFolder(root=f"dataset/train")
CLASS_NAME = TRAIN_DATASET.classes

def evaluate_rgb(model_rgb, loader_rgb):
    """
    RGB 모델의 예측 결과를 반환합니다. 앙상블 대신 단일 모델 테스트를 수행합니다.
    """
    model_rgb.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs_rgb, labels_rgb in loader_rgb:
            # inputs_ir과 labels_ir은 더 이상 필요 없으므로 삭제
            inputs_rgb, labels_rgb = inputs_rgb.to(device), labels_rgb.to(device)

            # IR 모델의 예측 로직 삭제
            outputs_rgb = torch.softmax(model_rgb(inputs_rgb), dim=1)
            
            # 가중 평균 대신 RGB 모델의 예측 결과만 사용
            _, preds = torch.max(outputs_rgb, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_rgb.cpu().numpy())
            
    return all_preds, all_labels

def custom_evaluate(model_rgb, model_name, report_save_path="image.jpg", matrix_save_path="matrix.jpg"):
    """
    RGB 모델의 성능을 테스트하고 리포트와 Confusion Matrix를 생성합니다.
    """
    print(f"\n===== Evaluating: {model_name} (RGB only) =====")
    
    # 데이터 로드 및 예측
    _, _, all_loader_rgb = load_data()
    preds, labels = evaluate_rgb(model_rgb, all_loader_rgb)
    

    # --- 1. Classification Report 생성 및 출력 ---
    print(classification_report(labels, preds, output_dict=False, zero_division=0))
    report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
    f1_score_comparison(report_dict, model_name, report_save_path)

    # --- 2. Confusion Matrix 생성 및 시각화 ---
    preds_labels = [CLASS_NAME[p] for p in preds]
    true_labels = [CLASS_NAME[l] for l in labels]
    matrix = confusion_matrix(true_labels, preds_labels, labels=CLASS_NAME)
    
    # 콘솔 출력용 DataFrame
    df_cm_for_print = pd.DataFrame(matrix, index=CLASS_NAME, columns=CLASS_NAME)
    print("\n--- Confusion Matrix ---")
    print(df_cm_for_print)

    # 이미지 저장용 Heatmap
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm_for_print, annot=True, fmt='d', cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(matrix_save_path)
    plt.close() # 중요: plt 객체를 닫아 메모리 누수 방지
    
    print(f"\nConfusion Matrix heatmap saved to: {matrix_save_path}")
    print("=" * (len(model_name) + 26))
    
def main():
    # 각 모델의 설정을 딕셔너리 리스트로 중앙에서 관리

    model_registry = {
        # timm_name : build_function
        "resnet50": build_resnet50,
        "resnet34": build_resnet34,
        "vit_base_patch16_224": build_vit_b_16, # timm 이름과 함수 매핑
        "vit_base_patch32_224": build_vit_b_32, # timm 이름과 함수 매핑
        "efficientnet_b0": build_efficientnet_b0,
        "efficientnet_b3": build_efficientnet_b3,
        "densenet121": build_densenet121,
        "densenet169": build_densenet169,
        # training_jobs에 있었던 다른 모델들도 필요하면 여기에 추가
    }


    model_configs = [
        # --- 1. ResNet 계열 ---
        {
            "model_name": "resnet50",
            "path": "./checkpoints/ResNet50_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_resnet50.jpg",
            "matrix_image_path": "image/confusion_matrix_resnet50.jpg"
        },
        {
            "model_name": "resnet34",
            "path": "./checkpoints/ResNet34_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_resnet34.jpg",
            "matrix_image_path": "image/confusion_matrix_resnet34.jpg"
        },

        # --- 2. ViT 계열 ---
        {
            "model_name": "vit_base_patch16_224", 
            "path": "./checkpoints/ViT_B_16_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_vit_base_patch16_224.jpg",
            "matrix_image_path": "image/confusion_matrix_vit_base_patch16_224.jpg"
        },
        {
            "model_name": "vit_base_patch32_224", 
            "path": "./checkpoints/ViT_B_32_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_vit_base_patch32_224.jpg",
            "matrix_image_path": "image/confusion_matrix_vit_base_patch32_224.jpg"
        },

        # --- 3. EfficientNet 계열 ---
        {
            "model_name": "efficientnet_b0",
            "path": "./checkpoints/EfficientNet_B0_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_efficientnet_b0.jpg",
            "matrix_image_path": "image/confusion_matrix_efficientnet_b0.jpg"
        },
        {
            "model_name": "efficientnet_b3",
            "path": "./checkpoints/EfficientNet_B3_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_efficientnet_b3.jpg",
            "matrix_image_path": "image/confusion_matrix_efficientnet_b3.jpg"
        },

        # --- 4. DenseNet 계열 ---
        {
            "model_name": "densenet121",
            "path": "./checkpoints/DenseNet121_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_densenet121.jpg",
            "matrix_image_path": "image/confusion_matrix_densenet121.jpg"
        },
        {
            "model_name": "densenet169",
            "path": "./checkpoints/DenseNet169_best.pth", # <-- 경로 수정
            "f1_score_image_path": "image/f1_score_densenet169.jpg",
            "matrix_image_path": "image/confusion_matrix_densenet169.jpg"
        }
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 설정 리스트를 순회하며 작업 수행
    for config in model_configs:
        print(f"\n===== Processing: {config['model_name']} =====")
        
        # 1. 모델 로딩 (책임 분리)
        model = load_model_from_checkpoint(
            config=config,
            num_classes=3, # 분류 클래스 수
            device=device,
            model_registry=model_registry # build 함수 딕셔너리 전달
        )

        # 2. 평가 (모델이 성공적으로 로드된 경우에만 진행)
        if model is not None:
            custom_evaluate(
                model,
                config['model_name'],
                config['f1_score_image_path'],
                config['matrix_image_path']
            )
        else:
            print(f"Skipping evaluation for {config['model_name']} due to loading failure.")


if __name__ == '__main__':
    main()
