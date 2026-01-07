import pickle

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time # (print를 위해 import하는 것이 좋습니다)

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 공통 전처리
from torchvision import transforms

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),

        # 1. 50% 확률로 좌우 반전
        transforms.RandomHorizontalFlip(),
        
        # 2. 50% 확률로 상하 반전 (회전 불변 데이터에 효과적)
        transforms.RandomVerticalFlip(), 

        # 3. -180도 ~ +180도 사이에서 무작위 회전 (모든 각도 커버)
        transforms.RandomRotation(180), 
        
        # 4. 색감 변화
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def load_data(batch_size=32):
    train_data = datasets.ImageFolder(root=f"dataset/train", transform=transform["train"])
    val_data   = datasets.ImageFolder(root=f"dataset/val",   transform=transform["val"])
    test_data  = datasets.ImageFolder(root=f"dataset/test",  transform=transform["val"])
    
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, shuffle=False),
            DataLoader(test_data, batch_size=batch_size, shuffle=False))


def build_resnet34(num_classes=3):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def build_resnet50(num_classes=3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def build_vit_b_16(num_classes=3):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model.to(device)

def build_vit_b_32(num_classes=3):
    model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model.to(device)

def build_efficientnet_b0(num_classes=3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)

def build_efficientnet_b3(num_classes=3):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)

def build_densenet121(num_classes=3):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model.to(device)

def build_densenet169(num_classes=3):
    model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model.to(device)


def load_model_from_checkpoint(config, num_classes, device, model_registry):
    """
    주어진 설정에 따라 torchvision 모델을 생성하고 체크포인트 가중치를 불러옵니다.
    (timm 대신 build 함수 레지스트리 사용)
    """
    model_name = config['model_name']
    checkpoint_path = config['path']
    print(f"Loading model: {model_name}...")
    
    try:
        # --- 1. Model Build (Torchvision + build_func) ---
        
        # model_registry에서 빌드 함수를 찾습니다.
        if model_name not in model_registry:
            print(f"[Error] '{model_name}'에 해당하는 build 함수가 model_registry에 없습니다.")
            return None
        
        build_func = model_registry[model_name]
        
        # build_func를 사용해 모델 구조 생성
        # (build_func 내부에서 .to(device)를 이미 처리함)
        model = build_func(num_classes=num_classes) 

    except Exception as e:
        print(f"[Error] 모델 빌드 중 오류 발생 ({model_name}): {e}")
        return None

    try:
        # --- 2. Load Weights (KeyError 수정) ---
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError # FileNotFoundError를 강제로 발생시킴
            
        print(f"  -> Loading weights from: {checkpoint_path}")
        
        # 'state_dict' 키 없이 가중치(state_dict) 자체를 로드합니다.
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # --- 3. Apply Weights ---
        model.load_state_dict(state_dict)
        model.to(device) # device로 확실하게 이동

        print(f"Model '{model_name}' loaded successfully.")
        return model

    except FileNotFoundError:
        print(f"[Warning] Checkpoint file not found for {model_name} at: {checkpoint_path}.")
        return None
    except Exception as e:
        # (예: state_dict 키 불일치 등)
        print(f"[Error] An error occurred while loading weights for {model_name}: {e}")
        return None
    

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, save_path="models/best_model.pth", patience=10):
    """
    모델을 학습하고 조기 종료 및 최고 모델 저장을 수행합니다.

    Args:
        model (torch.nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터로더
        val_loader (DataLoader): 검증 데이터로더
        num_epochs (int): 최대 에포크 수
        lr (float): 학습률
        save_path (str): 최고 성능 모델이 저장될 경로
        patience (int): 조기 종료를 위한 '인내심' (val_loss가 이 횟수만큼 개선되지 않으면 중지)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training using device: {device}\n")

    # --- 조기 종료 및 최고 성능 저장을 위한 변수 ---
    best_val_loss = float('inf') # val_loss는 낮을수록 좋음 (무한대로 초기화)
    best_acc = 0.0               # (참고용) best_val_loss일 때의 val_acc
    epochs_no_improve = 0        # 성능이 개선되지 않은 에포크 카운트
    # --------------------------------------------

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        start_time = time.time() # 에포크 시작 시간
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader
            
            # loader가 None인지 체크 (지난번 오류 방지)
            if loader is None:
                print(f"'{phase}' loader is None, skipping phase.")
                continue

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) # .data 추가 (더 안전함)

            # 데이터셋 크기 확인
            if len(loader.dataset) == 0:
                print(f"Error: '{phase}' loader.dataset is empty.")
                continue

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}", flush=True)

            # --- 기록 및 조기 종료 로직 ---
            if phase == "train":
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # val_loss를 기준으로 최고 모델 저장 및 조기 종료 카운트
                if epoch_loss < best_val_loss:
                    print(f"  Validation loss improved ({best_val_loss:.4f} -> {epoch_loss:.4f}). Saving model...")
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc.item() # 이 시점의 정확도를 최고 정확도로 기록
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    epochs_no_improve += 1
                    print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s). (Patience: {patience})")
            # --------------------------------

        epoch_time = time.time() - start_time
        print(f"Epoch complete in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")

        # --- 에포크 종료 후 조기 종료 여부 체크 ---
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break
        # ----------------------------------------
    
    # --- 학습 종료 후 ---
    print("\nTraining finished.")
    print(f"Best Validation Loss: {best_val_loss:.4f}, Best Validation Acc: {best_acc:.4f}")
    
    # 가장 좋았던 가중치를 다시 로드 (학습이 끝난 후 모델은 'best' 상태가 아닐 수 있으므로)
    if os.path.exists(save_path):
        print(f"Loading best model weights from {save_path}...")
        model.load_state_dict(torch.load(save_path))
    else:
        print("Warning: No best model was saved (perhaps training was too short or 'val' loader was missing).")

    return history


def main():
        # 1. 모든 학습 작업을 하나의 리스트로 명확하게 정의
    training_jobs = [
        # --- 1. ResNet 계열 ---
        {'model_name': 'ResNet50', 'build_func': build_resnet50, 'num_epochs': 50},
        {'model_name': 'ResNet34', 'build_func': build_resnet34, 'num_epochs': 50}, # (작은 버전)

        # --- 3. ViT (Vision Transformer) 계열 ---
        {'model_name': 'ViT_B_16', 'build_func': build_vit_b_16, 'num_epochs': 50},
        # {'model_name': 'ViT_L_16', 'build_func': build_vit_l_16, 'num_epochs': 50}, # (큰 버전)
        {'model_name': 'ViT_B_32', 'build_func': build_vit_b_32, 'num_epochs': 50}, # (다른 패치)

        # --- 4. EfficientNet 계열 ---
        {'model_name': 'EfficientNet_B0', 'build_func': build_efficientnet_b0, 'num_epochs': 50},
        # (B3 등 다른 버전을 추가하려면 build_efficientnet_b3 함수 정의 후 여기에 추가)
        {'model_name': 'EfficientNet_B3', 'build_func': build_efficientnet_b3, 'num_epochs': 50}, 

        # --- 5. DenseNet 계열 ---
        {'model_name': 'DenseNet121', 'build_func': build_densenet121, 'num_epochs': 50},
        {'model_name': 'DenseNet169', 'build_func': build_densenet169, 'num_epochs': 50}, # (큰 버전)
        # {'model_name': 'DenseNet201', 'build_func': build_densenet201, 'num_epochs': 50}, # (더 큰 버전)
    ]

    # --- 1. 모델 가중치를 저장할 폴더 생성 ---
    # (예: /checkpoints/ResNet50_best.pth)
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 2. 하이퍼파라미터 설정 ---
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 50 # 조기 종료가 있으니 넉넉하게 설정
    PATIENCE = 10   # 조기 종료 '인내심'

    # --- 5. 메인 학습 루프 ---
    all_histories = {} # 모든 모델의 학습 기록을 저장할 딕셔너리
    train_loader, val_loader, test_loader = load_data()

    for job in training_jobs:
        model_name = job['model_name']
        build_func = job['build_func']
        # num_epochs = job['num_epochs'] # MAX_EPOCHS로 통일

        print(f"\n{'='*20}")
        print(f"🚀 [모델 학습 시작]: {model_name}")
        print(f"{'='*20}")

        # --- 3. 모델별 고유 저장 경로 생성 ---
        model_save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        print(f"   -> 모델 저장 경로: {model_save_path}")

        # 4. 모델 빌드
        # (build_func가 device 인자를 받도록 수정했거나,
        #  train_model 함수 내부에서 to(device)가 실행되어야 함)
        model = build_func(num_classes=3) 

        # 5. 모델 학습 실행
        history = train_model(
            model=model,
            train_loader=train_loader, # 또는 그냥 train_loader
            val_loader=val_loader,   # 또는 그냥 val_loader
            num_epochs=MAX_EPOCHS,
            lr=LEARNING_RATE,
            save_path=model_save_path, # <-- ★★★ 고유한 경로 전달 ★★★
            patience=PATIENCE
        )
        
        # 6. 결과 저장
        all_histories[model_name] = history
        print(f"🏁 [모델 학습 완료]: {model_name}")

    print("\n\n🎉 === 모든 모델 학습 완료! === 🎉")

    # 6. 모든 학습 이력을 파일 하나로 저장
    with open("all_training_histories.pkl", "wb") as f:
        pickle.dump(all_histories, f)
    
    print("\nAll training jobs are complete. Histories saved to 'all_training_histories.pkl'.")

if __name__ == '__main__':
    main()


