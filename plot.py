import pickle
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_training_histories(file_path):
    """
    지정된 .pkl 파일에서 학습 이력 딕셔너리를 불러옵니다.
    
    Args:
        file_path (str): 불러올 .pkl 파일의 경로.
    
    Returns:
        dict: 파일에 저장된 all_histories 딕셔너리.
    """
    try:
        with open(file_path, 'rb') as f:
            all_histories = pickle.load(f)
        print(f"'{file_path}' 파일에서 데이터를 성공적으로 불러왔습니다.")
        return all_histories
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

def setup_matplotlib_font():
    """운영체제에 따라 matplotlib 폰트를 설정합니다."""
    system = platform.system()
    if system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        # 리눅스 환경: 'NanumGothic' 설치 필요
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지


def plot_training_history(all_histories, metric, title, ylabel, save_path='image/my_plot.png', epochs=16):
    """
    주어진 메트릭(손실 또는 정확도)에 대한 학습 이력을 그래프로 그립니다.
    
    Args:
        all_histories (dict): 모델별 학습 이력 데이터가 담긴 딕셔너리.
        metric (str): 그래프로 그릴 메트릭 키 ('val_loss' 또는 'val_acc').
        title (str): 그래프 제목.
        ylabel (str): y축 라벨.
        epochs (int): 그래프에 표시할 최대 에포크 수.
    """
    plt.figure(figsize=(12, 7))
    colors = {
        'ResNet18': '#33AFFF',
        'DenseNet': '#FFC300',
        'EfficientNet': '#33FFCE',
        'ViT': '#A633FF'
    }

    styles = {
        'ResNet18': {'marker': 'o', 'linestyle': '-'},
        'ViT': {'marker': 's', 'linestyle': '--'},
        'EfficientNet': {'marker': '^', 'linestyle': '-.'},
        'DenseNet': {'marker': 'D', 'linestyle': ':'}
    }

    # 각 모델의 학습 이력 그리기
    for model_name, history in all_histories.items():
        color = colors.get(model_name, 'gray')
        style = styles.get(model_name, {'marker': 'o', 'linestyle': '-'})
        
        plt.plot(
            range(1, len(history[metric]) + 1),
            history[metric],
            marker=style['marker'],
            linestyle=style['linestyle'],
            color=color,
            label=model_name,
            zorder=10
        )

    # 그래프 스타일 설정
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, 3.2 if 'loss' in metric else 1.0) # y축 범위 동적 설정
    plt.xticks(np.arange(1, epochs + 1, 3))
    plt.legend(
        loc='lower center',         
        bbox_to_anchor=(0.5, 0.95), 
        ncol=4,                     
        frameon=False,             
        fontsize=10                
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def f1_score_comparison(report, model_name, save_path):
    # 3. F1-Score 데이터 추출 및 준비
    # -----------------------------------------------------------------
    df = pd.DataFrame(report).transpose()
    # 클래스(Defect A, Defect B, No Defect)와 F1-Score만 추출합니다.
    # 'accuracy', 'macro avg', 'weighted avg' 등은 제외합니다.
    class_names = [name for name in df.index if name not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = df.loc[class_names, 'f1-score'].values

    # 4. 막대 그래프 시각화 (Matplotlib)
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # 막대 그래프 생성
    bars = plt.bar(class_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c']) # 각 클래스에 다른 색상 적용

    # 그래프 제목 및 축 라벨 설정
    plt.title(f'F1-Score Comparison ({model_name})', fontsize=16)
    plt.xlabel('Classification Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)

    # Y축 범위 설정 (0부터 1까지)
    plt.ylim(0, 1.05)

    # X축 레이블을 45도 회전하여 겹치지 않도록 설정 (필요 시)
    plt.xticks(rotation=0, ha='center', fontsize=10)

    # 각 막대 위에 정확한 F1-Score 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
                f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # 레이아웃 조정
    plt.savefig(save_path)

if __name__=='__main__':
    # 함수 호출하여 데이터 불러오기
    loaded_histories = load_training_histories("data/training_histories_ir.pkl")

    setup_matplotlib_font()
    plot_training_history(loaded_histories, 'val_loss', 'IR/ 모델별 학습 손실(Loss) 비교', '손실값 (Loss)', 'image/models_train_Loss_IR.png')
    plot_training_history(loaded_histories, 'val_acc', 'IR/ 모델별 학습 정확도(Accuracy) 비교', '정확도 (Accuracy)', 'image/models_train_Acc_IR.png')

    # 함수 호출하여 데이터 불러오기
    loaded_histories = load_training_histories("data/training_histories_rgb.pkl")

    setup_matplotlib_font()
    plot_training_history(loaded_histories, 'val_loss', 'RGB/ 모델별 학습 손실(Loss) 비교', '손실값 (Loss)', 'image/models_train_Loss_RGB.png')
    plot_training_history(loaded_histories, 'val_acc', 'RGB/ 모델별 학습 정확도(Accuracy) 비교', '정확도 (Accuracy)', 'image/models_train_Acc_RGB.png')

   