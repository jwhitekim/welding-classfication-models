import os
import shutil
import random

def split_dataset(modalities, input_dir="datasets", output_dir="dataset_split",train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.seed(42)  # 재현성을 위해 시드 고정
    
    classes = os.listdir(f"{input_dir}/{modalities}")  # ['Good', 'Bad']
    for cls in classes:
        cls_dir = os.path.join(input_dir, modalities, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        # 각 split 폴더 생성 후 복사
        for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(output_dir, modalities, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)

    print("✅ Dataset split complete!")

if __name__=='__main__':
    split_dataset("IR", "datasets", "dataset_split")
    split_dataset("RGB", "datasets", "dataset_split")