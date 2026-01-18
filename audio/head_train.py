# head_train.py
import os, joblib, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from embeddings import wav_to_embedding

DATASET_ROOT = "dataset"  # audio/dataset/...

AUDIO_EXTS = {".wav"}

def crawl_dataset(root):
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    
    # Only process red, blue, yellow folders - WHITE DISABLED
    target_classes = ["red", "blue", "yellow"]  # Removed "white"
    classes = [cls for cls in classes if cls in target_classes]
    
    files, labels = [], []
    for ci, cls in enumerate(classes):
        if cls not in target_classes:
            print(f"Skipping {cls} folder (disabled)")
            continue
        print(f"Processing {cls} folder...")
        
        class_files = []
        for f in (root/cls).iterdir():
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                class_files.append(str(f))
                labels.append(ci)
        
        print(f"  Found {len(class_files)} audio files")
        files.extend(class_files)
    return classes, files, labels

def main():
    classes, files, labels = crawl_dataset(DATASET_ROOT)
    if not files:
        raise SystemExit(f"No audio found under {DATASET_ROOT}/<class>/")

    print(f"Classes: {classes}")
    print(f"Total files: {len(files)}")

    # Compute embeddings
    X = []
    for p in tqdm(files, desc="Embedding"):
        X.append(wav_to_embedding(p))
    X = np.stack(X, axis=0)
    y = np.array(labels)

    # Train/val split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # Multinomial logistic regression head
    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",            # good for multiclass
        multi_class="multinomial",
        C=2.0                     # you can tune 0.5–5
    )
    clf.fit(Xtr, ytr)

    # Eval
    yhat = clf.predict(Xte)
    print(classification_report(yte, yhat, target_names=classes, digits=3))

    # Save
    joblib.dump({"clf": clf, "classes": classes}, "bird_head.joblib")
    print("Saved bird_head.joblib")

if __name__ == "__main__":
    main()
