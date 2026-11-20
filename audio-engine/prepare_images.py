"""ÉTAPE 1 bis (CNN) — comme le TP7 : .wav -> images PNG du mel-spectrogramme, puis découpage en dossiers K-Fold.

    python prepare_images.py

Résultat (même organisation que le TP7 : un dossier par classe, lisible directement) :
    data/images/
      all/normal/*.png, all/abnormal/*.png          <- toutes les images
      train_kfold1/ test_kfold1/                    <- fold 1 : apprentissage / validation
      train_kfold2/ test_kfold2/
      train_kfold3/ test_kfold3/
      test/                                         <- 20 % mis de côté : note finale du VOTE

Différences avec le TP7 (pour comparer le CNN au SVM de façon juste) :
  - TP7 : KFold(3) sur 100 % des images, le "test" de chaque fold sert aussi de note finale.
  - ici : 20 % mis de côté AVANT (dossier test/), puis StratifiedKFold(3) sur les 80 % restants
          -> EXACTEMENT les mêmes sons que pour le SVM (le découpage ne dépend que des labels y et de random_state=42).
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

from prepare_data import list_files, mel_image

OUT = Path("data/images")
CLASSES = {0: "normal", 1: "abnormal"}
N_FOLDS = 3


def wav_to_mel_images():
    """Comme Wav_To_melSpect_imgs du TP7 : un PNG par .wav, rangé dans le dossier de sa classe.
    Le numéro devant le nom garde l'ordre de list_files() -> même ordre que features.npz."""
    names, labels = [], []
    for i, (path, label) in enumerate(list_files()):
        name = f"{CLASSES[label]}/{i:05d}_{Path(path).stem}.png"
        try:
            Image.fromarray(mel_image(path)).save(OUT / "all" / name)
        except Exception:
            print("Fichier ignoré (corrompu) :", path)
            continue
        names.append(name)
        labels.append(label)
    return np.array(names), np.array(labels)


def copy_images(names, folder):
    """Comme moveTrain / moveTest du TP7 : copie les images dans un dossier train/test."""
    for name in names:
        shutil.copy(OUT / "all" / name, OUT / folder / name)


if __name__ == "__main__":
    if OUT.exists():
        print("Déjà fait :", OUT, "(supprime-le pour recommencer)")
        exit()

    # 1. .wav -> PNG
    folders = ["all", "test"] + [f"{s}_kfold{k}" for k in range(1, N_FOLDS + 1) for s in ("train", "test")]
    for folder in folders:
        for cls in CLASSES.values():
            (OUT / folder / cls).mkdir(parents=True, exist_ok=True)
    names, y = wav_to_mel_images()
    print(f"{len(y)} images -> normal: {(y == 0).sum()}, pathologique: {(y == 1).sum()}")

    # Vérification : mêmes labels, même ordre que features.npz (sinon les folds seraient différents du SVM)
    if Path("data/features.npz").exists():
        same = np.array_equal(np.load("data/features.npz")["y"], y)
        print("Même ordre que features.npz :", "OUI" if same else "NON -> les folds ne seront pas les mêmes que le SVM !")

    # 2. 20 % mis de côté pour la note finale (mêmes paramètres que kfold.load_split)
    train_names, test_names, y_train, _ = train_test_split(names, y, test_size=0.2, stratify=y, random_state=42)
    copy_images(test_names, "test")

    # 3. K-Fold stratifié sur les 80 % restants (mêmes paramètres que kfold.kfold_scores)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for k, (tr, val) in enumerate(skf.split(train_names, y_train), start=1):
        copy_images(train_names[tr], f"train_kfold{k}")
        copy_images(train_names[val], f"test_kfold{k}")
        print(f"fold {k} : {len(tr)} images d'apprentissage, {len(val)} de validation")
    print(f"test : {len(test_names)} images (note finale)")
