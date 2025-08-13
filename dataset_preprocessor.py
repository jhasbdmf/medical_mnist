import os
import numpy as np
from PIL import Image

def build_and_save_datasets(root_dir="./dataset",
                            image_size=(64, 64),
                            seed=42):
    # 1) reproducible shuffle
    np.random.seed(seed)

    # 2) find class subfolders
    classes = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    # 3) collect image paths + labels
    paths, labels = [], []
    for class_idx, class_name in enumerate(classes):
        folder = os.path.join(root_dir, class_name)
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, fname))
                labels.append(class_idx)

    N = len(paths)
    H, W = image_size

    # 4) preallocate arrays
    cnn_x = np.empty((N, H, W), dtype=np.uint8)
    cnn_y = np.array(labels,     dtype=np.uint8)
    mlp_x = np.empty((N, H * W), dtype=np.uint8)
    mlp_y = cnn_y.copy()

    # 5) load & fill
    for i, path in enumerate(paths):
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        if img.size != (W, H):
            img = img.resize((W, H))
        arr = np.asarray(img, dtype=np.uint8)
        cnn_x[i] = arr
        mlp_x[i] = arr.ravel()

    # 6) shuffle in unison
    perm = np.random.permutation(N)
    cnn_x, cnn_y = cnn_x[perm], cnn_y[perm]
    mlp_x, mlp_y = mlp_x[perm], mlp_y[perm]

    # 7) save each array in its own uncompressed .npy file
    np.save("cnn_x.npy", cnn_x)
    np.save("cnn_y.npy", cnn_y)
    np.save("mlp_x.npy", mlp_x)
    np.save("mlp_y.npy", mlp_y)

    print(f"Saved {N} samples to:")
    print("  • cnn_x.npy  →", cnn_x.shape)
    print("  • cnn_y.npy  →", cnn_y.shape)
    print("  • mlp_x.npy  →", mlp_x.shape)
    print("  • mlp_y.npy  →", mlp_y.shape)


if __name__ == "__main__":
    build_and_save_datasets(root_dir="./dataset")