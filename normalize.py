import numpy as np

def normalize(X_dataset):
    X_normalized = np.ndarray(X_dataset.shape, dtype=np.float32)
    X_normalized[:, :, :, 0] = X_dataset[:, :, :, 0] / 256
    X_normalized[:, :, :, 1] = X_dataset[:, :, :, 1] / 256
    X_normalized[:, :, :, 2] = X_dataset[:, :, :, 2] / 256
    # print(X_normalized[:, :, :, 0])

    # X_normalized = X_dataset

    # Try normalizing to grayscale
    #gray = np.mean(rgb, -1)

    # X_normalized[:, :, :, 0] = X_dataset[:, :, :, 0] / 256
    # from skimage import color

    # Grayscale and normalize
    # 0.2125 R + 0.7154 G + 0.0721 B
    # 0.299 R + 0.587 G + 0.114 B

    # X_normalized = np.ndarray(X_dataset.shape, dtype=np.float32)
    # X_normalized[:, :, :, 0] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.299)) / 3) / 256
    # X_normalized[:, :, :, 1] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.587)) / 3) / 256
    # X_normalized[:, :, :, 2] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.114)) / 3) / 256

    # print("normalized shape")
    # print(X_normalized.shape)
    # print(X_normalized[:, :, :, 0])

    return X_normalized
