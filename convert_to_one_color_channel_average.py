# X_normalized = np.ndarray((n_train, X_dataset.shape[1], X_dataset.shape[2], 1), dtype=np.float32)
# X_normalized[:, :, :, 0] = ( (X_dataset[:, :, :, 0] + X_dataset[:, :, :, 1] + X_dataset[:, :, :, 2]) / 3) / 256
