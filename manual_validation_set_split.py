# n_valid = int(n_train * .10)
# print("Number of validation =", n_valid)

# #print("Number of validation labels", y_train.shape)
# #print(X_train.shape)
# #print(y_train.shape)

# X_valid = np.ndarray((n_valid, X_train.shape[1], X_train.shape[2], X_train.shape[3]), dtype=np.float32)
# y_valid = np.ndarray(n_valid, dtype=np.int32)

# X_valid[0:n_valid, :, :, :] = X_train[0:n_valid, :, :, :]

# X_train = X_train[n_valid:, :, :, :]

# y_valid[0:n_valid] = y_train[0:n_valid]

# y_train = y_train[n_valid:]

print(X_valid.shape)
print(X_train.shape)

print(y_valid.shape)
print(y_train.shape)

print(y_train)
