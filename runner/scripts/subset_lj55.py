import numpy as np

arr = np.load("../../data/all_data_LJ55-1000-part1.npy")
rng = np.random.default_rng(42)
idx = np.arange(len(arr))
rng.shuffle(idx)
arr_test = arr[idx[:10000]]
arr_val = arr[idx[10000:20000]]
arr_train = arr[idx[20000:30000]]
print(arr_test.shape)
print(idx[:10])
print(idx[10000:10010])
print(idx[20000:20010])
np.save("../../data/lj55_idx.npy", idx[:30000])
np.save("../../data/test_split_LJ55-1000-part1.npy", arr_test)
np.save("../../data/val_split_LJ55-1000-part1.npy", arr_val)
np.save("../../data/train_split_LJ55-1000-part1.npy", arr_train)
