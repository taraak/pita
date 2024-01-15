import numpy as np

arr = np.load("../../data/all_data_LJ55-1000-part1.npy")
rng = np.random.default_rng(42)
idx = np.arange(len(arr))
rng.shuffle(idx)
arr_test = arr[idx[:10000]]
print(arr_test.shape)
print(idx[:10])
np.save("../../data/lj55_idx.npy", idx[:10000])
np.save("../../data/test_split_LJ55-1000-part1.npy", arr_test)
