import numpy as np
from torch.utils.data import Dataset

def Row(x):
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def ins_gen(n_j, n_m, low, upper,seed=None):
    if seed != None:
        np.random.seed(seed)

    time0 = np.random.randint(low=low, upper=upper, size=(n_j, n_m,n_m-1))
    time1=np.random.randint(low=1, upper=upper, size=(n_j, n_m,1))
    times=np.concatenate((time0,time1),-1)
    for i in range(n_j):
        times[i]= Row(times[i])
    return times


def override(fn):
    return fn


class Dataset(Dataset):

    def __init__(self,n_j, n_m, low, upper,num_samples=1000,seed=None,  offset=0, distribution=None):
        super(Dataset, self).__init__()
        self.data_set = []
        if seed != None:
            np.random.seed(seed)
        time0 = np.random.uniform(low=low, upper=upper, size=(num_samples, n_j, n_m, n_m - 1))
        time1 = np.random.uniform(low=0, upper=upper, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = Row(times[j][i])
        self.data = np.array(times)
        self.size = len(self.data)

    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]



