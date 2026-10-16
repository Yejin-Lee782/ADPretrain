import numpy as np
from torch.utils.data import Sampler
from .real_iad import RealIAD


class BalancedSampler(Sampler):
    def __init__(self, dataset: RealIAD, batch_size: int):
        super(BalancedSampler, self).__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = len(dataset) // batch_size
           
        labels = []
        for view in dataset.data_items_by_view.keys():
            for data_item in dataset.data_items_by_view[view]:
                label = data_item['label']
                labels.append(label)
        labels = np.array(labels)
        
        normal_inds = np.argwhere(labels == 0).flatten()
        abnormal_inds = np.argwhere(labels == 1).flatten()

        self.normal_generator = self.randomGenerator(normal_inds)
        self.abnormal_generator = self.randomGenerator(abnormal_inds)

        self.n_normal = batch_size // 2
        self.n_abnormal = batch_size - self.n_normal

    def randomGenerator(self, array):
        while True:
            permuted_array = np.random.permutation(array)
            for i in permuted_array:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))  # get the sampling normal inds

            for _ in range(self.n_abnormal):
                batch.append(next(self.abnormal_generator))  # get the sampling abnormal inds

            yield batch


class BalancedSampler2(Sampler):
    def __init__(self, dataset: RealIAD, batch_size: int):
        super(BalancedSampler2, self).__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
           
        labels = []
        for view in dataset.data_items_by_view.keys():
            for data_item in dataset.data_items_by_view[view]:
                label = data_item['label']
                labels.append(label)
        labels = np.array(labels)
        
        normal_inds = np.argwhere(labels == 0).flatten()
        abnormal_inds = np.argwhere(labels == 1).flatten()

        self.steps_per_epoch = len(abnormal_inds) * 3 // batch_size
        
        self.normal_generator = self.randomGenerator(normal_inds)
        self.abnormal_generator = self.randomGenerator(abnormal_inds)

        self.n_normal = batch_size // 2
        self.n_abnormal = batch_size - self.n_normal

    def randomGenerator(self, array):
        while True:
            permuted_array = np.random.permutation(array)
            for i in permuted_array:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))  # get the sampling normal inds

            for _ in range(self.n_abnormal):
                batch.append(next(self.abnormal_generator))  # get the sampling abnormal inds

            yield batch
