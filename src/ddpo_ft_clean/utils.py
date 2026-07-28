import grain
from src.sequence_inference import build_triplets
import numpy as np

class LR_input_source(grain.sources.RandomAccessDataSource):
    def __init__(self, path, train_idx):
        self.data = np.load(path, mmap_mode="r")[train_idx]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def _getshape(self):
        return self.data.shape


class BuildTripletsFlatMap(grain.experimental.FlatMapTransform):

    def __init__(self, mean, std, max_fan_out):
        self.mean = mean
        self.std = std
        self.max_fan_out = max_fan_out # can build 318 triplets with a sequence of length 320

    def flat_map(self, seq):
        triplets = build_triplets(seq, self.mean, self.std)
        for t in triplets:
            yield t



