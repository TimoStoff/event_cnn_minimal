from torch.utils.data import DataLoader
# local modules
from .dataset import DynamicH5Dataset
from utils.data import concatenate_datasets

class InferenceDataLoader(DataLoader):
    """
    """
    def __init__(self, data_path, num_workers=1, pin_memory=True, dataset_kwargs={}):
        dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
