from torch.utils.data import DataLoader
# local modules
from .dataset import DynamicH5Dataset, MemMapDataset, SequenceDataset
from utils.data import concatenate_subfolders, concatenate_datasets

class InferenceDataLoader(DataLoader):

    def __init__(self, data_path, num_workers=1, pin_memory=True, dataset_kwargs=None, ltype="H5"):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        elif ltype == "MMP":
            dataset = MemMapDataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


class HDF5DataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class MemMapDataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
