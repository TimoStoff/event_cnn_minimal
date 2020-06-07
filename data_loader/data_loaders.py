from torch.utils.data import DataLoader
# local modules
from .dataset import DynamicH5Dataset


class InferenceDataLoader(DataLoader):

    def __init__(self, data_path, num_workers=1, pin_memory=True, dataset_kwargs=None, ltype="H5"):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        elif ltype == "MMP":
            dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
