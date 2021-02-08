import os
import h5py
import chainer
from chainer.dataset.dataset_mixin import DatasetMixin


def read_hdf5_file(path):
    f = h5py.File(path, 'r')
    sample = f["sample"][()]
    label = f["label"][()].astype("float32")
    f.close()
    return sample, label


def create_paths_list(dir):
    filenames = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".hdf5"):
                dirname = root.split(os.path.sep)[-1]
                filenames.append(os.path.join(dirname, file))

    return filenames


class CosmoDataset(DatasetMixin):
    def __init__(self, root='.', dtype=None):
        file_names = create_paths_list(root)
        paths = [path.strip() for path in file_names]
        self._paths = paths
        self._root = root
        self._dtype = chainer.get_dtype(dtype)

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        sample, label = read_hdf5_file(path)
        return sample, label



