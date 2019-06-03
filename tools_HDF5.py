import numpy
import h5py
# ----------------------------------------------------------------------------------------------------------------------
class HDF5_store(object):
    def __init__(self, filename, object_shape=None, dtype=numpy.float32, compression="gzip", chunk_len=1):
        self.filename = filename
        self.dataset_name = 'dataset'
        self.object_shape = object_shape
        self.size = 0

        if object_shape is not None:
            # create
            with h5py.File(self.filename, mode='w') as f:
                self.dataset = f.create_dataset(self.dataset_name, shape=(0,) + object_shape, maxshape=(None,) + object_shape, dtype=dtype, compression=compression, chunks=(chunk_len,) + object_shape)
        else:
            with h5py.File(self.filename, mode='r') as f:
                self.size = len(f[self.dataset_name][:])
# ----------------------------------------------------------------------------------------------------------------------
    def append(self, values):
        with h5py.File(self.filename, mode='a') as f:
            dset = f[self.dataset_name]
            dset.resize((self.size + 1,) + self.object_shape)
            dset[self.size] = [values]
            self.size += 1
            f.flush()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self,i):
        f = h5py.File(self.filename, mode='r')
        res = f[self.dataset_name][i]
        f.close()
        return res