import numpy as np

class BanditData(object):
    """
    - stores a sample per step
    - samples batches of training data
    - sampling uniformly random
    - assumes i.i.d samples
    """

    def __init__(self, batch_size, epoch_len=32):
        self._epoch_len = epoch_len
        self._batch_size = batch_size
        self._n_samples = 0

        # design matrix + targets (y)
        # the last column contains the targets
        self._D = None

    def sample_batches(self):
        """
        epoch_len: number of batches

        total number of samples
        = epoch_len * batch_size
        """
        n_samples, _ = self._D.shape
        indices = np.arange(n_samples)

        if self._n_samples < self._batch_size * self._epoch_len:
            batch_len = self.n_samples // self._batch_size
        else:
            batch_len = self._epoch_len

        for _ in range(batch_len):
            indices = np.random.choice(indices,
                                       size=self._batch_size,
                                       replace=False)
            X = self._D[indices, :-1]
            y = self._D[indices, -1][:, None]

            yield (X, y)


    def sample_most_recent(self, size=None):
        n_samples, _ = self._D.shape
        if size is None:
            return self._D[-self._batch_size:, :]
        else:
            return self._D[-size:, :]


    def add_sample(self, x, y):
        """
        assumes one sample update.
        in an online setting.
        """
        if self._D is None:
            X = np.array(x).reshape(1, len(x))
            y = np.array(y).reshape(1, 1)
            self._D = np.hstack( (X, y) )
        else:
            assert np.isscalar(y)
            sample = np.concatenate( (x, [y]) )
            self._D = np.vstack( (self._D, sample) )

        self._n_samples += 1

    @property
    def n_samples(self):
        return self._n_samples


    def __len__(self):
        return self._n_samples

