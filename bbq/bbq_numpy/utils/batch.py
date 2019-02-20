import numpy as np


def batch_generator(arrays, batch_size, wrap_last_batch=False):
    """
    Batch generator() function for yielding [x_train, y_train] batch slices for
    numpy arrays
    Appropriately deals with looping back around to the start of the dataset
    Generate batches, one with respect to each array's first axis.
    :param arrays:[array, array]  or [array, None]...
                  e.g. [X_trn, Y_trn] where X_trn and Y_trn are ndarrays
    :param batch_size: batch size
    :param wrap_last_batch: whether the last batch should wrap around dataset
    to include first datapoints (True), or be smaller to stop at the end of
    the dataset (False).
    :return:
    """
    starts = [0] * len(
        arrays)  # pointers to where we are in iteration     --> [0, 0]
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                if wrap_last_batch:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                else:
                    batch = array[start:]
                    starts[i] = 0
            batches.append(batch)
        yield batches
