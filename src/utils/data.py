import numpy as np


def horizontal_flip(nnet):

    print("Horizontal flips of data set...")

    flipped = nnet.X_train[:, :, :, ::-1]
    nnet.X_train = np.concatenate((nnet.X_train, flipped))
    nnet.y_train = np.concatenate((nnet.y_train, nnet.y_train))
    nnet.n_samples *= 2

    # shuffle data set
    perm = np.arange(nnet.n_samples)
    np.random.shuffle(perm)

    nnet.X_train = nnet.X_train[perm]
    nnet.y_train = nnet.y_train[perm]

    print("Done.")


def data_augmentation(nnet):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """

    assert nnet.n_samples % nnet.batch_size == 0
    n_batches = nnet.n_samples // nnet.batch_size

    data_shape = nnet.X_train[:nnet.batch_size].shape
    cropsize = int(0.9 * len(nnet.X_train[0, 0, :, 0]))

    augmented_data = np.zeros((nnet.x_augmentation * nnet.n_samples,
                               data_shape[1], data_shape[2], data_shape[3]),
                              dtype=nnet.floatX)
    augmented_labels = np.zeros(nnet.x_augmentation * nnet.n_samples,
                                dtype=np.int32)

    for k in range(nnet.x_augmentation):
        for i in range(n_batches):

            indices = i * nnet.batch_size + np.arange(nnet.batch_size)

            augmented_data[k * nnet.n_samples + indices] = \
                augment_batch(nnet.X_train[indices], cropsize)
            augmented_labels[k * nnet.n_samples + indices] = \
                nnet.y_train[indices]

    nnet.X_train = np.asarray(augmented_data)
    nnet.y_train = np.asarray(augmented_labels)
    nnet.n_samples = len(nnet.X_train)

    ii = np.arange(nnet.n_samples)
    np.random.shuffle(ii)

    nnet.X_train = nnet.X_train[ii]
    nnet.y_train = nnet.y_train[ii]


def augment_batch(data, cropsize):

    data_out = np.zeros_like(data)

    crop_xs, crop_ys, flag_mirror = get_params_crop_and_mirror(data.shape,
                                                               cropsize)

    # mirror
    data[np.where(flag_mirror)] = data[np.where(flag_mirror)][:, :, :, ::-1]

    # random crop
    data_out[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :] = \
        data[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

    return data_out


def get_params_crop_and_mirror(data_shape, cropsize):

    rand = np.random.rand(2)

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = np.round(rand[0] * center_margin * 2).astype(np.int32)
    crop_ys = np.round(rand[1] * center_margin * 2).astype(np.int32)

    flag_mirror = np.random.randint(2, size=data_shape[0])

    return crop_xs, crop_ys, flag_mirror
