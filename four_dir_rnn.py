from keras.layers import Conv2D, Lambda, K, Reshape, Bidirectional, merge, CuDNNLSTM


def four_dir_rnn(layer, filter_num):
    """

    :param layer:
    :param filter_num:
    :return:
    """
    # layer为张量
    # int_shape(): Returns the shape of tensor or variable as a tuple of int or None entries.
    # layer_shape = K.int_shape(layer)  # (None, 512, 7, 7)
    layer_shape = filter_num
    # 1 x 1卷积
    layer = Conv2D(filter_num[-1], kernel_size=(1, 1))(layer)  # (?, 512, 7, 7)
    # layer_transpose = K.tf.transpose(layer, perm=[0, 2, 1, 3])
    # 会报错当用K.tf.transpose()时会报“tensor” has no attri "_keras_history"
    layer_transpose = Lambda(lambda x: K.tf.transpose(layer, perm=[0, 2, 1, 3]))(layer)
    layer = Reshape((-1, filter_num[-1]))(layer)
    layer_transpose = Reshape((-1, filter_num[-1]))(layer_transpose)  # (?, ?, 512)
    renet1 = Bidirectional(CuDNNLSTM(filter_num[-1], return_sequences=True))(layer)  # (?, ?, 1024)
    renet2 = Bidirectional(CuDNNLSTM(filter_num[-1], return_sequences=True))(layer_transpose)  # (?, ?, 1024)
    renet = merge([renet1, renet2], mode='concat')  # (?, ?, 2048)
    renet = Reshape((layer_shape[1], layer_shape[1], -1))(renet)  # (?, ?, 7, 7)
    renet = Conv2D(filter_num[-1], kernel_size=(1, 1))(renet)
    return renet
