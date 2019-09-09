import tensorflow as tf

def oct_conv_first_layer(x, channel_num, alpha, kernel_size=3, activation=tf.nn.tanh):
    H_channel_num = int(channel_num * alpha // 1)  # by alpha, I split channel to high freq and low freq chuncks
    L_channel_num = channel_num - H_channel_num

    H_x = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 activation=activation)(x)

    # since low freq catch the spatial stucture rather than catching detail, we use pooling on Low freq parts
    L_pooling = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(x)
    L_x = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 activation=activation)(L_pooling)

    return H_x, L_x

def oct_conv_block(H_x, L_x, channel_num, alpha, kernel_size=3, activation=tf.nn.tanh):
    H_channel_num = int(channel_num * alpha // 1)  # by alpha, I split channel to high freq and low freq chuncks
    L_channel_num = channel_num - H_channel_num

    H2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)

    # # dilation add-on
    # H2dilation = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
    #                                     dilation_rate=2,
    #                                     kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)
    # H2H = tf.concat([H2H, H2dilation], axis=3)
    # H2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
    #                              kernel_initializer=tf.keras.initializers.glorot_normal())(H2H)

    H2L = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(H_x)
    H2L = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(H2L)

    L2L = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(L_x)

    # upsampling to H freq parts size
    L2H_raw = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(L_x)
    L2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(L2H_raw)

    # # dilation add-on
    # L2dilation = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
    #                                     dilation_rate=2,
    #                                     kernel_initializer=tf.keras.initializers.glorot_normal())(L2H_raw)
    # L2H = tf.concat([L2H, L2dilation], axis=3)
    # L2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
    #                              kernel_initializer=tf.keras.initializers.glorot_normal())(L2H)

    return activation((H2H + L2H) / 2), activation((L2L + H2L) / 2)

def oct_conv_final_layer(H_x, L_x, channel_num, kernel_size=3, activation=tf.nn.tanh):
    L2H = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(L_x)
    L2H = tf.keras.layers.Conv2D(channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(L2H)
    H2H = tf.keras.layers.Conv2D(channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                 kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)

    return activation((H2H + L2H) / 2)

# def transformer_block(input_layer, attention_head_num, scope=None):
#     with tf.variable_scope(scope, 'transformer', reuse=tf.AUTO_REUSE):
#         # input_shape = input_layer.shape
#         input_shape = list(input_layer.shape)
#         input_shape[0] = -1
#         # flatten_layer = tf.keras.layers.Flatten()(input_layer.outputs)
#         flatten_layer = tf.reshape(input_layer, [-1, input_shape[1] * input_shape[2], input_shape[3]])
#         flatten_shape = flatten_layer.shape
#         # multi-head_attention
#
#         dmodel = int(flatten_shape[2])  # channel num
#
#         dv = dmodel // attention_head_num
#
#         Qs = [tf.layers.dense(flatten_layer, dv,
#                               kernel_initializer=tf.keras.initializers.he_normal(), name="Q_dense1") for head in
#               range(attention_head_num)]
#
#         Ks = [tf.layers.dense(flatten_layer, dv,
#                               kernel_initializer=tf.keras.initializers.he_normal(), name="K_dense1") for head in
#               range(attention_head_num)]
#
#         Vs = [tf.layers.dense(flatten_layer, dv,
#                               kernel_initializer=tf.keras.initializers.he_normal(), name="V_dense1") for head in
#               range(attention_head_num)]
#
#         attentionV_list = []
#
#         for i in range(len(Qs)):
#             attentionQ = tf.nn.softmax(
#                 (tf.matmul(Qs[i], Ks[i], transpose_b=True) / tf.log(tf.cast(dv, tf.float32))),
#                 axis=1)  # bpc,bcp -> bpp  # softmax with latent variable
#
#             attentionV = tf.transpose(tf.matmul(Vs[i], attentionQ, transpose_a=True),
#                                       perm=[0, 2, 1])  # bcp , bpp -> bcp , transpose(bcp) -> bpc
#             print("attentionV+shape", attentionV)
#             attentionV_list.append(attentionV)
#         attentionVs = tf.concat(attentionV_list, axis=2)
#         # attention_output1 = tf.keras.layers.Dense(flatten_shape[1], activation="relu",
#         #                                           kernel_initializer=tf.keras.initializers.he_normal())(attentionVs)
#         # attention_output2 = tf.keras.layers.Dense(flatten_shape[1], activation="relu",
#         #                                           kernel_initializer=tf.keras.initializers.he_normal())(attention_output1)
#
#         print("attentionVs_shape:", attentionVs)
#         attention_output1 = tf.layers.dense(attentionVs, dmodel, activation="tanh",
#                                             kernel_initializer=tf.keras.initializers.he_normal(),
#                                             name="atten_dense1")
#
#         attention_output2 = tf.layers.dense(attention_output1, dmodel, activation="tanh",
#                                             kernel_initializer=tf.keras.initializers.he_normal(),
#                                             name="atten_dense2")
#
#         flatten_layer += attention_output2
#
#         input_layer = tf.reshape(flatten_layer, shape=input_shape, name="atten_output_reshape")
#
#         return input_layer
