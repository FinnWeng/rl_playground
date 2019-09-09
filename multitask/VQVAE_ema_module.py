import tensorflow as tf

# EMA

class VQVAE:
    def __init__(self, embedding_dim, _num_embeddings, commit_loss_coef, scope):
        self.commit_loss_coef = commit_loss_coef
        self._num_embeddings = _num_embeddings  # the number of embed vectors
        self._embedding_dim = embedding_dim  # which means how many discrete symbol for a digit(base n numerical)
        self.scope = scope
        self.gamma = 0.99

        #  embedding_dim: length of latent variable, in this implementation it is channel number of input tensor(how many bits)
        """
        So, it is like this:
        the num_embed is like we recongnize all experience we enconter into several kind of situations. i.e. all 2000 pics to 10 kind of classes.
        the embedding_dim, is the memory we like to spend to memorize the classed. The more we spend, the more details model can remember.


        """
        # print("_num_embeddings:", self._num_embeddings)
        # print("self._embedding_dim:", self._embedding_dim)

    def variable_def(self):
        initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope(self.scope, 'VQVAE', reuse=tf.AUTO_REUSE):
            self._w = tf.get_variable('embedding', [self._embedding_dim, self._num_embeddings],
                                      initializer=initializer, trainable=True)

            self.embedding_total_count = tf.get_variable("embedding_total_count", [1, self._num_embeddings],
                                                         initializer=tf.zeros_initializer(),
                                                         dtype=tf.int32)

    def loop_assign_moving_avg(self, encodings, flat_inputs):
        # print("encodings:", encodings) # (b,H*W,self._num_embeddings)
        embedding_count = tf.reshape(tf.reduce_sum(encodings, axis=[0, 1]), [1, -1])  # [1,1024]

        # print("self.embedding_total_count:", self.embedding_total_count)

        embedding_total_count_temp = tf.math.floordiv(tf.cast(self.embedding_total_count, tf.float32),
                                                      (1 / self.gamma)) + tf.math.floordiv(embedding_count,
                                                                                           1 / (1 - self.gamma))

        embedding_total_count_temp = embedding_total_count_temp+1


        # print("embedding_total_count_temp:", embedding_total_count_temp)
        # floordiv to replace multiply + floor

        self.embedding_total_count = tf.assign(self.embedding_total_count, tf.cast(embedding_total_count_temp,tf.int32))  # [1,num_embedding]
        # print("self.embedding_total_count:", self.embedding_total_count)

        # self.expand_encodings = tf.expand_dims(encodings, -2)
        # expand_flat_inputs = tf.expand_dims(flat_inputs, -1)
        self.expand_encodings = encodings
        expand_flat_inputs = tf.transpose(flat_inputs,[0,2,1])
        # print("expand_encodings:",self.expand_encodings)
        # print("expand_flat_inputs:",expand_flat_inputs)

        input_contrib_per_embedding_value = tf.matmul(expand_flat_inputs,self.expand_encodings) #(?, 128, 1024)


        input_contrib_per_embedding_value = tf.reduce_sum(input_contrib_per_embedding_value, axis=[0])
        # input_contrib_per_embedding_value = tf.transpose(input_contrib_per_embedding_value,[1,0])
        input_contrib_per_embedding_value = input_contrib_per_embedding_value / tf.cast(self.embedding_total_count,
                                                                                        tf.float32)


        # print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value)

        w_temp = (self._w) * (self.gamma) + (input_contrib_per_embedding_value) * (1 - self.gamma)

        self._w = tf.assign(self._w, w_temp)

        return [self._w,self.embedding_total_count]

    def quantize(self, encoding_indices):
        w = tf.transpose(self._w, [1, 0])

        # print("w:", w)
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

    def VQVAE_layer(self, inputs):
        # Assert last dimension is same as self._embedding_dim

        input_shape = tf.shape(inputs)
        with tf.control_dependencies([
            tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                      [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2], self._embedding_dim])


        self.variable_def()  # set all variable

        # the _w is already qunatized: for each row, each idx(latent variable digit) have its own value to pass, value pf _w is quantized embd ouput

        def dist_fn(tensor_apart):
            dist = (tf.reduce_sum(tensor_apart ** 2, 1, keepdims=True)
                    - 2 * tf.matmul(tensor_apart, self._w)
                    + tf.reduce_sum(self._w ** 2, 0, keepdims=True))  # different shape: tf.add broadcast
            return dist

        distances = tf.map_fn(dist_fn, flat_inputs)

        # print("distances:", distances)
        # distance.shape = [b,H*W,num_embeddings]
        encoding_indices = tf.argmin(distances,
                                     2)  # (batchsize,(1=>index)), find the _w which ressemble the inpus the most
        # print("encoding_indices:", encoding_indices)# (b,h*w)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)

        self.loop_assign_moving_avg(encodings, flat_inputs)

        # self._w = self._w * (self.gamma) + (1 - self.gamma) * ()

        # encodings = tf.reshape(encodings, [-1, input_shape[1], input_shape[2], self._num_embeddings])
        # print("encodings(after):", encodings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        # print("encoding_indices(after):", encoding_indices)

        quantized_embd_out = self.quantize(
            encoding_indices)  # Actually, this quantized method find the value from corespond econding_idx from w
        # print("quantized_embd_out:", quantized_embd_out)
        # print("inputs:", inputs)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized_embd_out) - inputs) ** 2)  # embedding loss
        # q_latent_loss = tf.reduce_mean((tf.stop_gradient(inputs) - quantized_embd_out) ** 2)

        VQ_loss = self.commit_loss_coef*e_latent_loss

        quantized_embd_out = inputs + tf.stop_gradient(
            quantized_embd_out - inputs)  # in order to pass value to decoder???

        # print("quantized_embd_out(after):", quantized_embd_out)

        avg_probs = tf.reduce_mean(encodings, 0)

        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

        # encodings_holder = tf.placeholder(tf.float32,
        #                                   [None, input_shape[1] * input_shape[2], self._num_embeddings])
        # flat_inputs_holder = tf.placeholder(tf.float32,
        #                                     [None, input_shape[1] * input_shape[2], self._embedding_dim])

        # assign_moving_avg_op = self.loop_assign_moving_avg(encodings_holder, flat_inputs_holder)
        assign_moving_avg_op = self.loop_assign_moving_avg(encodings, flat_inputs)

        return {'quantized_embd_out': quantized_embd_out,
                'VQ_loss': VQ_loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_indices,
                'assign_moving_avg_op': assign_moving_avg_op}

        # return {'quantized_embd_out': quantized_embd_out,
        #         'VQ_loss': VQ_loss,
        #         'perplexity': perplexity,
        #         'encodings': encodings,
        #         'encoding_indices': encoding_indices
        #         'encodings_holder': encodings_holder,
        #         'flat_inputs_holder': flat_inputs_holder,
        #         'assign_moving_avg_op': assign_moving_avg_op}

    def idx_inference(self, outer_encoding_indices):
        outer_encodings = tf.one_hot(outer_encoding_indices, self._num_embeddings)

        return outer_encodings
