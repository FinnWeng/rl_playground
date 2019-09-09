import gym
import tensorflow as tf
import numpy as np


class RL_model:
    def __init__(self, input_shape, seq_size=4, baseline=0.0, batch_size=32, TD_traj_leng=5, discount_rate=0.99,
                 training_LR=1e-6, ent_coef=0.0005, vf_coef=2):
        self.seq_size = seq_size  # MUST greater than 2. How many past states should we look for decide an action

        self.baseline = baseline

        self.batch_size = batch_size

        self.TD_traj_leng = TD_traj_leng  # actually past design is TD 1, now I assign it to estimate more step

        self.discount_rate = discount_rate

        self.training_LR = training_LR

        self.ent_coef = ent_coef

        self.vf_coef = vf_coef

        self.input_shape = input_shape

        self.build_net()

    def build_net(self):

        with tf.name_scope("inputs"):
            input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)

            x_holder = tf.placeholder(tf.float32, self.input_shape,
                                      name="states")  # reduce batch and timestep when input
            # value_y_holder = tf.placeholder(tf.float32, [None, 1], name="value_y")  # reward, None => how long it play

            actions_y_holder = tf.placeholder(tf.float32, [None, 4],
                                              name="action_y")  # action it have taken,[batch_size, 4]
            R_plus_plus1_v_holder = tf.placeholder(tf.float32, [None, 1], name="R_plus_plus1_v")  # [batch_size,1]
            episode_reward_holder = tf.placeholder(tf.float32, [], name="episode_reward")

        # define action op before loop
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        encoder_output = self.encoder_net(x_holder)  # (-1,210,160,3) feed an array of images, -1 = batch_size
        logits = self.actor_net(encoder_output)  # (batch_size,4 actions)
        prediction_prob = tf.add(tf.nn.softmax(logits), 1e-8)  # (batch_size,4 prob)

        print("prediction_prob:", prediction_prob.shape)

        # define critic op before loop
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        V_value = self.critic_net(encoder_output)  # batch_size, 1 value

        # RGB to Gray
        gray_img_output = tf.image.rgb_to_grayscale(input_state)

        # define actor training op before loop
        ########################################################

        # st

        entropy = -tf.reduce_sum(prediction_prob * tf.log(prediction_prob),
                                 name="entropy")  # batchsize, 1

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=tf.argmax(actions_y_holder, axis=1))

        advantage = R_plus_plus1_v_holder - V_value

        # policy_gradient = tf.multiply(advantage,
        #                               log_prob) + 0.001 * entropy  # (batch_size, 1 Q value))*(batch_size, 1 max prob) + (batch_size, 1 entropy)

        actor_loss = tf.reduce_mean(advantage * neglogpac)  # policy_gradient_loss

        # noisy_net

        noisy_net_ops = []

        target_net_only_variables = [v for v in tf.global_variables() if "actor" in v.name]

        for pre_noise_variable in target_net_only_variables:
            noisy_net_noise = tf.random.normal(shape=pre_noise_variable.shape)
            noisy_net_ops.append(tf.add(pre_noise_variable, noisy_net_noise))

        # define critic training op before loop
        ###############################################

        critic_loss = tf.losses.mean_squared_error(R_plus_plus1_v_holder,
                                                   V_value)

        total_loss = actor_loss - entropy * self.ent_coef + critic_loss * self.vf_coef
        # cross_entropy is minimize,  - entropy is minimize, critic loss is minimize

        optimizer = tf.train.AdamOptimizer(self.training_LR, name="adam")
        #
        # variables = [v for v in tf.global_variables()]
        #
        # gradients, variables_beside_gd = zip(
        #     *optimizer.compute_gradients(actor_loss, var_list=variables))
        #
        # clipped_gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in gradients]
        #
        # total_training_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

        total_training_op = optimizer.minimize(total_loss)

        max_advantage = tf.reduce_max(advantage)
        min_advantage = tf.reduce_min(advantage)
        avg_advantage = tf.reduce_mean(advantage)
        max_V_value = tf.reduce_max(V_value)
        min_V_value = tf.reduce_min(V_value)
        avg_V_value = tf.reduce_mean(V_value)

        summary_figure = [tf.summary.scalar("actor_loss", actor_loss),
                          tf.summary.scalar("critic_loss", critic_loss),
                          tf.summary.scalar("entropy", entropy),
                          tf.summary.histogram("actor_loss_hist", actor_loss),
                          tf.summary.histogram("critic_loss_hist", critic_loss),
                          tf.summary.scalar("total_loss", total_loss),
                          tf.summary.scalar("max_advantage", max_advantage),
                          tf.summary.scalar("min_advantage", min_advantage),
                          tf.summary.scalar("avg_advantage", avg_advantage),
                          tf.summary.scalar("max_V_value", max_V_value),
                          tf.summary.scalar("min_V_value", min_V_value),
                          tf.summary.scalar("avg_V_value", avg_V_value)
                          ]

        # outputs
        self.input_state = input_state
        self.x_holder = x_holder
        self.actions_y_holder = actions_y_holder
        self.R_plus_plus1_v_holder = R_plus_plus1_v_holder
        self.episode_reward_holder = episode_reward_holder
        self.summary_figure = summary_figure
        self.total_training_op = total_training_op
        self.gray_img_output = gray_img_output
        self.V_value = V_value
        self.prediction_prob = prediction_prob
        self.advantage = advantage
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.total_loss =total_loss

    def encoder_net(self, enc_x_holder):
        layer_count = 0
        gain = np.sqrt(2)
        x_holder_straight = tf.reshape(enc_x_holder, [-1, 210, 160, 1])

        normalized_x = (x_holder_straight) / 255

        def conv(inputs, nf, ks, strides, gain=1.0):
            return tf.layers.conv2d(inputs=inputs, filters=nf, kernel_size=ks,
                                    strides=(strides, strides), activation=tf.nn.relu,
                                    kernel_initializer=tf.orthogonal_initializer(gain=gain),
                                    name="enc_net_layer%s" % (layer_count),
                                    reuse=tf.AUTO_REUSE)

        h1 = conv(normalized_x, 32, 8, 4, gain)
        layer_count += 1
        h2 = conv(h1, 64, 4, 2, gain)
        layer_count += 1
        h3 = conv(h2, 64, 3, 1, gain)
        layer_count += 1
        # encoder_output_flat = tf.layers.flatten(h3)
        encoder_output_flat = h3

        return encoder_output_flat

    # def actor_net(self, encoder_output_flat):  # num_step = seq_size
    #
    #     gain = 1.0
    #
    #     h_nn3 = tf.layers.dense(inputs=encoder_output_flat, units=512, activation=tf.nn.relu,
    #                             kernel_initializer=tf.orthogonal_initializer(gain), name="actor_net_dense3",
    #                             reuse=tf.AUTO_REUSE)
    #
    #     # nn layer 4
    #     logits = tf.layers.dense(h_nn3, 4, use_bias=True,
    #                              name="actor_net_dense4",
    #                              reuse=tf.AUTO_REUSE)
    #
    #     print("logits:", logits)
    #
    #     return logits

    def actor_net(self, encoder_output):  # num_step = seq_size

        logits = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
                                        kernel_initializer=tf.keras.initializers.glorot_normal())(encoder_output)

        logits_shape = logits.shape.as_list()

        logits = tf.keras.layers.Conv2D(4, kernel_size=[logits_shape[1], logits_shape[2]],
                                        strides=1,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(), activation="sigmoid")(logits)

        logits = tf.reshape(logits, [-1, 4])

        print("logits:", logits)

        return logits
    #
    # def critic_net(self, encoder_output_flat):  # num_step = seq_size
    #
    #     gain = 1.0
    #
    #     h_nn3 = tf.layers.dense(inputs=encoder_output_flat, units=512, activation=tf.nn.relu,
    #                             kernel_initializer=tf.orthogonal_initializer(gain), name="critic_net_dense3",
    #                             reuse=tf.AUTO_REUSE)
    #
    #     # nn layer 4
    #     V_value = tf.layers.dense(h_nn3, 1,
    #                               kernel_initializer=tf.orthogonal_initializer(gain),
    #                               name="critic_net_dense4",
    #                               reuse=tf.AUTO_REUSE)
    #
    #     print("V_value:", V_value)
    #
    #     return V_value


    def critic_net(self, encoder_output):  # num_step = seq_size

        V_value = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
                                         kernel_initializer=tf.keras.initializers.glorot_normal())(encoder_output)

        logits_shape = V_value.shape.as_list()
        print("logits_shape:",logits_shape)
        kernel_size = [logits_shape[1], logits_shape[2]]
        print("kernel_size:",kernel_size)

        V_value = tf.keras.layers.Conv2D(1, kernel_size=kernel_size,
                                         strides=1,
                                         kernel_initializer=tf.keras.initializers.glorot_normal(), activation="sigmoid")(V_value)

        print("V_value shape:",V_value.shape.as_list())

        V_value = tf.reshape(V_value, [-1, 1])

        print("V_value:", V_value)

        return V_value
