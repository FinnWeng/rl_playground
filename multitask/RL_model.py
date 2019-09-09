import gym
import tensorflow as tf
import numpy as np
import octave_module
import VQVAE_ema_module


class RL_model:
    def __init__(self, input_shape, seq_size=4, baseline=0.0, batch_size=32, filter_num=32, TD_traj_leng=5,
                 discount_rate=0.99,
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

        self.filter_num = filter_num

        self.latent_base = 64
        self.latent_size = 256

        self.training_status = True

        self.kernel = tf.keras.initializers.glorot_normal()

        self.build_net()

    def build_net(self):
        with tf.name_scope("inputs"):
            input_state = tf.placeholder(shape=[208, 160, 3], dtype=tf.uint8)

            x_holder = tf.placeholder(tf.float32, self.input_shape,
                                      name="states")  # reduce batch and timestep when input
            # value_y_holder = tf.placeholder(tf.float32, [None, 1], name="value_y")  # reward, None => how long it play

            actions_y_holder = tf.placeholder(tf.float32, [None, 4],
                                              name="action_y")  # action it have taken,[batch_size, 4]
            R_plus_plus1_v_holder = tf.placeholder(tf.float32, [None, 1], name="R_plus_plus1_v")  # [batch_size,1]
            rt_holder = tf.placeholder(tf.float32, [None, 1], name="rt_holder")
            st_p1_holder = tf.placeholder(tf.float32, self.input_shape,
                                          name="p1_states")  # reduce batch and timestep when input

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

        # define predict next frame and reward op before loop
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        st_p1_predict = self.predict_next_frame_net(encoder_output)
        rt_predict = self.predict_next_reward_net(encoder_output)




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



        # # noisy_net
        #
        # noisy_net_ops = []
        #
        # target_net_only_variables = [v for v in tf.global_variables() if "actor" in v.name]
        #
        # for pre_noise_variable in target_net_only_variables:
        #     noisy_net_noise = tf.random.normal(shape=pre_noise_variable.shape)
        #     noisy_net_ops.append(tf.add(pre_noise_variable, noisy_net_noise))

        # define losses
        ###############################################

        actor_loss = tf.reduce_mean(advantage * neglogpac)  # policy_gradient_loss

        critic_loss = tf.losses.mean_squared_error(R_plus_plus1_v_holder,
                                                   V_value)

        reconst_loss = tf.reduce_mean(tf.squared_difference(st_p1_holder, st_p1_predict))

        rt_loss = tf.reduce_mean(tf.squared_difference(rt_holder, rt_predict))

        VQ_loss = tf.reduce_mean(self.top_VQ_loss + self.bottom_VQ_loss)


        total_loss = actor_loss - entropy * self.ent_coef + critic_loss * self.vf_coef + reconst_loss + rt_loss + VQ_loss
        # cross_entropy is minimize,  - entropy is minimize, critic loss is minimize

        optimizer = tf.train.AdamOptimizer(self.training_LR, name="adam")

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
        self.rt_holder = rt_holder
        self.st_p1_holder = st_p1_holder

        self.summary_figure = summary_figure
        self.total_training_op = total_training_op
        self.gray_img_output = gray_img_output
        self.V_value = V_value
        self.prediction_prob = prediction_prob
        self.advantage = advantage
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.total_loss = total_loss

    def encoder_net(self, enc_x_holder):
        oct_conv_first_layer = octave_module.oct_conv_first_layer
        oct_conv_block = octave_module.oct_conv_block
        oct_conv_final_layer = octave_module.oct_conv_final_layer
        x_holder_straight = tf.reshape(enc_x_holder, [-1, 208, 160, 1])
        normalized_x = (x_holder_straight) / 255
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("layer_1", reuse=tf.AUTO_REUSE):
                # level1
                l1_raw_output = tf.keras.layers.Conv2D(self.filter_num, kernel_size=3, strides=1, activation="tanh",
                                                       padding="SAME",
                                                       kernel_initializer=self.kernel)(normalized_x)

                l1_H_x, l1_L_x = oct_conv_first_layer(l1_raw_output, channel_num=16, alpha=0.8, kernel_size=3,
                                                      activation=tf.nn.tanh)
                l1_H_x, l1_L_x = oct_conv_block(l1_H_x, l1_L_x, channel_num=self.filter_num, alpha=0.8, kernel_size=3,
                                                activation=tf.nn.tanh)
                l1_H_x, l1_L_x = oct_conv_block(l1_H_x, l1_L_x, channel_num=self.filter_num, alpha=0.8, kernel_size=3,
                                                activation=tf.nn.tanh)

            with tf.variable_scope("layer_2", reuse=tf.AUTO_REUSE):
                # level2
                l2_H_x, l2_L_x = tf.keras.layers.MaxPool2D(pool_size=2, padding="SAME")(
                    l1_H_x), tf.keras.layers.MaxPool2D(
                    pool_size=2, padding="SAME")(l1_L_x)

                l2_H_x, l2_L_x = oct_conv_block(l2_H_x, l2_L_x, channel_num=self.filter_num * 2, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l2_H_x, l2_L_x = oct_conv_block(l2_H_x, l2_L_x, channel_num=self.filter_num * 2, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)

            with tf.variable_scope("layer_3", reuse=tf.AUTO_REUSE):
                # level3
                l3_H_x, l3_L_x = tf.keras.layers.MaxPool2D(pool_size=2, padding="SAME")(
                    l2_H_x), tf.keras.layers.MaxPool2D(
                    pool_size=2, padding="SAME")(l2_L_x)

                l3_H_x, l3_L_x = oct_conv_block(l3_H_x, l3_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l3_H_x, l3_L_x = oct_conv_block(l3_H_x, l3_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)

            with tf.variable_scope("layer_31", reuse=tf.AUTO_REUSE):
                # level31
                # l31_H_x, l31_L_x = tf.keras.layers.MaxPool2D(pool_size=2, padding="SAME")(l3_H_x), tf.keras.layers.MaxPool2D(
                #     pool_size=2, padding="SAME")(l3_L_x)

                l31_H_x, l31_L_x = oct_conv_block(l3_H_x, l3_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)
                l31_H_x, l31_L_x = oct_conv_block(l31_H_x, l31_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)

            with tf.variable_scope("layer_32", reuse=tf.AUTO_REUSE):
                # # level32
                #
                l32_H_x, l32_L_x = oct_conv_block(l31_H_x, l31_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)
                l32_H_x, l32_L_x = oct_conv_block(l32_H_x, l32_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)

            # print("l32_H_x, l32_L_x:",l32_H_x, l32_L_x)
            # (?, 36, 64, 38) ,(?, 18, 32, 10),

            with tf.variable_scope("layer_4", reuse=tf.AUTO_REUSE):
                # level4
                l4_H_x, l4_L_x = tf.keras.layers.MaxPool2D(pool_size=2, padding="SAME")(
                    l32_H_x), tf.keras.layers.MaxPool2D(
                    pool_size=2, padding="SAME")(l32_L_x)

                l4_H_x, l4_L_x = oct_conv_block(l4_H_x, l4_L_x, channel_num=self.filter_num * 4, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l4_H_x, l4_L_x = oct_conv_block(l4_H_x, l4_L_x, channel_num=self.filter_num * 4, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l4_raw_output = oct_conv_final_layer(l4_H_x, l4_L_x, channel_num=self.filter_num * 5, kernel_size=3,
                                                     activation=tf.nn.tanh)
                img_shape = l4_raw_output.shape  # (32, 18, 32, 80)

                # print("img_shape:", img_shape)

                l4_output = tf.keras.layers.Conv2D((self.latent_base), kernel_size=1, strides=1,
                                                   kernel_initializer=tf.keras.initializers.glorot_normal())(
                    l4_raw_output)

            # l4_output = tf.keras.layers.Dense((self.latent_base),
            #                                   kernel_initializer=tf.keras.initializers.glorot_normal())(l4_raw_output)

            # print("l4_output.shape:", l4_output.shape)

            with tf.variable_scope("top_VQVAE"):
                top_VQVAE_instance = VQVAE_ema_module.VQVAE(self.latent_base, self.latent_size, 0.25, "top_VQVAE")
                top_VQ_out_dict = top_VQVAE_instance.VQVAE_layer(l4_output)

            top_VQ_out = top_VQ_out_dict['quantized_embd_out']
            self.top_VQ_loss = top_VQ_out_dict["VQ_loss"]
            self.top_VQ_encodings = top_VQ_out_dict["encodings"]
            self.top_VQ_assign_moving_avg_op = top_VQ_out_dict['assign_moving_avg_op']

            # print("top_VQ_out:", top_VQ_out)

            # unflatten_ouput = VQ_out
            #
            # print("unflatten_ouput:", unflatten_ouput)

            channel_reconstruct = tf.keras.layers.Dense(img_shape[-1],
                                                        kernel_initializer=tf.keras.initializers.glorot_normal())(
                top_VQ_out)
            # print("channel_reconstruct:", channel_reconstruct)

            # level5
            with tf.variable_scope("layer_5", reuse=tf.AUTO_REUSE):
                l5_H_x, l5_L_x = oct_conv_first_layer(channel_reconstruct, channel_num=self.filter_num * 4, alpha=0.8,
                                                      kernel_size=3,
                                                      activation=tf.nn.tanh)

                # short cut
                ################
                l5_H_x, l5_L_x = self.short_cut_layer(l4_H_x, l5_H_x), self.short_cut_layer(l4_L_x, l5_L_x)

                ################

                l5_H_x, l5_L_x = oct_conv_block(l5_H_x, l5_L_x, channel_num=self.filter_num * 4, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l5_H_x, l5_L_x = oct_conv_block(l5_H_x, l5_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)

            with tf.variable_scope("layer_62", reuse=tf.AUTO_REUSE):
                # # level62
                l62_H_x, l62_L_x = [
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 3, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l5_H_x),
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 3, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l5_L_x)]

                l62_output = oct_conv_final_layer(l62_H_x, l62_L_x, channel_num=self.latent_base, kernel_size=3,
                                                  activation=tf.nn.tanh)

            # short cut
            ################
            print("top_VQ_out:", top_VQ_out.shape)
            print("l62_output:", l62_output.shape)
            resize_top_VQ_out = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(top_VQ_out)
            bottom_input = tf.concat([l62_output, resize_top_VQ_out], axis=3)

            with tf.variable_scope("bottom_VQVAE"):
                bottom_VQVAE_instance = VQVAE_ema_module.VQVAE(self.latent_base * 2, self.latent_size, 0.25,
                                                               "bottom_VQVAE")
                bottom_VQ_out_dict = bottom_VQVAE_instance.VQVAE_layer(bottom_input)
            bottom_VQ_out = bottom_VQ_out_dict['quantized_embd_out']
            self.bottom_VQ_loss = bottom_VQ_out_dict["VQ_loss"]
            self.bottom_VQ_encodings = bottom_VQ_out_dict["encodings"]
            self.bottom_VQ_assign_moving_avg_op = bottom_VQ_out_dict['assign_moving_avg_op']
            bottom_VQ_out = tf.concat([bottom_VQ_out, resize_top_VQ_out], axis=3)

            with tf.variable_scope("layer_61", reuse=tf.AUTO_REUSE):
                l61_H_x, l61_L_x = oct_conv_first_layer(bottom_VQ_out, channel_num=self.filter_num * 6, alpha=0.8,
                                                        kernel_size=3,
                                                        activation=tf.nn.tanh)

                # level61
                # short cut
                ################
                l61_H_x, l61_L_x = self.short_cut_layer(l31_H_x, l61_H_x), self.short_cut_layer(l31_L_x, l61_L_x)
                ################

                l61_H_x, l61_L_x = oct_conv_block(l61_H_x, l61_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)
                l61_H_x, l61_L_x = oct_conv_block(l61_H_x, l61_L_x, channel_num=self.filter_num * 3, alpha=0.8,
                                                  kernel_size=3,
                                                  activation=tf.nn.tanh)

            with tf.variable_scope("layer_7", reuse=tf.AUTO_REUSE):
                # level7
                l7_H_x, l7_L_x = [
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 2, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l61_H_x),
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 2, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l61_L_x)]

                # short cut
                ################
                l7_H_x, l7_L_x = self.short_cut_layer(l2_H_x, l7_H_x), self.short_cut_layer(l2_L_x, l7_L_x)
                ################

                # l7_H_x, l7_L_x = tf.keras.layers.SpatialDropout2D(rate=0.2)(l7_H_x, training=self.training_status), \
                #                  tf.keras.layers.SpatialDropout2D(rate=0.2)(l7_L_x, training=self.training_status)

                l7_H_x, l7_L_x = oct_conv_block(l7_H_x, l7_L_x, channel_num=self.filter_num * 2, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l7_H_x, l7_L_x = oct_conv_block(l7_H_x, l7_L_x, channel_num=self.filter_num * 1, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)

            with tf.variable_scope("layer_8", reuse=tf.AUTO_REUSE):
                # level8
                l8_H_x, l8_L_x = [
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 1, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l7_H_x),
                    tf.keras.layers.Conv2DTranspose(self.filter_num * 1, kernel_size=3, strides=2, padding="SAME",
                                                    kernel_initializer=self.kernel, activation=tf.nn.tanh)(
                        l7_L_x)]
                # short cut
                ################
                l8_H_x, l8_L_x = self.short_cut_layer(l1_H_x, l8_H_x), self.short_cut_layer(l1_L_x, l8_L_x)
                ################

                l8_H_x, l8_L_x = oct_conv_block(l8_H_x, l8_L_x, channel_num=self.filter_num * 1, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l8_H_x, l8_L_x = oct_conv_block(l8_H_x, l8_L_x, channel_num=self.filter_num * 1, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l8_output = oct_conv_final_layer(l8_H_x, l8_L_x, channel_num=self.filter_num, kernel_size=3,
                                                 activation=tf.nn.tanh)

        return l8_output

    def predict_next_frame_net(self, encoder_output):
        # self.x_holder
        # org_x_holder
        recon_out = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
                                           kernel_initializer=self.kernel)(encoder_output)
        recon_out = tf.keras.layers.Conv2D(3, kernel_size=3, strides=1, padding="SAME",
                                           kernel_initializer=self.kernel, activation="sigmoid")(recon_out)
        return recon_out

    def predict_next_reward_net(self, encoder_output):
        next_reward_out = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
                                                 kernel_initializer=self.kernel)(encoder_output)

        next_reward_out_shape = next_reward_out.shape.as_list()

        next_reward_out = tf.keras.layers.Conv2D(1, kernel_size=[next_reward_out_shape[1], next_reward_out_shape[2]],
                                                 strides=1,
                                                 kernel_initializer=self.kernel)(next_reward_out)

        next_reward_out = tf.reshape(next_reward_out, [-1, 1])

        return next_reward_out

    def actor_net(self, encoder_output):  # num_step = seq_size

        logits = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
                                        kernel_initializer=self.kernel)(encoder_output)

        logits_shape = logits.shape.as_list()

        logits = tf.keras.layers.Conv2D(4, kernel_size=[logits_shape[1], logits_shape[2]],
                                        strides=1,
                                        kernel_initializer=self.kernel, activation="sigmoid")(logits)

        logits = tf.reshape(logits, [-1, 4])

        print("logits:", logits)

        return logits

    def critic_net(self, encoder_output):  # num_step = seq_size

        V_value = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME",
                                         kernel_initializer=self.kernel)(encoder_output)

        logits_shape = V_value.shape.as_list()

        V_value = tf.keras.layers.Conv2D(1, kernel_size=[logits_shape[1], logits_shape[2]],
                                         strides=1,
                                         kernel_initializer=self.kernel, activation="sigmoid")(V_value)

        V_value = tf.reshape(V_value, [-1, 1])

        print("V_value:", V_value)

        return V_value

    def short_cut_layer(self, enc_layer, dec_layer):
        short_cut1_H_x = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="SAME",
                                                kernel_initializer=tf.keras.initializers.glorot_normal())(enc_layer)

        short_cut1_H_x = tf.keras.layers.Conv2D(dec_layer.shape.as_list()[-1], kernel_size=3, strides=1, padding="SAME",
                                                kernel_initializer=tf.keras.initializers.glorot_normal())(
            short_cut1_H_x)

        expand_dec_layer = tf.concat([short_cut1_H_x, dec_layer], axis=3)

        return expand_dec_layer
