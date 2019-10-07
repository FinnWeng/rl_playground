
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
        
        # global_step = tf.Variable(0, trainable=False)
        # self.training_LR = tf.train.piecewise_constant_decay(global_step,training_LR[0],training_LR[1])

        self.training_LR = training_LR

        self.ent_coef = ent_coef

        self.vf_coef = vf_coef

        self.input_shape = input_shape

        self.filter_num = filter_num

        self.latent_base = 64
        self.latent_size = 256

        self.training_status = True

        self.beta = 0.2
        self.rl_coef = 0.5

        self.kernel = tf.keras.initializers.glorot_normal()

        self.build_net()

    def build_net(self):
        with tf.name_scope("inputs"):
            input_state = tf.placeholder(shape=[208, 160, 3], dtype=tf.uint8)

            x_holder = tf.placeholder(tf.float32, self.input_shape,
                                      name="states")  # reduce batch and timestep when input
            p1_st_holder = tf.placeholder(tf.float32, self.input_shape,
                                      name="p1_states")  # reduce batch and timestep when input
            
            # value_y_holder = tf.placeholder(tf.float32, [None, 1], name="value_y")  # reward, None => how long it play

            actions_y_holder = tf.placeholder(tf.float32, [None, 4],
                                              name="action_y")  # action it have taken,[batch_size, 4]
            actions_prob_holder = tf.placeholder(tf.float32, [None, 4],
                                              name="action_prob")  # action it have taken,[batch_size, 4]
            R_plus_plus1_v_holder = tf.placeholder(tf.float32, [None, 1], name="R_plus_plus1_v")  # [batch_size,1]
            rt_holder = tf.placeholder(tf.float32, [None, 1], name="rt_holder")


            episode_reward_holder = tf.placeholder(tf.float32, [], name="episode_reward")

        # define action op before loop
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        encoder_output = self.encoder_net(x_holder)  # (-1,210,160,3) feed an array of images, -1 = batch_size
        p1_st_encoder_output = self.encoder_net(p1_st_holder)  # (-1,210,160,3) feed an array of images, -1 = batch_size

        logits = self.actor_net(encoder_output)  # (batch_size,4 actions)
        prediction_prob = tf.add(tf.nn.softmax(logits,axis=1), 1e-8)  # (batch_size,4 prob)

        print("prediction_prob:", prediction_prob.shape)

        # define critic op before loop
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        V_value = self.critic_net(encoder_output)  # batch_size, 1 value

        # # define curious op
        # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print("encoder_st:",encoder_output)
        print("p1_st_encoder_output:",p1_st_encoder_output)

        inverse_loss = tf.reduce_mean(self.inverse_net(encoder_output,p1_st_encoder_output,actions_prob_holder))
        print("inverse_loss:",inverse_loss)
        forward_loss = tf.reduce_mean(self.forward_net(encoder_output,p1_st_encoder_output,actions_y_holder))


        # RGB to Gray
        gray_img_output = tf.image.rgb_to_grayscale(input_state)

        # define actor training op before loop
        ########################################################

        # st

        entropy = -tf.reduce_sum(prediction_prob * tf.log(prediction_prob),
                                 name="entropy")  # batchsize, 1

        
        # actor_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(actions_y_holder, axis=1))
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=tf.argmax(actions_y_holder, axis=1))

        
        # softmax_value = -tf.reduce_max(tf.log(tf.nn.softmax(logits,axis=-1,name="softmax_test")*actions_y_holder),axis=-1)
        # softmax_value = tf.reduce_mean(softmax_value)
        # neglogpac = -tf.reduce_max(tf.log(tf.nn.softmax(logits,axis=-1,name="softmax_test")*actions_y_holder),axis=-1) # exactly the same with cross-entropy
        neglogpac = -tf.log(tf.reduce_max(tf.nn.softmax(logits,axis=-1,name="softmax_test")*actions_y_holder,axis=-1))

        advantage = R_plus_plus1_v_holder - V_value
        
        print("advantage:",advantage)
        print("neglogpac:",neglogpac)
        print("R_plus_plus1_v_holder:",R_plus_plus1_v_holder)
        print("value:", V_value)



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

        # actor_loss = tf.reduce_mean(actor_cross_entropy)  # policy_gradient_loss
        
        policy_gradient_loss = tf.reduce_mean(advantage * neglogpac)  # policy_gradient_loss
        print("policy_gradient_loss:",policy_gradient_loss)

        critic_loss = tf.losses.mean_squared_error(R_plus_plus1_v_holder,
                                                   V_value)


        # rt_loss = tf.reduce_mean(tf.squared_difference(rt_holder, rt_predict))

        # VQ_loss = tf.reduce_mean(self.top_VQ_loss + self.bottom_VQ_loss)


        total_loss = self.rl_coef*(policy_gradient_loss - entropy * self.ent_coef + critic_loss * self.vf_coef) + (1-self.beta)*inverse_loss + (self.beta)*forward_loss
        # cross_entropy is minimize,  - entropy is minimize, critic loss is minimize

        optimizer = tf.train.AdamOptimizer(self.training_LR, name="adam")

        total_training_op = optimizer.minimize(total_loss)

        max_advantage = tf.reduce_max(advantage)
        min_advantage = tf.reduce_min(advantage)
        avg_advantage = tf.reduce_mean(advantage)
        max_V_value = tf.reduce_max(V_value)
        min_V_value = tf.reduce_min(V_value)
        avg_V_value = tf.reduce_mean(V_value)

        summary_figure = [
                          tf.summary.scalar("critic_loss", critic_loss),
                          tf.summary.scalar("inverse_loss", inverse_loss),
                          tf.summary.scalar("forward_loss", forward_loss),
                          tf.summary.scalar("entropy", entropy),
                          tf.summary.scalar("polycy_gradient_loss",policy_gradient_loss),
                          tf.summary.scalar("neglogpac_mean",tf.reduce_mean(neglogpac)),
                    
                        #   tf.summary.scalar("softmax_value_for_test:",softmax_value),
                        #   tf.summary.scalar("softmax_for_verification",tf.reduce_mean(softmax_for_verification)),
                        #   tf.summary.scalar("crossentropy_for_verification",tf.reduce_mean(crossentropy_for_verification)),

                          tf.summary.histogram("critic_loss_hist", critic_loss),
                          tf.summary.histogram("inverse_loss_hist", inverse_loss),
                          tf.summary.histogram("forward_loss_hist", forward_loss),
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
        self.p1_st_holder = p1_st_holder
        self.actions_prob_holder = actions_prob_holder

        self.summary_figure = summary_figure
        self.total_training_op = total_training_op
        self.gray_img_output = gray_img_output
        self.V_value = V_value
        self.prediction_prob = prediction_prob
        self.advantage = advantage
        self.policy_gradient_loss = policy_gradient_loss
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

                normalized_x
                Coord_x = CoordConv2D(with_r = False)(normalized_x)
                l1_raw_output = tf.keras.layers.Conv2D(self.filter_num, kernel_size=3, strides=1, activation="tanh",
                                                       padding="SAME",
                                                       kernel_initializer=self.kernel)(Coord_x)

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

            with tf.variable_scope("layer_4", reuse=tf.AUTO_REUSE):
                # level4
                l4_H_x, l4_L_x = tf.keras.layers.MaxPool2D(pool_size=2, padding="SAME")(
                    l3_H_x), tf.keras.layers.MaxPool2D(
                    pool_size=2, padding="SAME")(l3_L_x)

                l4_H_x, l4_L_x = oct_conv_block(l4_H_x, l4_L_x, channel_num=self.filter_num * 4, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l4_H_x, l4_L_x = oct_conv_block(l4_H_x, l4_L_x, channel_num=self.filter_num * 4, alpha=0.8,
                                                kernel_size=3,
                                                activation=tf.nn.tanh)
                l4_raw_output = oct_conv_final_layer(l4_H_x, l4_L_x, channel_num=self.filter_num * 5, kernel_size=3,
                                                     activation=tf.nn.tanh)
                img_shape = l4_raw_output.shape  # (32, 18, 32, 80)

                print("img_shape:", img_shape)

                l4_output = tf.keras.layers.Conv2D((self.filter_num * 5), kernel_size=[img_shape.as_list()[1], img_shape.as_list()[2]], strides=1,kernel_initializer=tf.keras.initializers.glorot_normal())(l4_raw_output)
                l4_output = tf.reshape(l4_output, [-1,self.filter_num * 5])


        return l4_output



    def actor_net(self, encoder_output):  # num_step = seq_size

        act_l1 = tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                     activation=tf.nn.tanh)(encoder_output)
        logits = tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.glorot_normal())(act_l1)
        print("logits:", logits)

        return logits

    def critic_net(self, encoder_output):  # num_step = seq_size
        critic_l1 = tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                     activation=tf.nn.tanh)(encoder_output)
        V_value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_normal())(critic_l1)

        print("V_value:", V_value)

        return V_value

    def inverse_net(self,encoder_st,encoder_p1_st,actions_prob_holder):
        '''
        encoder_st: batch_size, self.filter_num*5
        encoder_p1_st: batch_size, self.filter_num*5
        action_y_holder: batch_size, 4
        '''
        print("encoder_st:",encoder_st)
        print("encoder_p1_st:",encoder_p1_st)

        inverse_l1 = tf.concat([encoder_st,encoder_p1_st],axis=-1)
        inverse_l2 = tf.keras.layers.Dense(self.filter_num*3, kernel_initializer=tf.keras.initializers.glorot_normal(),activation=tf.nn.tanh)(inverse_l1)
        inverse_l3 = tf.keras.layers.Dense(self.filter_num*3, kernel_initializer=tf.keras.initializers.glorot_normal())(inverse_l2)
        inverse_logits =  tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.glorot_normal())(inverse_l3)
        inverse_loss = tf.nn.softmax_cross_entropy_with_logits_v2(actions_prob_holder,inverse_logits,axis=-1,name="inverse_loss")
        return inverse_loss
    
    def forward_net(self,encoder_st,encoder_p1_st,action_y_holder):
        '''
        encoder_st: batch_size, self.filter_num*5
        action_y_holder: batch_size, 4
        '''
        forward_l1 = tf.concat([encoder_st,action_y_holder],axis=-1)
        forward_l2 = tf.keras.layers.Dense(self.filter_num*3, kernel_initializer=tf.keras.initializers.glorot_normal(),activation=tf.nn.tanh)(forward_l1)
        forward_l3 = tf.keras.layers.Dense(self.filter_num*5, kernel_initializer=tf.keras.initializers.glorot_normal())(forward_l2)
        forward_p1_st = tf.keras.layers.Dense(self.filter_num*5, kernel_initializer=tf.keras.initializers.glorot_normal())(forward_l3)

        forward_loss=tf.reduce_mean(tf.squared_difference(encoder_p1_st, forward_p1_st), axis=-1)
        return forward_loss

class CoordConv2D:
    def __init__(self, with_r = False):
        self.with_r = with_r
    def __call__(self,input):
        self.x_dim = input.shape.as_list()[2]
        self.y_dim = input.shape.as_list()[1]
        batch_size_tensor = tf.shape(input)[0]
        xy_vector = tf.ones([self.y_dim,1])
        yx_vector = tf.ones([1,self.x_dim])
        x_range = tf.reshape(tf.range(1,self.x_dim+1,1,dtype=tf.float32),[1,self.x_dim])
        y_range = tf.reshape(tf.range(1,self.y_dim+1,1,dtype=tf.float32),[self.y_dim,1])
        x_normal_range = tf.multiply(x_range,1/self.x_dim)
        y_normal_range = tf.multiply(y_range,1/self.y_dim)
        x_mat = tf.matmul(xy_vector,x_normal_range)
        y_mat = tf.matmul(y_normal_range,yx_vector)

        x_mat = tf.reshape(x_mat,[1,self.y_dim,self.x_dim,1])
        y_mat = tf.reshape(y_mat,[1,self.y_dim,self.x_dim,1])
        x_mats = tf.tile(x_mat,[batch_size_tensor,1,1,1])
        y_mats = tf.tile(y_mat,[batch_size_tensor,1,1,1])


        
        if self.with_r == True:
            # # orgin
            # r = ((x_mats-0.5)**2 + (y_mats-0.5)**2)
            # r = tf.sqrt(r)

            # I test 
            r = (tf.sqrt((x_mats-0.5)**2) + tf.sqrt((y_mats-0.5)**2))

            input = tf.concat([input,x_mats,y_mats,r],axis=-1)
            return input
        else:
            input = tf.concat([input,x_mats,y_mats],axis=-1)
            return input


    

if __name__ == "__main__":

    logs_path = "./A2C_multitask/tf_log"

    critic_loss_history = []
    neg_entropy_history = []

    replay_buffer = []

    seq_size = 4  # MUST greater than 2. How many past states should we look for decide an action

    baseline = 0.0

    batch_size = 32

    filter_num = 16

    TD_traj_leng = 5  # actually past design is TD 1, now I assign it to estimate more step

    discount_rate = 0.99

    update_times = 0

    training_LR = 1e-5

    ent_coef = 0.0005

    vf_coef = 2

    input_shape = [None, 208, 160, 1]
    model = RL_model(input_shape, seq_size, baseline, batch_size, filter_num, TD_traj_leng, discount_rate,
                              training_LR, ent_coef, vf_coef)