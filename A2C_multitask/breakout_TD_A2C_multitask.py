import gym
import tensorflow as tf
import numpy as np
import gc
# from os import getpid
import os
import time
import random
import collections
import pprint

import RL_model

if __name__ == "__main__":

    logs_path = "./A2C_multitask/tf_log/"

    critic_loss_history = []
    actor_loss_history = []
    neg_entropy_history = []

    replay_buffer = []

    # use which GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    env = gym.make('Breakout-v0')

    seq_size = 4  # MUST greater than 2. How many past states should we look for decide an action

    baseline = 0.0

    batch_size = 4

    filter_num = 2

    TD_traj_leng = 5  # actually past design is TD 1, now I assign it to estimate more step

    discount_rate = 0.99

    update_times = 0

    training_LR = 1e-6

    ent_coef = 0.0005

    vf_coef = 2

    input_shape = [None, 208, 160, 1]

    model = RL_model.RL_model(input_shape, seq_size, baseline, batch_size, filter_num, TD_traj_leng, discount_rate,
                              training_LR, ent_coef, vf_coef)

    init = tf.global_variables_initializer()

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(list(tf.global_variables()))

    learning_figures = model.summary_figure

    merged_summary_op = tf.summary.merge(learning_figures)
    episode_reward_summary_op = tf.summary.merge([tf.summary.scalar("episode_reward", model.episode_reward_holder)])

    with tf.Session() as sess:

        # train in firsttime
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # # # keep training
        # saver = tf.train.Saver(max_to_keep=1)
        # model_file = tf.train.latest_checkpoint('./A2C_multitask/model/')
        # saver.restore(sess, model_file)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        noise_target_prob = 50

        for i_episode in range(10000000):

            # if i_episode > 0 and i_episode % 20 == 0:
            #     noise_target_prob = 500.
            #     print("do all episode random sample!!")

            # train_sample_buffer = []

            episode_reward = 0.
            inner_train_sample_buffer = []
            train_buffer = []
            TD_N_reward = []
            training_policy_target = []
            training_action = []
            training_st = []
            training_st_p1 = []
            training_rt = []

            # start a round
            observation = env.reset()
            print("observation:", observation.shape)
            observation = observation[1:209, :, :]
            print("observation:", observation.shape)
            observation = sess.run([model.gray_img_output], feed_dict={model.input_state: observation})[0]
            print("observation:", observation.shape)

            # step_size = 100000
            step_size = 10000

            # sess.run(noisy_net_ops)

            for step in range(step_size):
                training_step = step - ((TD_traj_leng - 1 + 1))
                # seq_size + TD_traj_leng - 1, -1 for TD cummulating, +1 for states+1

                env.render()

                # # temp include st, at, rt, st+1
                temp = {"st": observation}  # temp append st

                action_prob = sess.run(model.prediction_prob,
                                       feed_dict={
                                           model.x_holder: observation.reshape([1, 208, 160, 1])})  # generate action
                action_prob += 1e-8

                print("action_prob[0]", action_prob[0])

                # take action
                action_one_hot = np.zeros([4])
                # action = np.argmax(action_prob, axis=1)
                # print("action", action)
                action = np.random.choice(np.arange(len(action_prob[0])), p=action_prob[0])

                # # # epsilon noise (type2)
                # if noise_target_prob > 50:
                #     noise_target_prob -= 0.5
                #     print("random rate decreasing...")
                #
                # noise_threshold = 1000 - noise_target_prob
                # if np.random.randint(0, 1000, dtype=int) > noise_threshold:
                #     print("do random sample!!!!")
                #     action = [env.action_space.sample()]

                print("action", action)

                action_one_hot[action] = 1  # turn predict prob to one hot
                observation, reward, done, info = env.step(action)  # get result of action
                observation = observation[1:209, :, :]
                observation = sess.run([model.gray_img_output], feed_dict={model.input_state: observation})[0]
                p1_observation = observation

                episode_reward += reward

                # after action net decide how to take action and get reward, append at, rt

                temp["rt"] = reward
                temp["at"] = action_one_hot
                temp["done"] = done
                temp["p1_st"] = p1_observation

                # append temp to train sample buffer

                inner_train_sample_buffer.append(temp)

                if training_step >= 0:

                    # to compute one TD loss

                    training_reward = 0.0
                    policy_target = 0.0

                    if not inner_train_sample_buffer[-1]["done"]:
                        policy_target = sess.run(model.V_value, feed_dict={
                            model.x_holder: inner_train_sample_buffer[-1]["p1_st"].reshape(
                                [-1, 208, 160, 1])})[0][0]  # plus1 state value

                    critic_targets = []
                    actions = []

                    for transition in inner_train_sample_buffer[:-(TD_traj_leng + 1):-1]:
                        training_reward = transition["rt"] + discount_rate * training_reward
                        policy_target = transition["rt"] + discount_rate * policy_target

                    TD_N_reward.append(training_reward)
                    training_policy_target.append(policy_target)
                    training_action.append(inner_train_sample_buffer[-(TD_traj_leng)]["at"])
                    training_st.append(inner_train_sample_buffer[-(TD_traj_leng)]["st"])
                    training_rt.append(inner_train_sample_buffer[-(TD_traj_leng)]["rt"])
                    training_st_p1.append(inner_train_sample_buffer[-1]["p1_st"])

                    if len(training_policy_target) == 2 * batch_size or done == 1:
                        TD_N_reward_array = np.array(TD_N_reward[-batch_size:]).reshape(-1, 1)
                        training_policy_target_array = np.array(training_policy_target[-batch_size:]).reshape(-1, 1)
                        training_action_array = np.array(training_action[-batch_size:])
                        training_st_array = np.array(training_st[-batch_size:])
                        training_rt_array = np.array(training_rt[-batch_size:])
                        training_st_p1_array = np.array(training_st_p1[-batch_size:])
                        print("TD_N_reward:", TD_N_reward_array.shape)
                        print("training_policy_target_array :", training_policy_target_array.shape)
                        print("training_action:", training_action_array.shape)
                        print("training_st:", training_st_array.shape)
                        print("training_rt_array:", training_rt_array.shape)
                        print("training_st_p1_array:", training_st_p1_array.shape)

                        # sess.run(noisy_net_ops)

                        step_feed_dict = {model.x_holder: training_st_array,
                                          model.R_plus_plus1_v_holder: training_policy_target_array,
                                          model.actions_y_holder: training_action_array,
                                          model.rt_holder:training_rt_array,
                                          model.st_p1_holder:training_st_p1_array}

                        # encoder_output_real = sess.run(encoder_output, feed_dict=step_feed_dict)
                        # print("encoder_output_real:", encoder_output_real.shape)

                        V_value_real = sess.run([model.V_value], feed_dict=step_feed_dict)
                        print("V_value_real:", V_value_real)
                        # print("total_training_end!!")

                        _, actor_loss_real, critic_loss_real, total_loss_real, advantage_real, summary = sess.run(
                            [model.total_training_op, model.actor_loss, model.critic_loss, model.total_loss,
                             model.advantage, merged_summary_op],
                            feed_dict=step_feed_dict
                        )
                        print("advantage_real:", advantage_real)

                        summary_writer.add_summary(summary, global_step=update_times)

                        # when update count reach limit, update the target net variable

                        # sess.run(noisy_net_ops)
                        update_times += 1

                        del TD_N_reward[:batch_size]
                        del training_policy_target[:batch_size]
                        del training_action[:batch_size]
                        del training_st[:batch_size]
                        del training_st_p1[:batch_size]

                        # sess.run(noisy_net_ops)

                # gc.collect()
                print("i_episode:", i_episode)
                print("step:", step)

                if done:
                    episode_reward_summary = sess.run([episode_reward_summary_op],
                                                      feed_dict={model.episode_reward_holder: np.array(episode_reward)})
                    summary_writer.add_summary(episode_reward_summary[0], global_step=i_episode)
                    print("Episode finished after {} timesteps".format(step))
                    break

            # Save the variables to disk.
            save_path = saver.save(sess, "./A2C_multitask/breakout_A2C_one_loss_model.ckpt", global_step=i_episode)
            print("Model saved in path: %s" % save_path)
