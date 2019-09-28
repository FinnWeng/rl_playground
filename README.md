# rl_playground

This playground is to record some result and tips from my experiment.



The name of Reinforcemnet Learning is quite comfusing. 

For me, I see "learning" as infrastructure, which not necessarily contains neural network.

Deep "learning" is simple one. 

Reinforcemnet Learning is complex one which combines HMM.

Self-supervised learning is using multi-task trick to enhance model ability.



So by this insight, my implementations contain several modules:

1. Main: the "Break_out_TD_A2C". The most important implementation. It defines how the model been trained, the training method is exactly the "algorithm of reinforcement.

2. Model: the "RL_model". Some neural network implementation. Could be as easy as several CNN layers. Or could contain complex implementation of curious-driven unit.

3. Module: like "octave_module". Some handy implementation.



The hardest part is how to compute the reward correctly. The design must rightly deal with reward, timing of training, end of game. 
The second one is the policy gradient loss. It is very different from the loss we use every day.

 
Even I implement above all correctly, still I can't promise it will work perfectly for every case.
For me, there's severly tips to deal with conditions that model is broken.

1. If all probability of action stuck for every frame in the end: decrease the learning rate.

2. How to know that all hyperparameter are been set properly: observe the probabilities change frame by frame.

3. How to know that it is converging: the episode reward should go up. But the process is extremely slow. I take a month to train. So if it seems not converge for days, be patient.

4. The losses of actor and critic explain few.

5. The advantage should be sometimes positive, sometimes negative.


By all the tips and correct implementation, I take a month to train. It performs not particular good. But it indeed learned how to catch the ball.

For now, I try curiosity model to accelerate the training process.





