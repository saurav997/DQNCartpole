import subprocess
import sys

# Function to install dependencies from requirements.txt
def install_requirements():
    with open('requirements.txt') as f:
        requirements = f.readlines()
    requirements = [x.strip() for x in requirements]
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
    except subprocess.CalledProcessError as e:
        print("Error: Failed to install required packages:", e)

# Check if all required packages are installed
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import gym
    import random
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("Some required packages are not installed. Installing...")
    install_requirements()


class DQN:
    
    REPLAY_MEMORY_SIZE = 2000             # number of tuples in exp_tup replay  
    EPSILON = 1                         # epsilon of epsilon-greedy exploration
    EPSILON_DECAY = 0.95                 # exponential decay multiplier for epsilon
    EPSILON_DECAY_FREQ = 10           # epsilon decay frequency
    k = 128
    HIDDEN1_SIZE = k                    # size of hidden layer 1
    HIDDEN2_SIZE = k                  # size of hidden layer 2
    HIDDEN3_SIZE = k                    # size of hidden layer 3
    EPISODES_NUM = 50               # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200                     # maximum number of steps in an episode 
    LEARNING_RATE = 1e-4                 # learning rate and other parameters for SGD/RMSProp/Adam
    MINIBATCH_SIZE = 256                    # size of minibatch sampled from the exp_tup replay
    DISCOUNT_FACTOR = 0.9                 # MDP's gamma
    TARGET_UPDATE_FREQ =10             # number of steps (not episodes) after which to update the target networks 
    LOG_DIR = './logs'                     # directory wherein logging takes place
    REPLAY = True
    EPS_MIN= 1e-5
    # Creating and initializing the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]        # In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n                    # In case of cartpole, 2 actions (right/left)
    
    # Create the Q-network
    def initialize_network(self):

          # placeholder for the state-space input to the q-network
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        ############################################################
        # Designing q-network here.
        #############################################################

        with tf.name_scope('output'):
            
            #primary network weights
            self.primary_weights = {
                'w1' : tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], mean = 0, stddev=0.1), dtype = tf.float32, name = "weight1" ),
                'b1' : tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name = "bias1" ),
                'w2' : tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], mean = 0, stddev=0.01), name = "weight2" ),
                'b2' : tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name = "bias2" ),
                'w3' : tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.HIDDEN3_SIZE], mean = 0, stddev=0.01), name = "weight3" ),
                'b3' : tf.Variable(tf.zeros(self.HIDDEN3_SIZE), name = "bias3" ),
                'w4' : tf.Variable(tf.truncated_normal([self.HIDDEN3_SIZE, self.output_size], mean = 0, stddev=0.01), name = "weight4" ),
                'b4' : tf.Variable(tf.zeros(self.output_size), name = "bias4" )
            }

            # Defining the primary network
            with tf.name_scope('Primary_network'):
                p_h1 = tf.nn.relu(tf.matmul(self.x, self.primary_weights['w1']) +  self.primary_weights['b1'])
                p_h2 = tf.nn.relu(tf.matmul(p_h1, self.primary_weights['w2']) + self.primary_weights['b2'])
                p_h3 = tf.nn.relu(tf.matmul(p_h2, self.primary_weights['w3']) + self.primary_weights['b3'])
                # Q value from primary network
                self.Q_p = tf.matmul(p_h3, self.primary_weights['w4']) + self.primary_weights['b4']
            
            #target netwok weights
            self.target_weights = {
                'w1' : tf.Variable(tf.truncated_normal( [self.input_size, self.HIDDEN1_SIZE], mean = 0, stddev=0.01), dtype = tf.float32, name = "target_weight1" ),
                'b1' : tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name = "target_bias1" ),
                'w2' : tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], mean = 0, stddev=0.01), name = "target_weight2" ),
                'b2' : tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name = "target_bias2" ),
                'w3' : tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.HIDDEN3_SIZE], mean = 0, stddev=0.01), name = "target_weight3" ),
                'b3' : tf.Variable(tf.zeros(self.HIDDEN3_SIZE), name = "target_bias3"),
                'w4' : tf.Variable(tf.truncated_normal([self.HIDDEN3_SIZE, self.output_size], mean = 0, stddev=0.01), name = "target_weight4" ),
                'b4' : tf.Variable(tf.zeros(self.output_size), name = "target_bias4")
            }
            # Defining the target network
            with tf.name_scope('Target_network'):
                t_h1 = tf.nn.relu(tf.matmul(self.x, self.target_weights['w1']) + self.target_weights['b1'])
                t_h2 = tf.nn.relu(tf.matmul(t_h1, self.target_weights['w2']) + self.target_weights['b2'])            
                t_h3 = tf.nn.relu(tf.matmul(t_h2, self.target_weights['w3']) + self.target_weights['b3']) 
                # Q value from target network
                self.Q_t = tf.matmul(t_h3, self.target_weights['w4']) + self.target_weights['b4']


        ############################################################
        # Loss Computation steps:
        #
        # 1.compute the q-values for the actions in the (s,a,s',r) 
        # tuples from the exp_tup replay's minibatch
        #
        # 2.compute the l2 loss between these estimated q-values and 
        # the target (which is computed using the frozen target network)
        #
        ############################################################
        self.exp_rep_buffer = []
        self.buffer_pos = 0
        with tf.name_scope('pre-processing'):
            self.sel_action = tf.placeholder(dtype = tf.int32, shape= [None], name = 'actions') 
            self.one_hot_sel_act = tf.one_hot(self.sel_action, self.output_size, 1.0, 0.0, name = 'one_hot_sel_act')
            self.predicted = tf.reduce_sum(tf.multiply(self.Q_p, self.one_hot_sel_act, name = 'predicted'), reduction_indices = [1])
            self.expected = tf.placeholder(dtype = tf.float32, shape= [None], name = 'expected')                
            self.loss = tf.losses.mean_squared_error(self.expected, self.predicted)

        ############################################################
        # Chosen gradient descent algorithm : Adam. 
        ############################################################
        with tf.name_scope('Optimizer'):
            self.loss = tf.losses.mean_squared_error(self.expected, self.predicted)
            optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)


        ############################################################

    def train(self, episodes_num=EPISODES_NUM):
        
        # Initialize summary for TensorBoard                         
        summary_writer = tf.summary.FileWriter(self.LOG_DIR)    
        summary = tf.Summary()    
        # Note for self: Alternatively, real-time plots from matplotlib can be used 
        # (https://stackoverflow.com/a/24228275/3284912)
        
        # Initialize the TF session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
        ############################################################
        # Initialize other variables (like the replay memory)
        ############################################################

        ############################################################
        # Main training loop
        # 
        # In each episode the following actions are performed:
        #    pick the action for the given state, 
        #    perform a 'step' in the environment to get the reward and next state,
        #    update the replay buffer,
        #    sample a random minibatch from the replay buffer,
        #    perform Q-learning,
        #    update the target network, if required.
        ############################################################
        
        
        
        #initially assuming all both primary and target have same initial primary_weights
        for w in self.primary_weights:
            self.session.run(tf.assign(self.target_weights[w], self.primary_weights[w]))
        glob_step = 0
        returns = []
        for episode in range(episodes_num):
            state = self.env.reset()
            state = np.reshape(state, [1, self.input_size])
            ############################################################
            # Episode-specific initializations done here.
            ############################################################
            eps_len = 0
            eps_rew = 0
            #
            ############################################################
            while True:
                ############################################################
                # Picking the next action using epsilon greedy and and executing it
                ############################################################
                if np.random.uniform(0,1)<self.EPSILON:
                    action = self.env.action_space.sample()
                else:
                    Q_values = self.session.run(self.Q_p, feed_dict={self.x : state})
                    action = np.argmax(Q_values)
                

                ############################################################
                # Step in the environment. :
                ############################################################

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.input_size])
                glob_step += 1 
                eps_len += 1 
                eps_rew += reward
                


                ############################################################
                # Update the (limited) replay buffer. 
                #
                # Note : when the replay buffer is full, I 
                # remove an entry to accommodate a new one.
                ############################################################

                # Your code here
                if len(self.exp_rep_buffer) < self.REPLAY_MEMORY_SIZE:
                    # creating space for expirience tupple
                    self.exp_rep_buffer.append(None)
                # The expirience comes out to be
                exp_tup = (state, action, reward, next_state, done)
                # add expirience_tupple at the buffer_position
                self.exp_rep_buffer[self.buffer_pos] = exp_tup
                self.buffer_pos = (self.buffer_pos+1)%self.REPLAY_MEMORY_SIZE

                ############################################################
                # Sample a random minibatch and perform Q-learning (fetch max Q at s') 
                #
                # the target (r + gamma * max Q) is computed    
                # with the help of the target network.
                # Compute this target and pass it to the network for computing 
                # and minimizing the loss with the current estimates
                #
                ############################################################

                #Note: 2 conditions are applicability of REPLAY option 
                #and buffer_size bigger than minibatch_size
                if len(self.exp_rep_buffer) >= self.MINIBATCH_SIZE and self.REPLAY==True:
                    #selecting random batch
                    batch = random.sample(self.exp_rep_buffer, self.MINIBATCH_SIZE)
                    # Separately extracting states,actions,next_states,rewards,dones from each batch
                    states = [exp[0] for exp in batch]
                    actions = [exp[1] for exp in batch]
                    rewards = [exp[2] for exp in batch]
                    next_states = [exp[3] for exp in batch]
                    dones = [exp[4] for exp in batch]
                    #And feeding minibatch information to Q-network
                    for i in range(len(states)):
                        exp_val = rewards[i]
                        exp_val = np.expand_dims(exp_val, axis = 0)
                        if not dones[i]:
                            exp_val = self.DISCOUNT_FACTOR*np.amax(self.session.run(self.Q_t, feed_dict={self.x : next_states[i]}), axis=1) + rewards[i]
                            actions[i] = np.expand_dims(actions[i], axis = 0)
                            _, losses = self.session.run([self.train_op, self.loss], feed_dict={self.x : states[i], self.expected : exp_val, self.sel_action : actions[i]})
                        else:
                            exp_val = np.expand_dims(rewards[i],axis=0)
                            actions[i] = np.expand_dims(actions[i], axis = 0)
                            _, losses = self.session.run([self.train_op, self.loss], feed_dict={self.x : states[i], self.expected : exp_val, self.sel_action : actions[i]})
               
                #else if replay option is not applicable
                else:
                    if not done:
                        exp_val = self.DISCOUNT_FACTOR*np.amax(self.session.run(self.Q_t, feed_dict={self.x : next_state}), axis=1) + reward
                    else:
                        exp_val = reward
                        
                #Updating current state to the next
                state = next_state
                ############################################################
                  # Update target primary_weights. 
                ############################################################
                # Epsilon Decay
                if glob_step%self.EPSILON_DECAY_FREQ == 0:
                    self.EPSILON *= self.EPSILON_DECAY
                    if self.EPSILON < self.EPS_MIN:  
                        self.EPSILON = self.EPS_MIN
                if glob_step % self.TARGET_UPDATE_FREQ == 0:
                    for w in self.primary_weights:
                        self.session.run(tf.assign(self.target_weights[w], self.primary_weights[w]))
                
                
                

                
                ############################################################
                # Break out of the loop if the episode ends
                ############################################################

                if done or (eps_len == self.MAX_STEPS):
                    returns.append(eps_rew)
                    break
            ############################################################
            # Logging. 
            #
            # This is what gives an idea of how good the current
            # experiment is, and if one should terminate and re-run with new parameters
            # The earlier you learn how to read and visualize experiment logs quickly,
            # the faster you'll be able to prototype and learn.
            ############################################################
                
            print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, eps_len, glob_step))
            summary.value.add(tag="episode length", simple_value=eps_len)
            summary_writer.add_summary(summary, episode)
            plt.plot(returns)

    # Simple function to visually 'test' a policy
    def playPolicy(self):
        done = False
        steps = 0
        state = self.env.reset()
        # the CartPole task is taken to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            self.env.render()
            q_vals = self.session.run(self.Q_p, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1
        return steps
    def closeenv(self):
        #closing the environment
        self.env.close()

if __name__ == '__main__':
    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    dqn.initialize_network()
    print("\nStarting training...\n")
    dqn.train()
    print("\nFinished training...\nCheck out some demonstrations\n")
    # Visualize the learned behaviour for a few episodes
    results = []
    for i in range(50):
        eps_len = dqn.playPolicy()
        print("Test steps = ", eps_len)
        results.append(eps_len)
        dqn.closeenv()
    print("Mean steps = ", sum(results) / len(results))    
    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")