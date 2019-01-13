import numpy as np
import matplotlib.pyplot as plt
import gym


# Helper to return the max action from the sum of Q1 and Q2
def max_action(Q1, Q2, state):
    values = np.array([Q1[state, a] + Q2[state, a] for a in range(2)])
    action = np.argmax(values)
    return action


# Turn the spaces from continuous --> Discrete
# Could have used a neural network to estimate, but just lop off at the
# end of the distributions
pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
pole_theta_vel_space = np.linspace(-4, 4, 10)
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)


def get_state(observation):
    cart_x, cart_x_dot, cart_theta, cart_theta_dot = observation
    cart_x = int(np.digitize(cart_x, cart_pos_space))
    cart_x_dot = int(np.digitize(cart_x_dot, cart_vel_space))
    cart_theta = int(np.digitize(cart_theta, pole_theta_space))
    cart_theta_dot = int(np.digitize(cart_theta_dot, pole_theta_vel_space))

    return (cart_x, cart_x_dot, cart_theta, cart_theta_dot)


# Helper to plot the running average of total rewards
def plot_running_average(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(total_rewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.show()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # Model Hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    # Construct our state space
    states = []
    for i in range(len(cart_pos_space) + 1):
        for j in range(len(cart_vel_space) + 1):
            for k in range(len(pole_theta_space) + 1):
                for l in range(len(pole_theta_vel_space) + 1):
                    states.append((i, j, k, l))

    # Initialize both Q tables
    Q1, Q2 = {}, {}
    for s in states:
        for a in range(2):
            Q1[s, a] = 0
            Q2[s, a] = 0

    num_games = 10000
    total_rewards = np.zeros(num_games)
    for i in range(num_games):
        # Print the game number every 1000 games
        if i % 1000 == 0:
            print('Starting game ' + str(i))
        done = False
        ep_rewards = 0
        observation = env.reset()
        while not done:
            s = get_state(observation)
            rand = np.random.random()
            # Choose the epsilon greedy strategy, or some random one
            a = max_action(Q1, Q2, s) if rand < (1-EPS) else env.action_space.sample()
            observation_, reward, done, info = env.step(a)
            ep_rewards += reward
            s_ = get_state(observation_)
            rand = np.random.random()

            if rand <= 0.5:
                a_ = max_action(Q1, Q1, s)
                Q1[s, a] = Q1[s, a] + ALPHA*(reward + GAMMA*Q2[s_, a_] - Q1[s, a])

            elif rand > 0.5:
                a_ = max_action(Q2, Q2, s)
                Q2[s, a] = Q2[s, a] + ALPHA*(reward + GAMMA*Q1[s_, a_] - Q2[s, a])

            observation = observation_

        # Update epsilon --> want epsilon to decay over more iterations
        EPS -= 2/(num_games) if EPS > 0 else 0
        total_rewards[i] = ep_rewards

    plot_running_average(total_rewards)

