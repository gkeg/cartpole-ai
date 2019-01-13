import numpy as np
import matplotlib.pyplot as plt
import gym


# Helper function to help us take the max action
def max_action(Q, state):
    values = np.array([Q[state, a] for a in range(2)])
    action = np.argmax(values)
    return action


# Make our observation space discrete
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


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # Model hyperparams
    ALPHA = 0.1
    GAMMA = 0.9
    EPS = 1.0

    # Construct our state space
    states = []
    for i in range(len(cart_pos_space)+1):
        for j in range(len(cart_vel_space)+1):
            for k in range(len(pole_theta_space)+1):
                for l in range(len(pole_theta_vel_space)+1):
                    states.append((i, j, k, l))

    Q = {}
    for s in states:
        for a in range(2):
            Q[s, a] = 0

    num_games = 1000
    total_rewards = np.zeros(num_games)
    for i in range(num_games):
        if i % 100 == 0:
            print('Starting game' + str(i))

        # Get the cart x position, cart velocity, pole theta, and pole velocity
        observation = env.reset()
        s = get_state(observation)
        rand = np.random.random()
        a = max_action(Q, s) if rand < (1-EPS) else env.action_space.sample()
        done = False
        ep_rewards = 0

        while not done:
            observation_, reward, done, info = env.step(a)
            s_ = get_state(observation_)
            rand = np.random.random()
            a_ = max_action(Q, s_) if rand < (1-EPS) else env.action_space.sample()
            ep_rewards += reward
            Q[s, a] += ALPHA*(reward + GAMMA*Q[s_, a_] - Q[s, a])
            a, a = s_, a_
        EPS -= 2/(num_games) if EPS > 0 else 0
        total_rewards[i] = ep_rewards



