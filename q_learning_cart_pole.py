import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

MAX_STATES = 10**4
GAMMA = 0.9
ALPHA = 0.01


# Helps us extract finding maximal element of the Q array between <state, action>
# Needed to find the maximal probability action for a given state
def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


def initialize_bins():
    bins = np.zeros((4, 10))

    # Lop off the distributions at either end
    # obs[0] is the cart position --> -4.8 ... 4.8
    # obs[1] is the cart velocity --> -inf ... inf
    # obs[2] is the pole angle --> -41.8, 41.8
    # obs[3] is the pole velocity --> -inf ... inf
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)

    return bins


# Bucket the continuous observations into discrete bins
def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


# Turn a state into a string for keying purposes
def state_to_string(state):
    return ''.join(str(int(e)) for e in state)


def all_states_to_strings():
    states = []
    for i in range(MAX_STATES):
        states.append(str(i).zfill(4))

    return states


# Initializes our Q table to contain all zeros
def initialize_Q():
    Q = {}

    all_states = all_states_to_strings()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q


# Helper function to play one game
def play_one_game(bins, Q, epsilon=0.5):
    observation = env.reset()
    done = False
    count = 0
    state = state_to_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < epsilon: # Get a random action 50% of the time
            act = env.action_space.sample() # Epsilon greedy strategy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _ = env.step(act)

        total_reward += reward

        if done and count < 200:
            reward = -300

        state_new = state_to_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA*(reward - GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, count


def play_many_games(bins, N=1000):
    Q = initialize_Q()

    lengths = []
    rewards = []
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)

        episode_reward, episode_length = play_one_game(bins, Q, eps)

        # Print every 100 iterations
        if n % 100 == 0:
            print(n, '%.4f' % eps, episode_reward)
        lengths.append(episode_length)
        rewards.append(episode_reward)
    return lengths, rewards


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        # Take the average of the last 100 games
        running_avg[t] = np.mean(total_rewards[max(0, t-100): (t+1)])

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == "__main__":
    bins = initialize_bins()
    episode_lengths, episode_rewards = play_many_games(bins)
    plot_running_avg(episode_rewards)

