import random
# Assign states
states = [0, 1, 2, 3, 4, 5, 6]
# Assign actions
actions = ["LEFT", "RIGHT"]
# P[state][action] = ...
P = {
    # State:
    # Action: [(probability, next_state), (probability, next_state)]
    1: {
        "LEFT": [(0.8, 0), (0.2, 2)],
        "RIGHT": [(0.2, 0), (0.8, 2)]
    },
    2: {
        "LEFT": [(0.8, 1), (0.2, 3)],
        "RIGHT": [(0.2, 1), (0.8, 3)]
    },
    3: {
        "LEFT": [(0.8, 2), (0.2, 4)],
        "RIGHT": [(0.2, 2), (0.8, 4)]
    },
    4: {
        "LEFT": [(0.8, 3), (0.2, 5)],
        "RIGHT": [(0.2, 3), (0.8, 5)]
    },
    5: {
        "LEFT": [(0.8, 4), (0.2, 6)],
        "RIGHT": [(0.2, 4), (0.8, 6)]
    }

}
# Terminal States
terminal_states = [0, 6]
# Discount factor
gamma = 0.85

# We need to be able to keep track of the agents current position
class HallwayEnv:
    # Create initializer
    def __init__(self, initial_state=3, gamma=0.85):
        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.actions = actions
        self.P = P
        self.terminal_states = [0, 6]
        self.gamma = gamma # Discount factor
        self.initial_state = initial_state
        self.state = initial_state
    # method to reset the position
    def reset(self):
        self.state = self.initial_state
        return self.state
    def reward(self, next_state):
        if next_state == self.states[-1]:
            return 1
        elif next_state == self.states[0]:
            return -1
        else:
            return 0
    def is_terminal(self, state):
        return state in self.terminal_states
    def step(self, action): # Return the next state, reward, if_done
        if self.is_terminal(self.state):
            return self.state, 0, True

        # Get probabilities and possible next states of a given state and action
        transitions = self.P[self.state][action]

        # Collect probabilities into a list
        probabilities = [transition[0] for transition in transitions]
        possible_states = [transition[1] for transition in transitions]

        next_state = random.choices(possible_states, weights=probabilities)[0]

        self.state = next_state
        r = self.reward(self.state)
        done = self.is_terminal(self.state)
        return self.state, r, done

def random_policy():
    random_num = random.randint(1, 2)
    if random_num == 1:
        return "LEFT"
    elif random_num == 2:
        return "RIGHT"

h_env = HallwayEnv()

state = h_env.reset()

    
policy = random_policy()
# Run episodes and track return 
def run_episode(env, policy, max_steps=100):
    # Reset the environment
    state = env.reset()
    done = False
    step_count = 0
    trajectory = []
    total_reward = 0
    while (not done) and step_count < max_steps:
        action = policy(state)
        next_state, r, done = env.step(action)
        trajectory.append((state, action, next_state, r))
        total_reward += r * (gamma ** step_count)
        step_count += 1
        state = next_state
    return trajectory, total_reward, step_count

# We now will right an evaluate policy function to compare policies
# This doesn't use bellman equations yet, just average returns
def evaluate_policy(env, policy, n_episodes=1000):
    ## Loop through for requested number of episodes
    all_reward = 0
    for i in range(n_episodes):
        trajectory, reward, step_count = run_episode(env, policy(state))
        all_reward += reward
    return all_reward / n_episodes


def all_left(state):
    return "LEFT"

# Let's create an iterative policy evaluation
def iterative_policy_evaluation(env, policy, theta=1e-6):
    prev_V= {s: 0.0 for s in env.states}

    while True:
        V = {s: 0.0 for s in env.states}
        # make sure state is not terminal
        # loop through every state, 
        for s in env.states:
            if env.is_terminal(s):
                continue
            # gets us the next state and probability of that next state
            # given the current state a policy
            for probability, next_state in env.P[s][policy(s)]:
                r = env.reward(next_state)
                done = env.is_terminal(next_state)
                V[s] += probability * (r + env.gamma * prev_V[next_state] * (not done))
        delta = max([abs(V[s] - prev_V[s]) for s in env.states])
        if delta < theta:
            break 
        prev_V = V.copy()
    return V  



def all_right(state):
    return "RIGHT"

V = iterative_policy_evaluation(h_env, all_left)
print(V)
