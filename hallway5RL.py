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
    def step(self, action): # Return the current state, reward, if_done
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

env = HallwayEnv()

state = env.reset()
done = False

# All right policy
def all_left():
    return "LEFT"

while not done:
    action = all_left()
    next_state, reward, done = env.step(action)

    print(state, action, next_state, reward, done)

    state = next_state


