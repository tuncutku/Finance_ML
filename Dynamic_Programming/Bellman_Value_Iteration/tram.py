# Source: Stanford
# Link: https://www.youtube.com/watch?v=9g32v7bK3Co
# Topic: Lecture 7: Markov Decision Processes - Value Iteration | Stanford CS221

import sys
import os

sys.setrecursionlimit(10000)

### Model (Search problem)
class TransportationMDP(object):
    def __init__(self, N):
        self.N = N

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        # return list of alid actions
        result = []
        if state + 1 <= self.N:
            result.append("walk")
        if state * 2 <= self.N:
            result.append("tram")
        return result

    def succPropReward(self, state, action):
        # return list of (newState, prop, reward) triples
        # state s, action a, newState s'
        # prob = T(s, a, s'), reward = Reward(s, a, s')
        result = []
        if action == "walk":
            result.append((state + 1, 1.0, -1.0))
        elif action == "tram":
            failProb = 0.5
            result.append((state * 2, 1.0 - failProb, -2))
            result.append((state, failProb, -2))
        return result

    def discount(self):
        return 1

    def states(self):
        return range(1, self.N + 1)


# Inference (Algorithms)
def valueIteration(mdp):
    # Initialize
    V = {}  # state -> Vopt[state]
    for state in mdp.states():
        V[state] = 0.0

    def Q(state, action):

        List = []

        for newState, prob, reward in mdp.succPropReward(state, action):
            List.append(prob * (reward + mdp.discount() * V[newState]))

        return sum(List)

    while True:
        # Compute the new values (newV) given the old values (V)
        newV = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state] = 0.0
            else:
                newV[state] = max(Q(state, action) for action in mdp.actions(state))

        # Check for convergence
        if max(abs(V[state] - newV[state]) for state in mdp.states()) < 1e-10:
            break
        V = newV

        # Read out policy
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state] = "none"
            else:
                pi[state] = max(
                    (Q(state, action), action) for action in mdp.actions(state)
                )[1]

        # Print stuff out
        os.system("clear")
        print("{:15} {:15} {:15}".format("s", "V(s)", "pi(s)"))
        for state in mdp.states():
            print("{:15} {:15} {:15}".format(state, V[state], pi[state]))
        input()


mdp = TransportationMDP(N=10)
# print(mdp.actions(3))
# print(mdp.succPropReward(3, "walk"))
# print(mdp.succPropReward(3, "tram"))
valueIteration(mdp)
