import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import defaultdict


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # self.q_table = [(self.state, random.choice([None,'forward','left','right']))] 
        # self.q_table = [(self.state,  (random.choice([None,'forward','left','right'])))]
        # self.q_value = {}

        # def initialize_Q_values(self, val=4.0):
        self.next_waypoint = None
        self.reward = 0
        self.q_table = self.initialize_q_table()
        self.penalty = 0
        self.successes = 0        
        # self.q_table = {}
        # val = 4.0
        # for waypoint in ['left', 'right', 'forward']:
        #     for light in ['red', 'green']:
        #         for oncoming in ['left', 'right', 'forward', 'none']:
        #             for action in ['left', 'right', 'forward', 'none']:
        #                 self.q_table[((waypoint, light, oncoming), action)] = val
        # return self.q_value
        # return self.q_value      
        # initialize_Q_values(self, val=4.0)        
  

    def initialize_q_table(self, val=5.0):
        q_table = {}
        for waypoint in ['left', 'right', 'forward', None]:
            for light in ['red', 'green']:
                for left in ['left', 'right', 'forward', None]:
                    for oncoming in ['left', 'right', 'forward', None]:
                        for action in self.env.valid_actions:
                            q_table[((waypoint, light, left, oncoming), action)] = val
        return q_table

    def max_q(self, state):
        action = None
        max_Q = 0.0
        for a in self.env.valid_actions:
            q_value = self.q_table[(state, a)]
            if q_value > max_Q:
                action = a
                max_Q = q_value
        return (max_Q, action)

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        # self.state = None
        # self.next_waypoint = None

    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        penalty = 0
        successes = 0
        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'])
        # TODO: Select action according to your policy
        # action = random.choice([None,'forward','left','right'])
        (q_value, action) = self.max_q(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        alpha = 0.9
        gamma = 0.1
        
        # current_state = self.get_state

        # update input, waypoint, state
        inputs_new = self.env.sense(self)
        next_next_waypoint = self.planner.next_waypoint()
        next_state = (next_next_waypoint, inputs_new['light'], inputs['left'], inputs_new['oncoming'])

        (next_q, next_action) = self.max_q(next_state)
        q_value += alpha * (reward + gamma * next_q - q_value)
        self.q_table[(self.state, action)] = q_value
        # q_value =  (1.0 - alpha) * self.q_value[self.state][action] + alpha * (reward + gamma * argmax(self, self.state)) 

        # Print the status at update
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, waypoint{}".format(deadline, inputs, action, reward, self.next_waypoint)  # [debug]
        # Count values for result analysis
        if (reward < 0):
            self.penalty += 1
            print "Negative reward: deadline = {}, inputs = {}, action = {}, reward = {}, waypoint = {}, penalty = {}".format(deadline, inputs, action, reward, self.next_waypoint, self.penalty)
        # location = self.env.agent_states[self]['location']   
        # heading = self.env.agent_states[self]['heading'] 
        # destination = self.env.agent_states[self]["destination"]
        # next_location = ((self.env.agent_states[self]['location'][0] + self.env.agent_states[self]['heading'][0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
        #                     (self.env.agent_states[self]['location'][1] + self.env.agent_states[self]['heading'][1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1]) 
        # if self.env.act(self, action) == 10 :
        #     self.successes += 1
        #     # print "Success: deadline = {}, inputs = {}, action = {}, reward = {}, waypoint {}, penalty{}, successes{}".format(deadline, inputs, action, reward, self.next_waypoint, self.penalty, self.successes)
        # # print next_location 
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, waypoint = {}, penalties = {}".format(deadline, inputs, action, reward, self.next_waypoint, self.penalty)  # [debug]
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, waypoint = {}, penalties = {}, location = {}, destination = {}, successes = {}".format(deadline, inputs, action, reward, self.next_waypoint, self.penalty, location, destination, self.successes)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
