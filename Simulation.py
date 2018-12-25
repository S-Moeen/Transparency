import numpy as np
class Agent():
    def __init__(self,initial_belief = 1, old_belief_coefficient = 1, new_signal_coefficient = 1, public_belief_coefficient = 1, friends_signal_coefficient = 1,*args, **kwargs):
        '''initial belief is the agent's prior on state of the world being 1
        coefficients are coefficients in agents linear update '''
        self.belief = initial_belief
        self.old_belief_coefficient = old_belief_coefficient
        self.new_signal_coefficient = new_signal_coefficient
        self.public_belief_coefficient = public_belief_coefficient
        self.friends_signal_coefficient = friends_signal_coefficient
        self.private_coefficient_sum = self.old_belief_coefficient+self.new_signal_coefficient+self.public_belief_coefficient
        self.friend_coefficient_sum = self.old_belief_coefficient+self.friends_signal_coefficient+self.public_belief_coefficient
        self.neighbours = None
    
    def set_neighbours(self, neighbours):
        '''sets neighbours, otherwise neighbours are None'''
        self.neighbours = neighbours

    def update_belief_with_private(self, new_signal, public_signal):
        '''Updates the belief based on new private signal'''
        self.belief = (self.old_belief_coefficient*self.belief+
                    self.new_signal_coefficient*new_signal+
                    self.public_belief_coefficient*public_signal) / self.private_coefficient_sum

    def update_belief_from_friend(self, new_signal, public_signal):
        ''' Updates belief based on a friend's map'''
        self.belief = (self.old_belief_coefficient*self.belief+
                    self.friends_signal_coefficient*new_signal+
                    self.public_belief_coefficient*public_signal) / self.private_coefficient_sum

    def get_mapped_belief(self):
        '''returns map of the belief'''
        if (self.belief > 0.5):
            return 1
        return 0

    def recieve_and_act(self, private_signal, public_signal, transparency):
        ''' recieves signals, updates himself and stochastically
         based on transparency updates neighbours with map of his signal 
         returns its signal to others
         '''
        self.private_signal = private_signal
        self.update_belief_with_private(private_signal, public_signal)
        signal = self.get_mapped_belief()
        if( np.random.uniform(size = 1) < transparency):
            for agent in self.neighbours:
                agent.update_belief_from_friend(signal ,public_signal)
        return signal

class Simulation():
    def __init__(self, *args, **kwargs):
        pass
    
    def initialize(self, number_of_agents, transparency = 0.5, initial_belief = 1, old_belief_coefficient = 1, new_signal_coefficient = 1, public_signal_coefficient = 1, p = 0.5):
        '''Creates and initializes agents
        p is parameter of the bernouli
        '''
        self.number_of_agents = number_of_agents
        self.agents = [Agent(initial_belief ,old_belief_coefficient ,new_signal_coefficient,public_signal_coefficient) 
                    for i in range(number_of_agents)]
        self.public_signal = 0.5
        self.p = p
        self.transparency = transparency
        for agent in self.agents:
            neighbours = self.agents.copy()
            neighbours.remove(agent)
            agent.set_neighbours(neighbours)

    def simulate(self, number_of_experiments):
        '''Runs experiment for the number_of_experiments iterations
        at each step in every iteration each agent recieves a private signal and updates his/her belief
        based on transparency updates neighbours with map of his signal 
        public signal updates at the end of each cycle with map of agents\' signals '''
        for i in range(number_of_experiments):
            sum = 0
            for agent in self.agents:
                private_signal = self.create_signal()
                sum += agent.recieve_and_act(private_signal, self.public_signal, self.transparency)
            self.public_signal = (self.public_signal*i + sum/self.number_of_agents) / (i+1)
            print(self.public_signal)
            
    def create_signal(self):
        ''' create a signal based on informativeness '''
        return np.random.binomial(1, self.p, size=1)
        
sim = Simulation()
sim.initialize(10,transparency=1, p = 0.7)
sim.simulate(10000)

    


        

        