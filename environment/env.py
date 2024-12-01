import pomdp_py

from models.reward_model import *

class Environment(pomdp_py.Environment):

    def __init__(self, init_state, dim=(3, 3)):
        self.width, self.length = dim

        self.robot_state = init_state['robot']
        self.reward_state = init_state['reward']

        transition_model = TransitionModel(dim=dim)
        reward_model = RewardModel(transition_model=transition_model, reward_state=self.reward_state)

        super().__init__(self.robot_state, # agent state 
                         transition_model, 
                         reward_model
                         )
        
    def get_world_state(self):
        return [self.robot_state, self.reward_state]

    def get_robot_state(self):
        return self.cur_state

    def update_reward(self, new_reward_state, value):
        self.reward_state = new_reward_state
        self.reward_model.reward_pose = {new_reward_state.pose: value}

    def state_transition(self, action, execute=True):
        
        # self.cur_state defined in parent 

        next_state = self.transition_model.sample(state=self.cur_state, action=action)
        reward = self.reward_model.sample(state=self.cur_state, action=action, next_state=next_state)

        return next_state, reward
    
    
        