import pomdp_py

from domain.action import *
from domain.observation import *
from domain.state import *

import random

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, dim):
        self.dim = dim

    def get_all_actions(self, state=None, history=None): # parent class requires this argument format (self, state=None, history=None)
        # remove actions that pass boundaries?
        all_actions = ALL_ACTIONS.copy()
        if state == None:
            return all_actions
        
        else:
            cur_pose = state.pose
            if cur_pose[0] == 9:
                all_actions.remove(MoveEast2D)
            if cur_pose[0] == 0:
                all_actions.remove(MoveWest2D)
            
            if cur_pose[1] == 9:
                all_actions.remove(MoveSouth2D)
            if cur_pose[1] == 0:
                all_actions.remove(MoveNorth2D)
            return all_actions
    
    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]
    
    def rollout(self, state=None, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]

