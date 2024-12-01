import pomdp_py
from domain.action import *
from domain.state import *

'''
T(s'|s,a)
'''
class TransitionModel(pomdp_py.TransitionModel):

    # TODO: Encode obstacles into transitions

    def __init__(self, dim):
        self.prob_next = 0.8
        self.dim = dim # 10, 10 environment
    
    def get_all_states(self):
        # TODO: review source. all states or all reachable states?
        return [ObjectState(objtype='robot', objid=0, pose=(i, j)) for j in range(self.dim[1]) for i in range(self.dim[0])]

    def probability(self, next_state, state, action):
        # NOTE: not considering if obstacle present in next state
        if next_state.objtype != state.objtype:
            return 0.
        
        if next_state != self.argmax(state=state, action=action):
            return (1 - self.prob_next) / 4.
        
        return self.prob_next
    
    def argmax(self, state: ObjectState, action: MoveAction):
        '''
            Get next state from current state and taking action at current state
        '''
        # NOTE: not considering if obstacle present in next state
        objid = state.objid
        objtype = state.objtype
        pose = state.pose

        # do not update pose if out of bounds
        new_pose = pose
        if pose[0] + action.motion[0] in range(self.dim[0]) and pose[1] + action.motion[1] in range(self.dim[1]):
            new_pose = (pose[0] + action.motion[0], pose[1] + action.motion[1])

        new_state = ObjectState(objid=objid, objtype=objtype, pose=new_pose)
        return new_state
    
    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
