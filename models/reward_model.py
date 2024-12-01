import pomdp_py

from models.transition_model import *

from domain.action import *
from domain.state import *

class RewardModel(pomdp_py.RewardModel):

    # for reward states make object states: 0-robot-pose, 0-box-pose
    # check for the states of the boxes to reward robot for moving the box
    # only the robot can move the box
    '''
    during loop boxes randomly appear -> update reward state to reflect this
        logic:
            robot go to pick up box: reward_state for pick box
            robot go to place box: reward_state for place box

            change between these two reward goals based on robot state according to, busy (holding box) vs free (not holding box)
    ''' 

    def __init__(self, transition_model, 
                 reward_state # type: ObjectState 
                 ):

        # updating this might shift agents goals during execution
        self.reward_states = {reward_state: 10}
        self.reward_pose = {reward_state.pose: 10}
        self.transition_model = transition_model

    def probability(self, reward, state, action):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    # def argmax(self):
    #     pass

    def sample(self, state, action, next_state):
        # Obtain reward according to reward function.
        return self._reward_func(next_state=next_state, state=state, action=action)
    
    # def get_distribution(self):
    #     pass

    # reward function <- sum(s') R(s', a)T(s'|s, a)
    def _reward_func(self, state, action, next_state):
        cur_pose = state.pose
        next_pose = next_state.pose

        reward = 0

        if state.pose == next_pose:
            reward += -5

        next_reward = 0 if next_pose not in self.reward_pose else self.reward_pose[next_pose]

        reward += self.transition_model.probability(next_state=next_state, state=state, action=action) * next_reward

        # if state.pose in self.reward_states:
        #     print(reward, action)

        return reward

if __name__ == '__main__':
    tm = TransitionModel(dim=(3, 3))
    rm = RewardModel(transition_model=tm, reward_state=ObjectState(objtype='reward', objid=0, pose=(1, 1)))

    s = ObjectState(objtype='robot', objid=0, pose=(1,1))
    sn = ObjectState(objtype='robot', objid=0, pose=(0,1))

    r = rm.sample(state=sn, action=MoveStay2D, next_state=sn)
    print(r)