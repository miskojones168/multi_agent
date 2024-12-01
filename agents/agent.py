import pomdp_py

from models.transition_model import *
from models.observation_model import *
from models.reward_model import *
from models.policy_model import *

class Agent(pomdp_py.Agent):
    def __init__(
            self,
            robot_id,
            init_robot_state,  # initial robot state (assuming robot state is observable perfectly)
            reward_state,
            dim=(3,3)
        ):

        # self.current_state = init_robot_state

        init_robot_pose = init_robot_state.pose
        
        transition_model = TransitionModel(dim=dim) # <-- encode obstacles and walls into the transition model. They are not reachable
        observation_model = CustomObservationModel(dim=dim)
        reward_model = RewardModel(transition_model=transition_model, reward_state=reward_state) # <-- dinamically update reward_state during execution?
        policy_model = PolicyModel(dim=dim)
        init_belief = pomdp_py.Histogram({ObjectState(objtype='robot', objid=robot_id, pose=init_robot_pose): 1.0})

        super().__init__(
            init_belief,
            policy_model,
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=reward_model,
        )
    
    def clear_history(self):
        self._history = ()
    
    def update_belief(self, belief: pomdp_py.Histogram, prior: bool = False):
        self.set_belief(belief, prior=prior)
