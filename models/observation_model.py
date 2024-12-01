import pomdp_py

from domain.action import *
from domain.state import *
from domain.observation import *

class CustomObservationModel(pomdp_py.ObservationModel):
    def __init__(self, dim):
        self.dim = dim

    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        # assume P(o|s',a) = 1 full observability
        return 1.0

    def get_all_observations(self):
        return [Observation(objid=0, pose=(i, j)) for j in range(self.dim[1]) for i in range(self.dim[0])]

    def sample(self, next_state: ObjectState, action: Action):
        """Returns observation"""
        return Observation(next_state.objid, next_state.pose)

    def argmax(self, next_state: ObjectState, action: Action):
        # Obtain observation according to distribution.
        return Observation(next_state.objid, next_state.pose)