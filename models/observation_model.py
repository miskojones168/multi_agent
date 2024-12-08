"""Defines the ObservationModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Observation: {objid : pose(x,y) or NULL}. The sensor model could vary;
             it could be a fan-shaped model as the original paper, or
             it could be something else. But the resulting observation
             should be a map from object id to observed pose or NULL (not observed).

Observation Model

  The agent can observe its own state, as well as object poses
  that are within its sensor range. We only need to model object
  observation.

"""

import pomdp_py
import math
import random
import numpy as np

import domain.state as s
import domain.action as a
import domain.observation as o

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#### Observation Models ####
class MosObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""

    def __init__(self, dim, sensor, object_ids, sigma=0.01, epsilon=1):
        self.sigma = sigma
        self.epsilon = epsilon
        observation_models = {
            objid: ObjectObservationModel(
                objid, sensor, dim, sigma=sigma, epsilon=epsilon
            )
            for objid in object_ids
        }
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        ''' Get observations for any action '''
        factored_observations = super().sample(next_state, action, argmax=argmax)
        return o.MosOOObservation.merge(factored_observations, next_state) # add observations to set of all obsrvations

# need individual observation model for each target. in this case boxes
class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, objid, sensor, dim, sigma=0, epsilon=1):
        """
        sigma and epsilon are parameters of the observation model (see paper),
        dim (tuple): a tuple (width, length) for the dimension of the world"""
        self._objid = objid
        self._sensor = sensor
        self._dim = dim
        self.sigma = sigma
        self.epsilon = epsilon

    def _compute_params(self, object_in_sensing_region):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = self.epsilon
            beta = (1.0 - self.epsilon) / 2.0
            gamma = (1.0 - self.epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - self.epsilon) / 2.0
            beta = (1.0 - self.epsilon) / 2.0
            gamma = self.epsilon
        return alpha, beta, gamma

    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """

        if observation.objid != self._objid:
            raise ValueError("The observation is not about the same object")

        # The (funny) business of allowing histogram belief update using O(oi|si',sr',a).
        next_robot_state = kwargs.get("next_robot_state", None)

        if next_robot_state is not None:
            robot_pose = next_robot_state.pose

            if isinstance(next_state, s.BoxState):
                assert (
                    next_state["id"] == self._objid
                ), "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            elif isinstance(next_state, s.StationState):
                assert (
                    next_state["id"] == self._objid
                ), "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(self._objid)
        else:
            robot_pose = next_state.pose(self._sensor.robot_id)
            object_pose = next_state.pose(self._objid)

        # Compute the probability
        zi = observation.pose
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose)
        )

        # Requires Python >= 3.6
        prob = 0.0
        # Event A:
        # object in sensing region and observation comes from object i
        if zi == o.ObjectObservation.NULL:
            # Even though event A occurred, the observation is NULL.
            # This has 0.0 probability.
            prob += 0.0 * alpha
        else:
            gaussian = pomdp_py.Gaussian(
                list(object_pose), [[self.sigma**2, 0], [0, self.sigma**2]]
            )
            prob += gaussian[zi] * alpha

        # Event B
        prob += (1.0 / self._sensor.sensing_region_size) * beta

        # Event C
        pr_c = 1.0 if zi == o.ObjectObservation.NULL else 0.0  # indicator zi == NULL
        prob += pr_c * gamma
        return prob

    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        robot_pose = next_state.pose(self._sensor.robot_id)
        object_pose = next_state.pose(self._objid)

        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose)
        )

        # Requires Python >= 3.6
        event_occured = random.choices(
            ["A", "B", "C"], weights=[alpha, beta, gamma], k=1
        )[0]
        zi = self._sample_zi(event_occured, next_state)

        return o.ObjectObservation(self._objid, zi)

    def argmax(self, next_state, action, **kwargs):
        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose) # where do they come from?
        )

        event_probs = {"A": alpha, "B": beta, "C": gamma}
        event_occured = max(event_probs, key=lambda e: event_probs[e])
        zi = self._sample_zi(event_occured, next_state, argmax=True)
        return o.ObjectObservation(self._objid, zi)

    def _sample_zi(self, event, next_state, argmax=False):
        if event == "A":
            object_true_pose = next_state.object_pose(self._objid)
            gaussian = pomdp_py.Gaussian(
                list(object_true_pose), [[self.sigma**2, 0], [0, self.sigma**2]]
            )
            if not argmax:
                zi = gaussian.random()
            else:
                zi = gaussian.mpe()
            zi = (int(round(zi[0])), int(round(zi[1])))

        elif event == "B":
            width, height = self._dim
            zi = (
                random.randint(0, width),  # x axis
                random.randint(0, height),
            )  # y axis
        else:  # event == C
            zi = o.ObjectObservation.NULL
        return zi

if __name__ == "__main__":
    pass
