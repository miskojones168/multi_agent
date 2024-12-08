"""Reward model for 2D Multi-object Search domain"""

import pomdp_py
import domain.action as a
import domain.state as s

class MosRewardModel(pomdp_py.RewardModel):
    def __init__(self, target_objects, big=1000, small=1, robot_id=None):
        """
        robot_id (int): This model is the reward for one agent (i.e. robot),
                        If None, then this model could be for the environment.
        target_objects (set): a set of objids for target objects.
        """
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self._target_objects = target_objects

    def probability(
        self, reward, state, action, next_state, normalized=False, **kwargs
    ):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state, robot_id=robot_id)

    def argmax(self, state, action, next_state, normalized=False, robot_id=None):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state, robot_id=robot_id)


class GoalRewardModel(MosRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """

    def _reward_func(self, state, action, next_state, robot_id=None):
        # should gain reward for picking and placing a box
        if robot_id is None:
            assert (
                self._robot_id is not None
            ), "Reward must be computed with respect to one robot."
            robot_id = self._robot_id

        reward = -5
        if state.pose(robot_id)[:2] == next_state.pose(robot_id)[:2] and action.name != 'pick' and action.name != 'place':
            reward -= 5

        objs_dict = state.object_states
        avail_boxes_pose = [] # list of BoxState
        for obj_id in objs_dict:
            obj = objs_dict[obj_id]
            if isinstance(obj, s.BoxState) and type(obj.carrier_id) == type(None):
                avail_boxes_pose.append(obj.pose)

        next_robot_pose = next_state.pose(robot_id)[:2] # use robot_id explicitly passed to _reward_func
        robot_pose = state.pose(robot_id)[:2]
        goal_poses = next_state.station_states()
        if next_robot_pose in avail_boxes_pose and action.name == 'pick':
            next_state_reward = 10
        else: 
           next_state_reward = 0

        station_states = [station[1].pose for station in state.station_states()]


        if robot_pose in station_states and action.name == 'place':
            next_state_reward += 1000
        elif action.name == 'place':
            next_state_reward = 0
            
        reward += next_state_reward

        return reward
