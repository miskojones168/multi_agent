"""Policy model for 2D Multi-Object Search domain.
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
import domain.action as a
import domain.state as s

# use transition model to decide wheter to pick or place
class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, obstacles, grid_map=None):
        """ Assume policy only knows the state of its respective agent 
            During agent definition, make it so that each policy that the ID of its respective agent
        """
        self.robot_id = robot_id
        self._grid_map = grid_map
        self.obstacles = obstacles

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError
    
    def at_box_location(self, state: s.MosOOState):
        this_rob_pose = state.object_pose(self.robot_id)[:2]
        box_poses = []
        for obj in state.object_states.values():
            if isinstance(obj, s.BoxState):
                box_poses.append(obj.pose)
        
        if this_rob_pose in box_poses:
            return True
        return False

    def get_all_actions(self, state=None, history=None):

        # check if agent is holding a box
        holding = False if type(state.object_states[self.robot_id].load) == type(None) else True
        # check if at reward state
        at_reward = self.at_box_location(state) # true or false

        ret = a.ALL_MOTION_ACTIONS.copy()


        for obstacle in self.obstacles:
            next, pos = self.next_to(state.object_states[self.robot_id].pose, obstacle.pose)

            if next:
                try:
                    if pos[1] == -1:
                        ret.remove(a.MoveNorth)
                    elif pos[1] == 1:
                        ret.remove(a.MoveSouth)
                    elif pos[0] == -1:
                        ret.remove(a.MoveWest)
                    elif pos[0] == 1:
                        ret.remove(a.MoveEast)
                except:
                    pass

        robot_pose = state.pose(self.robot_id)[:2]
        station_states = [station[1].pose for station in state.station_states()]
        # can only pick boxes that are not being carried
        box_positions = [box[1].pose for box in state.box_states() if type(box[1].carrier_id) == type(None)]

        try:
            if robot_pose[0] == 5:
                ret.remove(a.MoveEast)
            if robot_pose[0] == 0:
                ret.remove(a.MoveWest)        
        except:
            pass
        try:
            if robot_pose[1] == 5:
                ret.remove(a.MoveSouth)
            if robot_pose[1] == 0:
                ret.remove(a.MoveNorth)
        except:
            pass

        if holding and robot_pose in station_states:
            ret  += [a.Place]
        if robot_pose in box_positions and not holding and not (robot_pose in station_states):
            # NOTE: pick should only happen if agent is at reward state
            ret += [a.Pick]
        if len(ret) == 0: # rare case the agent is trapped by 4 obstacles
            return [a.MoveStay]
        return ret

    def next_to(self, A,B):  
        if A[0] == B[0] and A[1] - B[1] == 1:
            return True, (0,-1)
        if A[1] == B[1] and A[0] - B[0] == 1:
            return True, (-1,0)
        if A[0] == B[0] and A[1] - B[1] == -1:
            return True, (0,1)
        if A[1] == B[1] and A[0] - B[0] == -1:
            return True, (1,0)
        
        return False, (0,0)

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
