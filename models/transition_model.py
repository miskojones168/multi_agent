"""Defines the TransitionModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

Transition: deterministic
"""

import pomdp_py
import copy
import math
import domain.state as s
import domain.observation as o
import domain.action as a

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

####### Transition Model #######
class MosTransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model; The transition model supports the
    multi-robot case, where each robot is equipped with a sensor; The
    multi-robot transition model should be used by the Environment, but
    not necessarily by each robot for planning.
    """

    def __init__(self, dim, sensors, object_ids, epsilon=1e-9):
        """
        sensors (dict): robot_id -> Sensor
        for_env (bool): True if this is a robot transition model used by the
             Environment.  see RobotTransitionModel for details.
        """
        self._sensors = sensors
        # create individual transition model for each object and robot
        transition_models = {
            objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
            for objid in object_ids
            if objid not in sensors
        }
        for robot_id in sensors:
            transition_models[robot_id] = RobotTransitionModel(
                sensors[robot_id], dim, epsilon=epsilon
            )
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs) # this samples next_state for each object given action,state
        # state after applying robot and static transitions
        tmp_new_state = s.MosOOState(oostate.object_states)

        # update box state here. Up to here the robot has picked or dropped the box, but the box state has not been updated
        if isinstance(action, a.PickAction):
            # assign box to robot carrying it
            for rid, r in tmp_new_state.robot_states():
                for bid, b in tmp_new_state.box_states():
                    if r.load == bid: # if robot load is this box_id
                        b.set_carrier_id(rid)

        elif isinstance(action, a.PlaceAction):

            # in reward model check if boxes are in stations and if action is place then
            boxes = tmp_new_state.box_states()
            for bid, b in boxes:
                b.set_carrier_id(None)
        
        else: # any move action
            # assign box to robot carrying it
            for rid, r in tmp_new_state.robot_states():
                if type(r.load) != type(None):
                    tmp_new_state.state(r.load).set_box_pose((r.pose[0], r.pose[1]))
                

        return tmp_new_state

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs) # this samples using argmax implemented in each transition model
        tmp_new_state = s.MosOOState(oostate.object_states)
        # update box state here. Up to here the robot has picked or dropped the box, but the box state has not been updated
        if isinstance(action, a.PickAction):
            # assign box to robot carrying it
            for rid, r in tmp_new_state.robot_states():
                for bid, b in tmp_new_state.box_states():
                    if r.load == bid: # if robot load is this box_id
                        b.set_carrier_id(rid)
        
        if isinstance(action, a.PlaceAction):

            # in reward model check if boxes are in stations and if action is place then
            boxes = tmp_new_state.box_states()
            for bid, b in boxes:
                b.set_carrier_id(None)

        return tmp_new_state


class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static. Just copies states no transitions performed"""

    # TODO: take 

    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state["id"]]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])


class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, sensor, dim, epsilon=1e-9):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self._sensor = sensor
        self._robot_id = sensor.robot_id
        self._dim = dim
        self._epsilon = epsilon

    @classmethod
    def if_move_by(cls, robot_id, state, action, dim, check_collision=True):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        if not isinstance(action, a.MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        robot_pose = state.pose(robot_id)
        rx, ry, rth = robot_pose
        if action.scheme == a.MotionAction.SCHEME_XYTH:
            dx, dy, th = action.motion
            rx += dx
            ry += dy
            rth = th
        elif action.scheme == a.MotionAction.SCHEME_VW:
            # odometry motion model
            forward, angle = action.motion
            rth += angle  # angle (radian)
            rx = int(round(rx + forward * math.cos(rth)))
            ry = int(round(ry + forward * math.sin(rth)))
            rth = rth % (2 * math.pi)

        if valid_pose(
            (rx, ry, rth),
            dim[0],
            dim[1],
            state=state,
            check_collision=check_collision,
            pose_objid=robot_id,
        ):
            return (rx, ry, rth)
        else:
            return robot_pose  # no change because change results in invalid pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """
        Returns the most likely next robot_state. It also changes the state of world based on the action.
        This should not change the real state of the world. 'state' argument should be a deepcopy

        'state' is sampled from agent.belief at the start of planning, so it should not affect real State

        """
        if isinstance(state, s.RobotState): # always false. stil here just in case
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]

        next_robot_state = copy.deepcopy(robot_state)
        if isinstance(action, a.MotionAction):
            # motion action
            next_robot_state["pose"] = RobotTransitionModel.if_move_by(
                self._robot_id, state, action, self._dim
            )

        elif isinstance(action, a.PickAction): # TODO: add station state to drop reward
            boxes = state.box_states()
            for box_id, b in boxes:
                # if any box pose equal next robot state pick box
                pose2d = robot_state.pose[:2]
                if b.pose == pose2d:
                    # NOTE: this assignment changes top-level MosOOState
                    next_robot_state.set_load_id(box_id)

        elif isinstance(action, a.PlaceAction):

            next_robot_state.set_load_id(None)
        
        return next_robot_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)


# Utility functions
def valid_pose(pose, width, length, state=None, check_collision=True, pose_objid=None):
    """
    Returns True if the given `pose` (x,y) is a valid pose;
    If `check_collision` is True, then the pose is only valid
    if it is not overlapping with any object pose in the environment state.
    """
    x, y = pose[:2]

    # Check collision with obstacles
    if check_collision and state is not None:
        object_poses = state.object_poses
        for objid in object_poses:
            if state.object_states[objid].objclass.startswith("obstacle"):
                if objid == pose_objid:
                    continue
                if (x, y) == object_poses[objid]:
                    return False
    return in_boundary(pose, width, length)


def in_boundary(pose, width, length):
    # Check if in boundary
    x, y = pose[:2]
    if x >= 0 and x < width:
        if y >= 0 and y < length:
            if len(pose) == 3:
                th = pose[2]  # radian
                if th < 0 or th > 2 * math.pi:
                    return False
            return True
    return False
