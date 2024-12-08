"""Defines the State for the 2D Multi-Object Search domain;

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

State space:

    :math:`S_1 \\times S_2 \\times ... S_n \\times S_r`
    where :math:`S_i (1\leq i\leq n)` is the object state, with attribute
    "pose" :math:`(x,y)` and Sr is the state of the robot, with attribute
    "pose" :math:`(x,y)` and "objects_found" (set).
"""

import pomdp_py
import math

# state looks fine. need 

###### States ######
class ObstacleState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose):
        if objclass != "obstacle" and objclass != "box":
            raise ValueError(
                "Only allow object class to be either 'box' or 'obstacle'.Got %s"
                % objclass
            )
        super().__init__(objclass, {"pose": pose, "id": objid})

    def __str__(self):
        return "ObstacleState(%s,%s)" % (str(self.objclass), str(self.pose))

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def objid(self):
        return self.attributes["id"]


class BoxState(pomdp_py.ObjectState):
    def __init__(self, box_id, pose, carrier_id):
        """
        box_id: ID of the box.
        pose: Pose of the box.
        carrier_id: ID of the robot carring object. Otherwise, None type.
        """
        # carrier id of robot carrying this box
        super().__init__(
            'box',
            {
                'id': box_id,
                'pose': pose,
                'carrier_id': carrier_id,
            }
        )

    def set_box_id(self, box_id):
        self.__setitem__('id', box_id)
    
    def set_box_pose(self, pose):
        self.__setitem__('pose', pose)
    
    def set_carrier_id(self, carrier_id):
        self.__setitem__('carrier_id', carrier_id)
    
    def __str__(self):
        return "BoxState(%s,%s|%s)" % (
            str(self.objclass),
            str(self.pose),
            str(self.carrier_id),
        )

    def __repr__(self):
        return str(self)
    
    @property
    def id(self):
        return self.attributes["id"]
    
    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def carrier_id(self):
        return self.attributes["carrier_id"]

class RobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, box_id):
        """
        robot_id: ID of the robot.
        pose: Pose of the robot.
        box_id: ID of the box being carried. Otherwise, None type.
        """
        super().__init__(
            "robot",
            {
                "id": robot_id,
                "pose": pose,  # x,y,th
                "load": box_id,
            }
        )

    def __str__(self):
        return "RobotState(%s,%s|%s)" % (
            str(self.objclass),
            str(self.pose),
            str(self.load),
        )

    def __repr__(self):
        return str(self)
    
    # def set_robot_id(self, robot_id):
    #     self.__setitem__('id', robot_id)
    
    def set_robot_pose(self, pose):
        self.__setitem__('pose', pose)
    
    def set_load_id(self, box_id):
        self.__setitem__('load', box_id)

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def id(self):
        return self.attributes["id"]

    @property
    def load(self):
        return self.attributes["load"]
    
class StationState(pomdp_py.ObjectState):
    def __init__(self, id, pose):
        super().__init__(
            'station',
            {
                'id': id,
                'pose': pose,
            }
        )
    
    @property
    def pose(self):
        return self.attributes["pose"]
    
    def __str__(self):
        return "StationState(%s,%s)" % (
            str(self.objclass),
            str(self.pose),
        )

    def __repr__(self):
        return str(self)


class MosOOState(pomdp_py.OOState):
    '''
        Represents multiple objects (boxes, robots, obstacles) as a single class
    '''
    def __init__(self, object_states):
        '''
            object_states: type dict()
            object_states: It holds multiple objects by ID (key) State (Value) pair.
            So far, implemented states as BoxState, RobotState, ObstacleState, StationState
        '''
        super().__init__(object_states)

    def get_object_states(self):
        return self.object_states

    def object_pose(self, objid):
        # print(self.object_states[objid], objid)
        return self.object_states[objid]["pose"]

    def pose(self, objid):
        return self.object_pose(objid)
    
    def state(self, objid):
        '''Return State associated with \'objid\'.'''
        return self.object_states[objid]
    
    def states(self):
        '''Return dict() of ids (key) states (value) pair.'''
        return self.object_states

    def box_states(self):
        '''Return all (ID, BoxState) pairs in a list().'''
        boxes = []
        for obj_id in self.states():
            if isinstance(self.state(obj_id), BoxState):
                boxes.append((obj_id, self.state(obj_id)))
        return boxes
    
    def robot_states(self):
        '''Return all (ID, RobotState) pairs in a list().'''
        robots = []
        for obj_id in self.states():
            if isinstance(self.state(obj_id), RobotState):
                robots.append((obj_id, self.state(obj_id)))
        return robots
    
    def station_states(self):
        '''Return all (ID, StationState) pairs in a list().'''
        st = []
        for obj_id in self.states():
            if isinstance(self.state(obj_id), StationState):
                st.append((obj_id, self.state(obj_id)))
        return st
    
    def obstacle_states(self):
        '''Return all (ID, StationState) pairs in a list().'''
        obs = []
        for obj_id in self.states():
            if isinstance(self.state(obj_id), ObstacleState):
                obs.append(self.state(obj_id))
        return obs
    
    def remove_state(self, objid):
        self.object_states.pop(objid)

    @property
    def object_poses(self):
        return {
            objid: self.object_states[objid]["pose"] for objid in self.object_states
        }

    def __str__(self):
        return "MosOOState%s" % (str(self.object_states))

    def __repr__(self):
        return str(self)
