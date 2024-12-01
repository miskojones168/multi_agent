import pomdp_py

class ObjectState(pomdp_py.State):
    def __init__(self, objtype, objid, pose):
        self.objid = objid
        self.objtype = objtype
        self.pose = pose

    def __hash__(self):
        return hash((self.objid, self.objtype, self.pose))
    
    def __eq__(self, other):
        if isinstance(other, ObjectState):
            return (self.objid == other.objid and 
                    self.objtype == other.objtype and 
                    self.pose == other.pose)
        return False

    def __str__(self):
        return (f'{self.objid}-{self.objtype}-{self.pose}')

    def __repr__(self):
        return (f'ObjectState({self.objid}-{self.objtype}-{self.pose})')
    

# class MosOOState(pomdp_py.OOState):
#     def __init__(self, object_states):
#         super().__init__(object_states)

#     def object_pose(self, objid):
#         return self.object_states[objid]["pose"]

#     def pose(self, objid):
#         return self.object_pose(objid)

#     @property
#     def object_poses(self):
#         return {
#             objid: self.object_states[objid]["pose"] for objid in self.object_states
#         }

#     def __str__(self):
#         return "MosOOState%s" % (str(self.object_states))

#     def __repr__(self):
#         return str(self)

# class RobotState(pomdp_py.State):
#     def __init__(self, objtype, ):
