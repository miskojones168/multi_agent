# dont need OOPOMDP maybe.
import pomdp_py
# import pomdp_py.framework

class Observation(pomdp_py.Observation):
    '''Define Observation class
        self.objid: Unique identifier for the observed object. (type str)
        self.pose: Pose of observed object. (type tuple)
    '''
    """The xy pose of the object is observed; or NULL if not observed"""
    
    NULL = None
    
    def __init__(self, objid, pose):
        self.objid = objid
        if type(pose) == tuple and len(pose) == 2 or pose == Observation.NULL:
            self.pose = pose
        else:
            raise ValueError("Invalid observation %s for object" % (str(pose), objid))

    def __hash__(self):
        # create hash for when observation is hashed
        return hash((self.objid, self.pose))
    
    def __eq__(self, other):
        # create equality operator
        if not isinstance(other, Observation):
            return False
        else:
            return self.objid == other.objid and self.pose == other.pose
        
    def __str__(self):
        return (f'{self.objid}-{self.pose}')

    def __repr__(self):
        return (f'Observation({self.objid}-{self.pose})')