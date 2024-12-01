import pomdp_py

class Action(pomdp_py.Action):
    '''
    Base Action class
    '''
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name

MOTION_SCHEME = "xy"  # can be either xy or vw
STEP_SIZE = 1

class MoveAction(Action):
    # scheme 3 (vx,vy)
    SCHEME_XY = "xy"
    EAST2D = (STEP_SIZE, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST2D = (-STEP_SIZE, 0)
    NORTH2D = (0, -STEP_SIZE)
    SOUTH2D = (0, STEP_SIZE)
    STAY2D = (0, 0)

    SCHEMES = {"xyth", "xy", "vw"}

    def __init__(self, motion, scheme=MOTION_SCHEME, distance_cost=1, motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        if scheme not in MoveAction.SCHEMES:
            raise ValueError("Invalid motion scheme %s" % scheme)

        if scheme == MoveAction.SCHEME_XY:
            if motion not in {
                MoveAction.EAST2D,
                MoveAction.WEST2D,
                MoveAction.NORTH2D,
                MoveAction.SOUTH2D,
                MoveAction.STAY2D,
            }:
                raise ValueError("Invalid move motion %s" % str(motion))

        self.motion = motion
        self.scheme = scheme
        self.distance_cost = distance_cost
        if motion_name is None:
            motion_name = str(motion)
        super().__init__("move-%s-%s" % (scheme, motion_name))
        # super().__init__(name) # this makes it so that methods in Action that rely on name are properly initialized

MoveEast2D = MoveAction(
    MoveAction.EAST2D, scheme=MoveAction.SCHEME_XY, motion_name="East2D"
)
MoveWest2D = MoveAction(
    MoveAction.WEST2D, scheme=MoveAction.SCHEME_XY, motion_name="West2D"
)
MoveNorth2D = MoveAction(
    MoveAction.NORTH2D, scheme=MoveAction.SCHEME_XY, motion_name="North2D"
)
MoveSouth2D = MoveAction(
    MoveAction.SOUTH2D, scheme=MoveAction.SCHEME_XY, motion_name="South2D"
)
MoveStay2D = MoveAction(
    MoveAction.STAY2D, scheme=MoveAction.SCHEME_XY, motion_name="Stay2D"
)

ALL_ACTIONS = [MoveEast2D, MoveWest2D, MoveSouth2D, MoveNorth2D, MoveStay2D]
ALL_ACTIONS_NO_STAY = [MoveEast2D, MoveWest2D, MoveSouth2D, MoveNorth2D]