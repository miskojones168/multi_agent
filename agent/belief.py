# Defines the belief distribution and update for the 2D Multi-Object Search domain;
#
# The belief distribution is represented as a Histogram (or Tabular representation).
# Since the observation only contains mapping from object id to their location,
# the belief update has no leverage on the shape of the sensing region; this is
# makes the belief update algorithm more general to any sensing region but then
# requires updating the belief by iterating over the state space in a nested
# loop. The alternative is to use particle representation but also object-oriented.
# We try both here.
import pomdp_py
import random
import copy
import domain.state as s


class MosOOBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""

    def __init__(self, robot_id, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.robot_id = robot_id
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return s.MosOOState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return s.MosOOState(pomdp_py.OOBelief.random(self, **kwargs).object_states)


def initialize_belief(
    dim,
    robot_id,
    object_ids,
    states,
    prior={},
    representation="histogram",
    robot_orientations={},
    num_particles=100,
):
    """
    Returns a GenerativeDistribution that is the belief representation for
    the multi-object search problem.

    Args:
        dim (tuple): a tuple (width, length) of the search space gridworld.
        robot_id (int): robot id that this belief is initialized for.
        object_ids (dict): a set of object ids that we want to model the belief distribution
                          over; They are `assumed` to be the target objects, not obstacles,
                          because the robot doesn't really care about obstacle locations and
                          modeling them just adds computation cost.
        prior (dict): A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}. If used, then
                      all locations not included in the prior will be treated to have 0 probability.
                      If unspecified for an object, then the belief over that object is assumed
                      to be a uniform distribution.
        robot_orientations (dict): Mapping from robot id to their initial orientation (radian).
                                   Assumed to be 0 if robot id not in this dictionary.
        num_particles (int): Maximum number of particles used to represent the belief

    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if representation == "histogram":
        return _initialize_histogram_belief(
            dim, robot_id, object_ids, prior, robot_orientations, states=states
        )
    # elif representation == "particles":
    #     return _initialize_particles_belief(
    #         dim, robot_id, object_ids, robot_orientations, num_particles=num_particles
    #     )
    else:
        raise ValueError("Unsupported belief representation %s" % representation)


def _initialize_histogram_belief(dim, robot_id, object_ids, prior, robot_orientations, states):
    """
    Returns the belief distribution represented as a histogram

    NEED TO UPDATE THIS. this gives the state used in planning. can assume full observability for now

    TODO: Review this code and do {state: 1.0} for each object in MosOOState for full observability.

    """
    # construct uniform distribution for the belief of each box
    oo_hists = {}  # objid -> Histogram() mapping
    width, length = dim
    # print(object_ids, states)
    # exit()
    for objid in object_ids:
        if isinstance(states.state(objid), s.StationState):
            stations = states.station_states()
            for stid, st in stations:
                oo_hists[stid] = pomdp_py.Histogram({st: 1.0})
        elif isinstance(states.state(objid), s.BoxState):
            hist = {}  # possible state -> prob mapping
            total_prob = 0

            # initialize uniform belief
            for x in range(width):
                for y in range(length):
                    state = s.BoxState(objid, (x, y), None)
                    hist[state] = 1.0
                    total_prob += hist[state]

            # Normalize
            for state in hist:
                hist[state] /= total_prob

            hist_belief = pomdp_py.Histogram(hist)
            oo_hists[objid] = hist_belief

    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    oo_hists[robot_id] = pomdp_py.Histogram(
        {s.RobotState(robot_id, init_robot_pose, None): 1.0}
    )

    return MosOOBelief(robot_id, oo_hists)

if __name__ == '__main__':
    pass