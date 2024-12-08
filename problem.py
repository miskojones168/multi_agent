'''
    Code adapted from Multi-Object Search Example from pomdp_py framework
    https://h2r.github.io/pomdp-py/html/
'''

import pomdp_py
import argparse
import time
import random
import copy

import env.env as e
import env.visual as vis
import agent.agent as ag
import example_worlds as exw
import domain.observation as o
import models.components.grid_map as gmap
import domain.state as s

import time
import numpy as np

from tabulate import tabulate

random.seed(42)

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

class MosOOPOMDP(pomdp_py.OOPOMDP):

    '''
        Defines POMDP problem class
    '''

    def __init__(
        self,
        robot_ids,
        env=None,
        grid_map=None,
        sensors=None,
        sigma=0.01,
        epsilon=1,
        belief_rep="histogram",
        prior={},
        num_particles=100,
        agent_has_map=False,
    ):

        self.robot_ids = robot_ids
        if env is None:
            assert grid_map is not None and sensors is not None, (
                "Since env is not provided, you must provide string descriptions"
                "of the world and sensors."
            )
            
            worldstr = e.equip_sensors(grid_map, sensors)
            dim, robots, objects, obstacles, sensors = e.interpret(worldstr)
            # print(robots)
            # exit()
            init_state = s.MosOOState({**objects, **robots})
            env = e.MosEnvironment(dim, init_state, sensors, obstacles=obstacles)

        # construct prior
        if type(prior) == str:
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                prior = {}
                for objid in env.target_objects:
                    groundtruth_pose = env.state.pose(objid)
                    prior[objid] = {groundtruth_pose: 1.0}

        # Potential extension: a multi-agent POMDP. For now, the environment
        # can keep track of the states of multiple agents, but a POMDP is still
        # only defined over a single agent. Perhaps, MultiAgent is just a kind
        # of Agent, which will make the implementation of multi-agent POMDP cleaner.
        grid_map = (
            gmap.GridMap(
                env.width,
                env.length,
                {objid: env.state.pose(objid) for objid in env.obstacles},
            )
            if agent_has_map
            else None
        )
        self.agents = {}


        for id in robot_ids:
            robot_id = id if type(id) == int else e.interpret_robot_id(id)
            agent = ag.MosAgent(
            robot_id,
            env.state.object_states[robot_id],
            env.target_objects,
            (env.width, env.length),
            env.sensors[robot_id],
            env.state.obstacle_states(),
            states=env.state,
            sigma=sigma,
            epsilon=epsilon,
            belief_rep=belief_rep,
            prior=prior,
            num_particles=num_particles,
            grid_map=grid_map,
            )
            self.agents[id] = agent

        super().__init__(
            self.agents[robot_ids[0]],
            env,
            name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)),
        )
        

    def set_active_agent(self, robot_id):
        if robot_id not in self.agents:
            raise ValueError(f"Robot ID {robot_id} is not valid.")
        self.robot_id = robot_id
        self.agent = self.agents[robot_id]


### Belief Update ###
def belief_update(agent, real_action, real_observation, next_robot_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
        for objid in agent.cur_belief.object_beliefs:
            belief_obj = agent.cur_belief.object_belief(objid)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    # discrete state filter update
                    new_belief = pomdp_py.update_histogram_belief(
                        belief_obj,
                        real_action,
                        real_observation.for_obj(objid),
                        agent.observation_model[objid],
                        agent.transition_model[objid],
                        # The agent knows the objects are static.
                        static_transition=objid != agent.robot_id,
                        oargs={"next_robot_state": next_robot_state},
                    )
                    # print(objid, len(new_belief))
                    # exit()
            else:
                raise ValueError(
                    "Unexpected program state."
                    "Are you using the appropriate belief representation?"
                )
            agent.cur_belief.set_object_belief(objid, new_belief)


### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def solve(
    problem,
    max_depth=10,  # planning horizon
    discount_factor=0.99,
    planning_time=1.0,  # amount of time (s) to plan each step
    exploration_const=1000,  # exploration constant
    visualize=True,
    max_time=120,  # maximum amount of time allowed to solve the problem
    max_steps=500,
):  # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.

    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    random_objid = random.sample(sorted(problem.env.target_objects), 1)[0]
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    planners = {}
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        for id in problem.robot_ids:
            print("ID: ", id)

            planner = pomdp_py.POUCT(
                max_depth=max_depth,
                discount_factor=discount_factor,
                # num_sims=3000,
                planning_time=planning_time,
                exploration_const=exploration_const,
                rollout_policy=problem.agents[id].policy_model,
            )  # Random by default

            planners[id] = planner
    else:
        raise ValueError(
            "Unsupported object belief type %s" % str(type(random_object_belief))
        )

    robot_id = problem.agent.robot_id
    if visualize:
        viz = vis.MosViz(
            problem.env, controllable=False
        )  # controllable=False means no keyboard control.
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id, None, None, None, problem.agent.cur_belief)
        viz.on_render()

    _time_used = 0
    _total_reward = 0  
    
    boxes_belief = []
    for i in range(max_steps):
        taken_actions = {}
        for id in problem.robot_ids:

            planner = planners[id]
            problem.set_active_agent(id)
            robot_id = problem.agent.robot_id
            
            # --------------
            boxes = problem.env.state.box_states()
            id_b = problem.agent.belief.object_beliefs
            bbelief_t = []
            for id, _ in boxes:
                dist = list(id_b[id].histogram.values())
                bbelief_t.append(dist)

            boxes_belief.append(bbelief_t)
            # --------------

            print(f'{bcolors.WARNING} AGENT: {robot_id} {bcolors.ENDC}')
            print( 'AGENT OM location:', problem.agent.observation_model)
            print("==== Step %d ====" % (i + 1))

            env_boxes = problem.env.state.box_states()

            print('Boxes before transition\n',tabulate(env_boxes, ['IDs', 'Box State']))
            # Plan action //////////////////////////////////////////////
            _start = time.time()
            real_action = planner.plan(problem.agent)
            _time_used += time.time() - _start
            if _time_used > max_time:
                break  # no more time to update.
            # /////////////////////////////////////////////////////////
            print(f'{bcolors.WARNING} Action taken: {real_action} {bcolors.ENDC}')
            taken_actions[robot_id] =  real_action
        # =================================================================================

        # Execute action
        reward = problem.env.state_transition(
            taken_actions, execute=True, robot_ids=list(taken_actions.keys()), agents=problem.agents
        )

        print('After transition \n')
        print(problem.env.state.robot_states())
        print(tabulate(problem.env.state.box_states(), ['IDs', 'Box State']))

        # ----------------------------------------
        for id in problem.robot_ids:
            planner = planners[id]
            problem.set_active_agent(id)
            robot_id = problem.agent.robot_id

            # Receive observation
            _start = time.time()
            real_observation = problem.env.provide_observation(
                problem.agent.observation_model, real_action
            )

            # Updates
            problem.agent.clear_history()  # truncate history
            problem.agent.update_history(real_action, real_observation)

            belief_update(
                problem.agent,
                real_action,
                real_observation,
                problem.env.state.object_states[robot_id],
                planner,
            )
            _time_used += time.time() - _start

        # Info and render
        _total_reward += reward   

        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
        
        print("==== Step %d ==== END" % (i + 1))
        
        if visualize:

            for id in problem.robot_ids:
                planner = planners[id]
                problem.set_active_agent(id)
                robot_id = problem.agent.robot_id

                robot_pose = problem.env.state.object_states[robot_id].pose
                viz_observation = o.MosOOObservation({})
                viz_observation = problem.env.sensors[robot_id].observe(
                    robot_pose, problem.env.state
                )
                
                viz.update(
                    robot_id,
                    real_action,
                    real_observation,
                    viz_observation,
                    problem.agent.cur_belief,
                )
            
            viz._env = copy.deepcopy(problem.env)
            viz._img = viz._make_gridworld_image()

            viz.on_loop()
            viz.on_render(iii=i)
            # =================================================================================
        
        # Termination check
        boxes = problem.env.state.box_states()
        stations = problem.env.state.station_states()
        in_st = 0
        for bid, b in boxes:
            for stid, st in stations:
                if b.pose == st.pose:
                    in_st += 1
        if (in_st == len(boxes)):
            print("Done!")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break

    # save belief
    boxes_belief = np.array(boxes_belief)
    np.save('beliefs.npy', boxes_belief)


# Test
def unittest():
    # random world
    # grid_map, robot_char = exw.random_world(6, 6, 3, 5)
    grid_map, robot_char, r2 = exw.world5
    laserstr = e.make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = e.make_proximity_sensor(20, False)

    sensors = {}

    robot_chars = [robot_char]
    print(grid_map)
    problem = MosOOPOMDP(
        robot_chars,  # r is the robot character
        sigma=0.05,  # observation model parameter
        epsilon=0.95,  # observation model parameter
        grid_map=grid_map,
        sensors={robot_char: proxstr, r2: proxstr},
        prior="uniform",
        agent_has_map=True,
    )
    solve(
        problem,
        max_depth=40,
        discount_factor=0.9,
        planning_time=1,
        exploration_const=4000,
        visualize=True,
        max_time=120,
        max_steps=500,
    )


if __name__ == "__main__":
    unittest()
