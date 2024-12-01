from agents.agent import *

from models.observation_model import *
from models.policy_model import *
from models.transition_model import *
from models.reward_model import *

from domain.state import *
from domain.action import *
from domain.observation import *

from environment.env import *
from environment.visual import *

import random

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


'''
    TODO: 
    Single agent setup:
    * Add terminal states or make agent shift goals 
    * Update RewardModel to pick/place box setting
    * Update TransitionModel to reflect obstacles
    * Update PolicyModel if necessary
    
    Multi-Agent setup:
    * Add support for multiple agents
        * environment.py, visual.py,
    
    Alternative update Multi-Object Search demo
    to add multiple agents 

    after first reward taken it seems to take random actions. check if observation and belief affects it

'''

def gen_rand_reward(dim):
    x = random.randint(0, dim[0] - 1)
    y = random.randint(0, dim[0] - 1)
    return (x, y)

def main():

    robot_id = 0
    
    init_robot_state = ObjectState(objtype='robot', objid=robot_id, pose=(4, 4))
    reward_state = ObjectState(objtype='box', objid=0, pose=(2, 2))
    world_state = {'robot': init_robot_state, 'reward': reward_state}
    # add obstacle state later: obstacles = {ObjectState(objtype='obstacle', objid=0, pose=(1, 1))}
    dim=(5,5)
    agent = Agent(robot_id=robot_id, init_robot_state=init_robot_state, reward_state=reward_state, dim=(5, 5))
    env = Environment(init_state=world_state, dim=(5,5))

    pomdp = pomdp_py.POMDP(agent=agent, env=env, name='MY POMDP')

    vis = MosViz(env=pomdp.env)
    if vis.on_init() == False:
        raise Exception("Environment failed to initialize")
    vis.update(robot_id, None, None, None, None)
    vis.on_render()

    # online planner POUCT uses MCTS
    planner = pomdp_py.POUCT(max_depth=30, discount_factor=0.95,
                       planning_time=1., num_sims=3000, exploration_const=3000,
                       rollout_policy=pomdp.agent.policy_model)
    
    steps = 40

    for s in range(steps):
        # obtain action
        print(f'{bcolors.OKGREEN}Before planning{bcolors.ENDC}: Agent rewards: \n Agent reward pose: {pomdp.agent.reward_model.reward_pose}\n')
        action = planner.plan(pomdp.agent)
        # print(planner._agent.tree.argmax())
        # exit()

        print(f'{action} from {pomdp.env.cur_state}')
        # get next state and reward
        next_state, reward = env.state_transition(action=action)
        # update state
        pomdp.env.apply_transition(next_state) 

        # TODO: Need to update belief representation //////// check agent and belief
        observ = pomdp.agent.observation_model.sample(pomdp.env.state, action)
        print('Observation', observ, next_state)

        pomdp.agent.clear_history()
        pomdp.agent.update_history(action, observ)
        # print(pomdp.agent.history)
        pomdp.agent.update_belief(pomdp_py.Histogram({ObjectState(objtype='robot', objid=robot_id, pose=next_state.pose): 1.0}))
        planner.update(pomdp.agent, action, observ)
        # ///////////

        print('Next state to transition:', next_state, '\nCurrent state based on ENV:', pomdp.env.cur_state)
        print(f'Agent rewards: {pomdp.agent.reward_model.reward_pose}\nAgents belief: {pomdp.agent.cur_belief}\nEnv rewards: {pomdp.env.reward_model.reward_pose}')
        print(f'{bcolors.WARNING}Reward{bcolors.ENDC}: {reward}')

        # if pomdp.env.cur_state.pose in pomdp.env.reward_model.reward_pose: # update visual to reflect change in reward
        #     new_reward_pose = gen_rand_reward(dim)
        #     pomdp.agent.reward_model.reward_pose = {new_reward_pose: 100}
        #     pomdp.env.update_reward(ObjectState(objtype='box', objid=0, pose=new_reward_pose), 100)

        #### update rendering
        robot_pose = pomdp.env.get_robot_state()
        vis.update(
            robot_id,
            action,
        )
        vis.on_loop()
        vis.on_render()
        ####
        if pomdp.env.cur_state.pose in pomdp.env.reward_model.reward_pose: # update visual to reflect change in reward
            print(f'{bcolors.OKBLUE}UPDATED REWARD{bcolors.ENDC}')
            new_reward_pose = gen_rand_reward(dim)
            pomdp.agent.reward_model.reward_pose = {new_reward_pose: 10}
            pomdp.env.update_reward(ObjectState(objtype='box', objid=0, pose=new_reward_pose), 10)
            vis._img = vis._make_gridworld_image()

        print(f'End iteration {s} --------------------------\n')

if __name__ == "__main__":
    main()