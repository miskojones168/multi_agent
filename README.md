### Multi-agent POMDPs for warehouse robots

## Dependencies
pomdp_py framework:
```
pip install pomdp-py
```

## How to run
Executing problem.py should run the simulation.
```
python problem.py
```
A window should pop-up with the simulation running

## POMDP formulation
The POMDP components are implemented in:

```domain```: holds the representation of the State, Action, and Observation.\
```agent```: implementation of the Agent and Belief representation.\
```models```: implementation of the Reward, Transition, Observation, and Policy models.\
```env```: implements the environment that holds true state of the world.

### Miscellaneous
```example_world.py```: how functions for randomly generating a 2D gridworld.
