# multi-agent-reinforcement-learning-using-group-decision-making-method

this code is a method using GMD in MARL  
including four part  
#### 1. GDM  
include GDM.py Model.py  
This part use Group Decision Making method to simulate multi-agent decision making, in order to make use the experiement of human society decision making to build the agent system.  

#### 2. Maze  
include coop_env.py maze.py maze_env2.py  
the environment maze_env2 is built by TKinter. we recode this maze from morvan. 
in our method, we require a class of maze and in this class we require these four API for each environment:  
* _build_env(self)  
This function is used to build the environment at the beginning of experiment.
* reset(self)  
This function is used to reset the environment at any step of experiment.
* step(self, action)  
This function is used to run the environment into next step using the action inputed.
* render(self)  
When a environment has GUI,this function is used to refresh the GUI without any other operation.  
  
We are now having three environment:
* maze  
This is a grid map without GUI. It's a simple example to debug the method.  
This map can further develop to a complex grid map.
* maze2  
This is a grid map with GUI. It's a simple example to watch the effect of mathod.  
This map can also further develop to a complex grid map.
* Magent  
This is a complex environment in which we can test our mathod totally.  
However, there's someting wrong in our code that the speed of environment is so low. Maybe we need to use tentorflow-GPU to solve this problem.

#### 3. DRL  
include test.py train.py RL_brain.py  
we recode the DQN from morvan.  
We use tensorflow to build the reinforcement learning model.
#### 4. API  
include API.py  
If you want to run this code, we have command line based user interaction for you.  
you should use this commend to run this code:
* python API.py --[additional parameter] ......  

Our additional parameter include:  
* parameters about framwork:
  * --model_exist
  * --MAgent
  * --gdm
* parameters about environment:
  * --max_episode
  * --n_features
  * --max_step
  * --num_agent
  * --map_size
* parameters about method:
  * --max_coop
  * --max_discuss
  * --cll_ba
  * --e_greedy_add
  * --memory_size
  * --lr
  * --gamma
  * --e_greedy
  * --batch_size
* other parameters:
  * --save_path
  * --model_name
  * --replace_target_iter
  * --output_graph
  * --num_goal
  * --num_walls
  * --alg
  * --scenario
  * --act_space
  * --benchmark_dir
  * --plots_dir