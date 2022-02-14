"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np

# Gym
from gym import spaces
from gym.core import Env

# Wrappers
from robosuite.wrappers import Wrapper

# Serialization
from rlkit.core.serializable import Serializable


class GymWrapper(Wrapper, Env, Serializable):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module.

    Hack: rlkit/robosuite wrapper
    Adapted for the rlkit-relational/robosuite setup. Does not use Gym environment. Was done to minimize adaptation code on the algo side.

    Serialization: 
    A __getstate__ and __setstate__ have been added along with the Serializable class in order to pickle this class as a wrapper for robosuite. 
    The __init__ method, experienced the addition of some extra checks necessary to properly load attributes that were saved in pickle.dumps. 
    So far, we have not been able to properly save the entire robosuite environment due to an offending 'env.robots' class. So, we focus on saving 
    appropriate attributes and then reconstruct afterwards.

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, rlkit_relational=True, keys=None):
        
        # Run super method for wrapper
        super().__init__(env=env)
        
        # Create name for gym. 
        # Serialization: check if self.env.robots exists (we do not load the robots class in pickle.loads settings due to __cinit__ problems). This object will be re-initialized afterwards. 
        if hasattr(self,'robots'):
            if hasattr(self.robots[0], 'robot_model'):
                robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
            else: 
                robots = "".join([robot for robot in self.env.robots]) #self.env.robot_names                 
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        # Get keys, ie ['object-state','robot0_proprio-state']
        if keys is None:
            keys = []
            
            # Add object obs if requested
            # if self.env.use_object_obs:
            #     keys += ["object-state"]
            
            # # Add image obs if requested
            # if self.env.use_camera_obs:
            #     keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            
            # # Iterate over all robots to add to state
            # for idx in range(len(self.env.robot_names)): # for idx in range(len(self.env.robos)):
            #     keys += ["robot{}_proprio-state".format(idx)]
        
        self.keys = keys 

        # Gym specific attributes
        self.spec     = None
        self.metadata = None

        # HACK: forcing rlkit_relational settings
        self.rlkit_relational = rlkit_relational


        # RLKIT/robosuite check 
        if not self.rlkit_relational:
            # set up observation and action spaces
            obs                 = self.env.reset()                              # dictionary of observables
            self.modality_dims  = {key: obs[key].shape for key in self.keys}
            flat_ob             = self._flatten_obs(obs)    # flatten's images... double check this
            
            self.obs_dim        = flat_ob.size              # concatenantes proprio, object, and image info into one long contiguous array
            high                = np.inf * np.ones(self.obs_dim)
            low                 = -high
            
            self.observation_space = spaces.Box(low=low, high=high)
            low, high           = self.env.action_spec
            self.action_space   = spaces.Box(low=low, high=high)

        if self.rlkit_relational:
            self.env.first_reset = True             # Note: if true, picking.py:Picking.reset_internal() goes through a standard reset path. otherwise skips due to our formalism in dealing with objects in the picking environment. 
            
            # Set obs & action spaces.             
            if hasattr(self.robots[0],'robot_model'):           # Check if self.env.robots exists (we do not load the robots class in pickle.loads settings due to __cinit__ problems)
                obs                 = self.env.reset()          # dictionary of observables
                self.modality_dims  = {key: obs[key].shape for key in self.keys}
                
                # Observation Dimensions... No flattening
                self.obs_dim        = obs['observation'].size                      # dict of obs contains the following keys: 'observations', 'achieved_goal', 'desired_goal'
           
                high                = np.inf * np.ones(self.obs_dim)
                low                 = -high

                #-------------------------------------------------------------------------------------------------------------
                ## rlkit-relational interface: The ObsDictRelabelingBuffer used for the ReplayBuffer checks for the following below:
                #-------------------------------------------------------------------------------------------------------------
                # Set the observation space as a spaces.Dict. Can check rlkit_relational/FCB/FCB/envs/robotics/robot_env.py__init__
                # Spatial Information:
                self.observation_space = spaces.Dict(dict(
                    desired_goal  = spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                    achieved_goal = spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                    observation   = spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                ))       

                # Action Dimensions... 
                # Action specs set in robot environ and depend on controller (gripper + robot)
                # i.e. if 2 gripper fingers => gripper is dim(1), if OSC controller 'fixed' dim is xyz rpy
                low, high           = self.env.action_spec
                self.action_space = spaces.Box(low=low, high=high)                          
            
            else: # Serialization: could not serialize self.env.X, will re-instantiate object.
                obs = None
                self.modality_dims  = dict(zip(self.keys, [self.env.variant['object_dim'],self.env.variant['robot_dim']]))
                self.obs_dim        = self.env.variant['object_dim']+self.env.variant['robot_dim']          
                high                = np.inf * np.ones(self.obs_dim)
                low                 = -high
                self.observation_space = spaces.Dict(dict(
                    desired_goal  = spaces.Box(-np.inf, np.inf, (self.env.variant['goal_dim'],), dtype='float32'),
                    achieved_goal = spaces.Box(-np.inf, np.inf, (self.env.variant['goal_dim'],), dtype='float32'),
                    observation   = spaces.Box(-np.inf, np.inf, (self.obs_dim,),                 dtype='float32'),
                ))  
                # Action specs
                high = np.ones([self.env.variant['action_dim'],1])
                low = -high
                self.action_space = spaces.Box(low=low, high=high)                                                                    

        # Serializable Class
        self._serializable_initialized = False
        Serializable.quick_init(self, locals()) # Save this classes args/kwargs             
    

    def gym_wrapper_flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        if self.rlkit_relational:
            return ob_dict
        else:
            return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function. 
        - Normally returns flattened observation instead of normal OrderedDict.
        - For relationalRL, we do not flatten. 

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) <flattened> observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        # Step into the environment
        ob_dict, reward, done, info = self.env.step(action)

        if self.rlkit_relational:
            return ob_dict, reward, done, info
        else:
            return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal:
            desired_goal: 
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        if self.rlkit_relational:
            return self.env.compute_reward(achieved_goal, desired_goal, info)
        else:
            return self.env.reward()

# Serialization
    def __getstate__(self):
        '''
         Retrieves args/kwargs. Called on pickle.dumps.
         No need to explicitly save the environment. We will re-instantiate in Serializable.__setstate__ 
        '''

        # (1) Retrieve the classes args/kwargs
        d = Serializable.__getstate__(self)   

        # (2) Extract all kwargs manualy
        # d = dict()        
        # d['name']           = self.name 
        # d['reward_range']   = self.reward_range     
        # d['keys']           = self.keys
        # #d['env.spec']       = self.env.spec # contains .env so that part needs to be instantiated
        # d['metadata']       = self.metadata
        # d['rlkit_relational']   = self.rlkit_relational
        # d['observation_space']  = self.observation_space
        # d['action_space']       = self.action_space

        return d 
        
    
    def __setstate__(self, d):
        '''
        __setstate_ will properly extract all args/kwargs and then pass them to the environment's constructure to re-insantiate the object.
        '''

        # (1) instantiate class and copy back to self
        Serializable.__setstate__(self, d)
        
        # (2) Manually set self attributes
        # Should I try to instantiate the pikcing class here?
        # self.name           = d['name']
        # self.reward_range   = d['reward_range']
        # self.keys           = d['keys']
        # #self.env.spec       = d['env.spec']
        # self.metadata       = d['metadata']
        # self.rlkit_relational   = d['rlkit_relational']   
        # self.observation_space  = d['observation_space']  
        # self.action_space       = d['action_space']