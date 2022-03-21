#core
import sys
import random
import os.path

import math
import numpy as np
from collections import OrderedDict

# Utilities
import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as camera_utils

from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler, robotUniformRandomSampler, UniformWallSampler

# 01 Objects
# Import desired|all objects (and visual objects). Use * to import large number of objects. 
from robosuite.models.objects import * 

import matplotlib.cm as cm
import cv2

# After importing we can extract objects by getting the modules via dir()
mods = dir()

# Name Assumptions:
# object names: oXXXX
# visual object names: oXXXXv
# class names will follow the same name as object names and visual object names. This is a departure from robosuite which sets them as: MilkObject, MilkVisualObject, Milk, and Milkvisual for object name, visual object name, object class, and visual object class name respectively. 

# Assumes names have VisualObject TODO: this may change to o00XXv
visual_objs_in_db   = [ item for item in mods if 'VisualObject' in item]
objs_in_db          = [ item.replace('Visual','') for item in visual_objs_in_db ] #
num_objs_in_db      = len(objs_in_db)

# 02 Manipulation Environment (Parent)
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

# 03 Arena: bins_arena.xml
from robosuite.models.arenas import BinsArena

# 04 Tasks
from robosuite.models.tasks import ManipulationTask

# 05 Observables
from robosuite.utils.observables import Observable, sensor

# 06 Mujoco
import mujoco_py

# 07 Gym Spaces
from gym import spaces

# 08 Serializable to resolve pickle
from rlkit.core.serializable import Serializable

# 09 Image Processing: segmentation, depth, orientation of blob
import robosuite.utils.img_processing as img

# 10 Renderer
import pygame

# Used in Reconstructing the original object
import robosuite as suite                              # will call all __init__ inside suite loading all relevant classes
from robosuite.wrappers import GymWrapper  
from rlkit.envs.wrappers import NormalizedBoxEnv 

# Globals
object_reset_strategy_cases = ['jumbled', 'wall']# ['organized', 'jumbled', 'wall', 'random']
_reset_internal_after_picking_all_objs = True


class Picking(SingleArmEnv, Serializable):
    """
    This class corresponds to a pick and place task for a single PANDA robot arm as defined in robosuite assets and trained via Richard Li's RelationalRL. 
    Currently, the task is modelled as a bin-picking task, but maybe extended to include putwalls. 

    The class allows the user to set:

    Args:
        #--- Robots --------------------------------------------------------------------------------------------
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
            TODO: find a way to integrate Baxter into a single-arm robot.

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        #--- Control --------------------------------------------------------------------------------------------
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        #--- Arena --------------------------------------------------------------------------------------------
        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects

        bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin

        #--- MDP --------------------------------------------------------------------------------------------
        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).


        #--- Reset & Objects --------------------------------------------------------------------------------------------

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        # Objects: (i) how many in hard drive, (ii) how many to load (randomly from file), and (iii) how many to model in graph
        num_objs_in_db:     number of objects available in database/folder
        num_obs_to_load:    number of objects to insert to bin
        num_blocks:         number of objects to model with graph nodes. This directly affects the reward scale. 
        
        object_reset_strategy (str): [organized, jumbled, wall, random]. 
                                     Organized: nicely stacked side-by-side, if no more fit, on-top of each other. left-to-right.
                                     Jumbled:   just dropped in from top at center, let them fall freely.
                                     Wall:      align objects with wall. start on far wall on the left and move clock-wise.
                                     Random:    select above strategy randomly with each new reset.
        object_randomization (bool): specifies whether new random objects are selected with every reset or not. I.e. 
                                     Ie. if 100 objs available in db, but you only place 20 in bin, you can continue to choose 
                                     20 new ones with each new reset. 

        # Init structs for object names, visual object names, object_to_ids, sorted_objects, objs _in_target_bing...        
        self.object_names                                   # list of names (string) currently modeled objects
        self.visual_object_names                            # same for visual

        self.not_yet_considered_object_names                # list of names of loaded objs that are not currently being modeled
        self.not_yet_considered_visual_object_names         # same for visual

        self.objects                                        # list of modeled instantiated objects
        self.visual_objects                                

        self.not_yet_considered_objects                     # list of unmodeled instantiated objects
        self.not_yet_considered_visual_objects              # same for visual        

        self.object_to_id                                   # dict with mappings between object and id
        self.object_id_to_sensors                           # Maps object id to sensor names for that object         

        self.sorted_objects_to_model                        # closest objects to the self.goal_object based on norm2 dist
        self.object_placements                              # placements for all objects upon reset    

        self.other_objs_than_goals   
        self.objects_in_target_bin                          # list of object names (str) in target bin        

        # Goal pose for HER setting
        self.goal_object                                    # holds name, pos, quat
        self.goal_pos_error_thresh                          # set threshold to determine if current pos of object is at goal.


        #--- Cameras --------------------------------------------------------------------------------------------
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
            
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering


        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        #--- 
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

    Raises:
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,

        # Robots
        robots,
        gripper_types       = "default",                
        env_configuration   = "default", # dictionary        

       # Control
        control_freq       = 20,        
        controller_configs = None,

        # Arena
        table_full_size = (0.39, 0.49, 0.82), # these dims are table*2 -0.01 in (x,y). z would have been 0.02*2 -1 = 0.03 but it is 0.82 ??
        table_friction  = (1, 0.005, 0.0001), # (sliding, torsional, rolling) rations across surfaces. 

        # bin1_pos = (0.1, -0.25, 0.8),           # Follows xml
        # bin2_pos = (0.1, 0.28, 0.8),

        # # move bins
        # bin1_pos=(-0.1, -0.25, 0.8),  # Follows xml
        # bin2_pos=(-0.1, 0.28, 0.8),
        # bin_thickness=(0, 0, 0.02),
        
        # Observations
        use_camera_obs = True,                  # TODO: Currently these two options are setup to work in oposition it seems. Can we have both to True?
        use_object_obs = False,

        # Rewards
        reward_scale    = 1.0,
        reward_shaping  = False,
      
        horizon     = 1000,
        ignore_done = False,

        # Objects & Resets & Goals
        num_blocks              = 1,        # blocks to consider in graph. affects rewards. 
        num_objs_to_load        = 1,        # from db       

        object_reset_strategy = "jumbled",   # [organized, jumbled, wall, random]. random will randomly choose between first three options. 
        object_randomization  = True,       # Randomly select new objects from database after reset or not

        # Reset
        hard_reset            = False,       # If True, re-loads model|sim|render object w reset call. Else, only call sim.reset and reset all robosuite-internal variables        

        # Goals
        goal                    = 0,
        objects_in_target_bin   = [],

        goal_pos_error_thresh   = 0.05,     # Used to determine if the current position of the object is within a threshold of goal position

        # Camera: RGB
        camera_names            = "robot0_eye_in_hand",
        camera_image_height     = 84,
        camera_image_width      = 84,
        camera_depths           = False,

        # Gray
        use_gray_img            = False,

        # Render 
        has_renderer            = False,
        has_offscreen_renderer  = True,
        # "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand').
        render_camera           = "agentview", #TODO: may need to adjust here for better angle for our work
        use_pygame_render       = False,
        render_gpu_device_id    = 0,            # was -1 

        # Segmentations
        camera_segmentations     = "instance",

        # Meshes
        render_collision_mesh   = False,
        render_visual_mesh      = False,

        visualize_camera_obs    = False,

        # Noise
        initialization_noise    ="default",
        suite_path              = "",    

        # Variant dictionary
        variant                 = {},           # Can be filled in when the class is instantiated in drl algo side.    

        # Reset
        first_reset             = True,
        top_down_grasp          = False,
    ):
        print('Generating Picking class.\n')
        # Task settings

        # (A) Objects: 
        # 1. Set num_objs_in_db
        # 2. set num_objs_to_load, but first check this number is less than num_objs_in_db
        # 3. set num_objects to represent in graph
        # 4. generate random list of objects to load: more involved as several checks required. 

        # Init structs for object names, visual object names, object_to_ids, sorted_objects, objs _in_target_bing...        
        self.object_names           = []                        # list of names (string) currently modeled objects
        self.visual_object_names    = []                        # same for visual

        self.not_yet_considered_object_names        = []        # list of names of loaded objs that are not currently being modeled
        self.not_yet_considered_visual_object_names = []        # same for visual

        self.objects                                = []        # list of modeled instantiated objects
        self.visual_objects                         = []        

        self.not_yet_considered_objects             = []        # list of unmodeled instantiated objects
        self.not_yet_considered_visual_objects      = []        # same for visual        

        self.object_to_id            = {}                       # dict with mappings between object and id
        self.object_id_to_sensors    = {}                       # Maps object id to sensor names for that object         

        self.sorted_objects_to_model = {}                       # closest objects to the self.goal_object based on norm2 dist
        self.object_placements       = {}                       # placements for all objects upon reset    

        self.other_objs_than_goals   = []
        self.objects_in_target_bin   = objects_in_target_bin    # list of object names (str) in target bin        

        # Goal pose for HER setting
        self.goal_object            = {}                     # holds name, pos, quat
        self.goal_pos_error_thresh  = goal_pos_error_thresh  # set threshold to determine if current pos of object is at goal.

        # Fallen objects flag
        self.fallen_objs        = []
        self.fallen_objs_flag   = False

        ## Organize object information

        # No longer use code below, instead use mods = dir() in preamble. This is left for considerations.
        #------------------------------------------------------------------------------------------------------------------
        # num_objs_to_load obtained from preamble via reading dir() modules correctly instead of files as below. 
        # obj_dir = os.path.join(suite_path, './models/assets/objects')
        # self.num_objs_in_db = len([name for name in next(os.walk(obj_dir))[2] if 'visual' not in name]) # looks for files without visual        
        self.num_objs_in_db = num_objs_in_db

        if num_objs_to_load < self.num_objs_in_db:
            self.num_objs_to_load  = num_objs_to_load           # tot num of object to load 
        else:
            self.num_objs_to_load = self.num_objs_in_db                   

        # if num_blocks > self.num_objs_to_load:
        #     self.num_blocks  = self.num_objs_to_load            # We cannot model more objects in the graph that what we have loaded
        #     self.num_objects = self.num_blocks
        # else:     
        self.num_blocks  = num_blocks
        self.num_objects = self.num_objs_to_load                       # tot num of objects to represent in graph (we use this more descriptive name here, but keep num_blocks for RelationalRL)
    
        # Given the available objects, randomly pick num_objs_to_load and return names, visual names, and name_to_id
        (self.object_names, self.visual_object_names, 
        self.not_yet_considered_object_names, self.not_yet_considered_visual_object_names, 
        self.object_to_id) = self.load_objs_to_simulate(self.num_objs_in_db,self.num_objs_to_load)

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Hard Reset: afects re-loading on reset
        self.hard_reset = hard_reset

        # (b) Strategies
        self.object_reset_strategy  = object_reset_strategy     # organized|jumbled|wall|random
        self.object_randomization   = object_randomization      # randomize with new objects after reset (bool)
        print(f"A total of {self.num_objs_to_load} objects are being loaded. A total of {self.num_objects} will be modeled as nodes.")
        print(f"The object reset strategy is: {self.object_reset_strategy} and new objects will be randomly picked after reset\n")
        #-----------        

        # (C) reward/reset configuration
        self.distance_threshold = 0.05                          # Determine whether obj reaches goal

        self.reward_scale   = reward_scale                      # Sets a scale for final reward
        self.reward_shaping = reward_shaping

        self.first_reset = first_reset
        self.do_reset_internal = True

        self.curr_learn_dist = 0.05                             # curr learn threshold

        # Variant dictionary
        self.variant = variant

        self.camera_image_height        = camera_image_height
        self.camera_image_width         = camera_image_width
        self.use_depth_obs              = camera_depths 
        self.use_gray_img               = use_gray_img
        
        self.use_pygame_render           = use_pygame_render
        self.visualize_camera_obs        = visualize_camera_obs
        self.top_down_grasp              = top_down_grasp,

        # Robot Observations
        self.is_grasping = np.asarray(False).astype(np.float32)
        self.blob_ori = np.zeros([1,2])

        if use_pygame_render:
            import pygame
            if self.visualize_camera_obs:
                if not self.use_depth_obs:
                    self.screen = pygame.display.set_mode((self.camera_image_width, self.camera_image_height))
                else:
                    self.screen = pygame.display.set_mode((self.camera_image_width, 2*self.camera_image_height))
            else:
                self.screen = pygame.display.set_mode((300, 300))

        # Initialize Parent Classes: SingleArmEnv->ManipEnv->RobotEnv->MujocoEnv
        super().__init__(
            robots                  = robots,
            mount_types             = "default",            # additional - set to default            
            gripper_types           = gripper_types,
            env_configuration       = env_configuration,

            control_freq            = control_freq,
            controller_configs      = controller_configs,

            use_camera_obs          = use_camera_obs,
            horizon                 = horizon,
            ignore_done             = ignore_done,
            hard_reset              = hard_reset,

            camera_names            = camera_names,
            camera_segmentations    = camera_segmentations,
            camera_heights          = camera_image_height,
            camera_widths           = camera_image_width,
            camera_depths           = camera_depths,

            has_renderer            = has_renderer,
            has_offscreen_renderer  = has_offscreen_renderer,
            render_camera           = render_camera,
            render_collision_mesh   = render_collision_mesh,
            render_visual_mesh      = render_visual_mesh,
            render_gpu_device_id    = render_gpu_device_id,            

            initialization_noise    = initialization_noise,
            top_down_grasp          = top_down_grasp,                     
        )

        # Serializable Class
        self._serializable_initialized = False
        Serializable.quick_init(self, locals()) # Save this classes args/kwargs

    def clear_object_strucs(self):

        self.objects.clear()
        self.visual_objects.clear()

        self.not_yet_considered_visual_objects.clear()
        self.not_yet_considered_objects.clear()

        self.object_names.clear()
        self.visual_object_names.clear()

        self.not_yet_considered_object_names.clear()
        self.not_yet_considered_visual_object_names.clear()

        self.object_to_id.clear()
        self.object_id_to_sensors.clear()
        self.sorted_objects_to_model.clear()

        self.object_placements.clear()

        self.other_objs_than_goals.clear()
        self.objects_in_target_bin.clear()

        self.goal_object.clear()        

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_goal_object(self):
        '''
        Plays the role of selecting a desired object for order fulfillment (name,pose). Selected when: (i) starting, or (ii) a previous goal has been picked successfully.
        Currently, randomly choose an object from the list of self.object_names which has num_objs_to_load.

        Assumes that (visual) object placements from reset_sim() are available: 
        self.object_placements = self.placement_initializer.sample()

        Returns:
            goal dict copy with name, pos, quat keys and string and np.arrays as values.             If objects are unavailable, return emptry name zero pos and identity quat.                
            other_objects list of object names
        '''
        goal_obj = {}
        
        if len(self.object_names) > 0:
            # Select a goal obj
            goal_obj['name'] = np.random.choice(self.object_names)

            # Extract the pos and quat from the (pos,quat) tuple of the object_placements
            goal_obj['pos']  = self.object_placements[goal_obj['name']][0]
            goal_obj['quat'] = self.object_placements[goal_obj['name']][1]
            
            # Prepare a list of other objs not containing goal. Note: it does not include the 'not_yet_considered_objects' that are not being modelled.
            other_objs_to_consider = self.object_names.copy()
            other_objs_to_consider.remove(goal_obj['name'])     


        # Else: "There are no objects to load!! zero out info."
        else:
            print("Picking.get_goal_object(): cannot choose goal as self.object_names list is empty")
            goal_obj['name'] = []
            goal_obj['pos']  = np.array([0, 0, 0])
            goal_obj['quat'] = np.array([1, 0, 0, 0])

            other_objs_to_consider = []

        return goal_obj.copy(), other_objs_to_consider.copy()
    
    def compute_reward(self, achieved_goal, desired_goal, is_grasping, info):
        """
        Computes discrete sparse reward, perhaps in an off-policy way during training. 
        
        A negative format is used: if goal is not met ==> penalized with -1. 
        The policy will be encouraged to minimize the cost. A perfect policy would have returns equal to 0.
        
        - Use achieved goal and desired goal positions to determine the reward. 
        - TODO: test if a normalized incremental reward would be better as done in reward()

        :param achieved_goal: (numpy array) goal_object's pos(3) and quat(4)
        :param desired_goal:  (numpy array) target pos(3) and quat(4)
        :param info:          (bool)        indicating success or failure
        :return:              (float)       reward 
        """
        
        # Compute position subdistances
        ag_pos = achieved_goal[:3]
        dg_pos = desired_goal[:3]
        dist   = self.goal_distance(ag_pos, dg_pos)

        # Sparse reward calculation: negative format
        # - If you do not reach your target, a -1 will be assigned as the reward, otherwise zero.
        # - a perfect policy would get returns equivalent to 0        
        # reward = -(dist > self.distance_threshold).astype(np.float32)
        # reward = np.min([-(dist > self.distance_threshold).astype(np.float32) for d in dist], axis=0)

        # Sparse reward calculation: positive format
        # - If we do not reach target, a 0 will be assigned as the reard, otherwise 1.
        # - a perfect policy would get returns equivalent to 1
        # reward_goal = (dist < self.distance_threshold).astype(np.float32)
        # reward_grasping = 0.05*is_grasping

        if is_grasping and achieved_goal[2] > 0.90:
            reward = 50
            # reward = 0.5

            # if achieved_goal[2] > 0.85:
            #     reward = 10*(achieved_goal[2]-0.85)
             
        elif is_grasping:
            reward = 0.5
        else:
            reward = 0

        reward = np.asarray(reward)
        return reward          

    def reward(self, action=None):
        """
        We compute a sparse normalized reward. It is possible that different number of objects are loaded in different experiments, 
        it is useful to have the sum of rewards be equal to scaled_reward (usually 1)
        
        1. Give a discrete reward of 1.0 per object if it is placed in its correct bin
        *Note that a successfully completed task (object in bin) will return 1.0 per object irregardless of whether the
        environment is using sparse or shaped rewards

        2. Check the length of objects in the target bin

        3. Scale and normalize        

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value

        TODO: given that there is a built-in scaled-reward assumption throughout the algorithms (though right now it is set to 1.0), it may be best to use positive rewards for success and 0 for failure.
        """
        # compute sparse rewards
        if self.is_success():
            reward = len(self.objects_in_target_bin)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)

        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= self.num_objs_to_load
        return reward

    def staged_rewards(self):
        """
        robotsuite staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        active_objs = []
        for i, obj in enumerate(self.objects+self.not_yet_considered_objects):
            if self.objects_in_target_bin[i]:
                continue
            active_objs.append(obj)

        # reaching reward governed by distance to closest object
        r_reach = 0.
        if active_objs:
            # get reaching reward via minimum distance to a target object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_obj.root_body,
                    target_type="body",
                    return_distance=True,
                ) for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms])
        ) * grasp_mult

        # lifting reward for picking up an object
        r_lift = 0.
        if active_objs and r_grasp > 0.:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name]
                                                     for active_obj in active_objs]][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                    lift_mult - grasp_mult
            )

        # hover reward for getting object above bin
        r_hover = 0.
        if active_objs:
            target_bin_ids = [self.object_to_id[active_obj.name.lower()] for active_obj in active_objs]
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name]
                                                     for active_obj in active_objs]][:, :2]
            y_check = (
                    np.abs(object_xy_locs[:, 1] - self.target_bin_placements[target_bin_ids, 1])
                    < self.bin_size[1] / 4.
            )
            x_check = (
                    np.abs(object_xy_locs[:, 0] - self.target_bin_placements[target_bin_ids, 0])
                    < self.bin_size[0] / 4.
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(
                self.target_bin_placements[target_bin_ids, :2] - object_xy_locs, axis=1
            )
            # objects to the left get r_lift added to hover reward,
            # those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(active_objs))
            r_hover_all[objects_above_bins] = lift_mult + (
                    1 - np.tanh(10.0 * dists[objects_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover_all[objects_not_above_bins] = r_lift + (
                    1 - np.tanh(10.0 * dists[objects_not_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_placement_initializer(self):
        """
        Helper function to define placement initializers that sample object poses within bounds.

        Create 3 samplers: 
        - picking, placing (i.e. goals) and for setting the robot(s) eef upon a reset according to a strategy. 

        Strategies:
        1. Jumbled: randomly place n objects and visual objects in appropriate bins.
        2. Wall: place objects neara the wall. 
        3. Stacked: (for boxed objects) stack them neatly in 3D
        
        Note: 
        - Each (visual) object/robot will get a instantiated sampler. 
        - Samplers can be accessed via the samplers object. Each sub-sampler obj is accessed as a dict key self.placement_initializer.samplers[sampler_obj_name]

        %---------------------------------------------------------------------------------------------------------
        TODO: 1) extend this function to place objects according to strategy: organized.
        %---------------------------------------------------------------------------------------------------------
        """
        # init eef [-0.02423557, -0.09839531,  1.02317629]
        if self.object_reset_strategy == 'random':
            self.object_reset_strategy = random.choice(object_reset_strategy_cases[0:2]) # Do not include random in selection

        if self.object_reset_strategy == 'jumbled':
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")  # Samples position for each object sequentially. Allows chaining multiple placement initializers together - so that object locations can be sampled on top of other objects or relative to other object placements.

            # can sample anywhere in bin
            bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05 # half of bin - edges (2*0.025 half of each side of each wall so that we don't hit the wall)
            bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05
            # pickObjectSampler: (non-visual) objects are sampled within the bounds of the picking bin #1 (with some tolerance) and outside the object radiuses
            self.placement_initializer.append_sampler(
                sampler = UniformRandomSampler(
                    name                            = "pickObjectSampler",
                    mujoco_objects                  = self.objects+self.not_yet_considered_objects,
                    x_range                         = [-0.75*bin_x_half, 0.75*bin_x_half], # [0, bin_x_half],    # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    y_range                         = [-0.75*bin_y_half, 0.75*bin_y_half], # [0, 0.75*bin_y_half],
                    # x_range                         = [-self.curr_learn_dist, self.curr_learn_dist],                # 5 cm from ref
                    # y_range                         = [-self.curr_learn_dist, self.curr_learn_dist],
                    rotation                        = None,                         # Add uniform random rotation
                    rotation_axis                   = 'z',                          # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    reference_pos                   = self.bin1_pos + self.bin1_surface,
                    # reference_pos                   = [-0.02423557, -0.09839531,  self.bin1_pos[2]+self.bin1_surface[2]],
                    z_offset                        = 0.,
                )
            )

        elif self.object_reset_strategy == 'wall':
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")  # Samples position for each object sequentially. Allows chaining multiple placement initializers together - so that object locations can be sampled on top of other objects or relative to other object placements.

            # can sample anywhere in bin
            bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05  # half of bin - edges (2*0.025 half of each side of each wall so that we don't hit the wall)
            bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

            # pickObjectSampler: (non-visual) objects are sampled within the bounds of the picking bin #1 (with some tolerance) and outside the object radiuses
            self.placement_initializer.append_sampler(
                sampler = UniformWallSampler(
                    name                            = "pickObjectSampler",
                    mujoco_objects                  = self.objects+self.not_yet_considered_objects,
                    # x_range                         = [-bin_x_half, bin_x_half],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    # y_range                         = [-bin_y_half, bin_y_half],
                    x_range                         =[-0.05, 0.02],  # 5 cm from ref
                    y_range                         =[-0.05, 0.05],
                    rotation                        = None,                             # Add uniform random rotation
                    rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    # reference_pos                   = self.bin1_pos + self.bin1_surface,
                    reference_pos                   =[-0.02423557, -0.09839531, self.bin1_pos[2] + self.bin1_surface[2]],
                    z_offset                        = 0.,
                )
            )

        # Stacked??
        else:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            # can sample anywhere in bin
            bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05  # half of bin - edges (2*0.025 half of each side of each wall so that we don't hit the wall)
            bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

            # pickObjectSampler: (non-visual) objects are sampled within the bounds of the picking bin #1 (with some tolerance) and outside the object radiuses
            self.placement_initializer.append_sampler(
                sampler                             = UniformRandomSampler(
                    name                            = "pickObjectSampler",
                    mujoco_objects                  = self.objects+self.not_yet_considered_objects,
                    x_range                         = [-bin_x_half, bin_x_half],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    y_range                         = [-bin_y_half, bin_y_half],
                    rotation                        = None,                             # Add uniform random rotation
                    rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    reference_pos                   = self.bin1_pos + self.bin1_surface,
                    z_offset                        = 0.,
                )
            )
        # placeObjectSamplers: each visual object receives a sampler that places it in the TARGET bin
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name                            = "placeObjectSampler",             # name for object sampler for each object
                mujoco_objects                  = self.visual_objects+self.not_yet_considered_visual_objects,
                x_range                         = [-bin_x_half/2, bin_x_half/2],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                y_range                         = [-bin_y_half/2, bin_y_half/2],
                rotation                        = None,                             # Add uniform random rotation
                rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                ensure_object_boundary_in_range = True,
                ensure_valid_placement          = True,
                reference_pos                   = self.bin1_pos + self.bin1_surface + 0.03,
                z_offset                        = 0.10,                             # Set a vertical offset of XXcm above the bin
                z_offset_prob                   = 0.50,  # probability with which to set the z_offset
            )
        )

        # robot_eefSampler:
        # TODO: this eefSampler probably best placed in robosuite/environments/robot_env.py.reset() where init_qpos + noise is computed.
        # Then, it's execution should go inside robosuite/controllers/base_controller.py:Controller.update_base_pose() via IK or interpolation/controller

        # Currently letting the eef take a position anywhere on top of bin1.
        # Could keep at center by changing xrange to the self.bin1_pos only
        min_z = 0.25  # set a min lower height for the eef above table (i.e. 25cm)
        max_z = min_z + 0.30  # set an upper height for the eef

        self.robot_placement_initializer = robotUniformRandomSampler(
            name="robot_eefSampler",
            mujoco_robots=self.model.mujoco_robots,
            x_range=[-bin_x_half, bin_x_half],
            y_range=[-bin_y_half, bin_y_half],
            z_range=[min_z, max_z],
            rotation=None,
            rotation_axis='z',
            reference_pos=self.bin1_pos + self.bin1_surface,
            )

    def _load_model(self):
        """
        Create a manipulation task object. 
        Requires a (i) mujoco arena, (ii) robot (+gripper), and (iii) object + visual objects + not_yet_considered_objects (and visual objects)
        
        Return an xml model under self.model
        """
        super()._load_model() # loads robot_env->robot; single_arm->gripper; manipulation_env-> robot/gripper/objects/bins
        
        # Extract hard coded starting pose for your robot
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"] # Access your robot's (./model/robots/manipulator/robot_name.py) base_xpose_offset hardcoded position as dict
        # Place your robot's base at the indicated position. 
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top/bins workspace
        mujoco_arena = BinsArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size       = mujoco_arena.table_full_size # bin_size is really the area covered by the two bins
        self.bin1_pos       = mujoco_arena.bin1_pos
        self.bin2_pos       = mujoco_arena.bin2_pos
        self.bin1_surface   = mujoco_arena.bin1_surface
        self.bin2_surface   = mujoco_arena.bin2_surface
        self.bin1_friction  = mujoco_arena.bin1_friction
        self.bin2_friction  = mujoco_arena.bin2_friction
        
        # Given the available objects, randomly pick num_objs_to_load and return: names, visual names, and not_yet_modelled_equivalents and name_to_id
        (self.object_names, self.visual_object_names, 
        self.not_yet_considered_object_names, self.not_yet_considered_visual_object_names, 
        self.object_to_id) = self.load_objs_to_simulate(self.num_objs_in_db,self.num_objs_to_load)

        # B. Extract class names (must match those in ./robosuite/models/objects/xml_objects.py ) and call them with the name of the MujocoXMLObject
        (self.visual_objects, self.objects, 
        self.not_yet_considered_visual_objects, self.not_yet_considered_objects)  = self.extract_obj_classes()

        self.mujoco_objects = self.visual_objects + self.objects + self.not_yet_considered_visual_objects + self.not_yet_considered_objects
        # print("{} self objs, {} self not yet cons objs".format(len(self.objects), len(self.not_yet_considered_objects)))
        # insantiate object model: includes arena, robot, and objects of interest. merges them to return a single model. 
        self.model = ManipulationTask(
            mujoco_arena    = mujoco_arena,
            mujoco_robots   = [robot.robot_model for robot in self.robots], 
            mujoco_objects  = self.mujoco_objects
        )

        # Create placement initializers for each existing object (self.placement_initializer): will place according to strategy
        self._get_placement_initializer()

    # Next 3 methods all handle observables
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data. 
        
        Locally
        01 Object body and geom ids 
        02 Set target_bin_placements

        Parent: (robot_env) sets up robot-specific references 
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.mujoco_objects):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects+self.not_yet_considered_objects), 3))
        for i, obj in enumerate(self.objects+self.not_yet_considered_objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2. # x.center point - 1/2 length
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.  # y.center point - 1/2 width
            bin_x_low += self.bin_size[0] / 4.      # now between wall & center.x
            bin_y_low += self.bin_size[1] / 4.      # now between wall & center.y
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled (i.e. for 1 object: _pos, _quat, _velp, _velr, eef_pos, eef_quat)

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables() # Creates observables for gripper and robot. Each observable has an interface to mujoco methods to extract the appropriate data xpos/quat/velp/velr and others

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix #'robot0'
            modality = "object"

            # Reset obj sensor mappings
            self.object_id_to_sensors = {}

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors  = [world_pose_in_gripper]
            names    = ["world_pose_in_gripper"]
            enableds = [True]
            actives  = [False]

            # convert obj name from string to obj

            # Create sensors for objects
            for i, obj in enumerate(self.objects+self.not_yet_considered_objects):
                # Create object sensors
                using_obj = True #(self.single_object_mode == 0 or self.object_id == i)
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality) # creates obj pos/quat/velp/velr/eef_pos/eef_quat snesors
                num_obj_sensors = len(obj_sensor_names)
                sensors     += obj_sensors
                names       += obj_sensor_names
                enableds    += [using_obj] * num_obj_sensors    # Created object sensors: world_pose_in_gripper + pos, quat, vel, obj_to_robot_eef_pos & quat for each object
                actives     += [using_obj] * num_obj_sensors
                self.object_id_to_sensors[i] = obj_sensor_names

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        We create: obj_pos

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        # ---- Additional object sensors added to replicate Fetch Tasks: velocity and angular velocity
        @sensor(modality=modality)
        def obj_velp(obs_cache):
            return np.array(self.sim.data.get_site_xvelp("gripper0_grip_site")) # Better to set this via gripper.important_sites["grip_site"]. not sure if i can access it here.                       

        @sensor(modality=modality)
        def obj_velr(obs_cache):
            return np.array(self.sim.data.get_site_xvelr("gripper0_grip_site"))

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_velp, obj_velr, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_velp",  f"{obj_name}_velr", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _activate_her(self, obj_pos, obj_quat, obj):
        """
        Place objects inside the gripper site 50% of the time.
        - offset: Approx offset between gripper site and gripper base
        - longitude_max: Max length of objects that the gripper can manage to grip
        Gripping strategies divided into 3 cases by checking min (x_radius, y_radius, vertical_radius/2):
        - x_radius grip
            1. lower obj height by subtracting offset from half of obj.vertical_radius/2 to prevent obj & grip base collision
            2. reset obj_quat to default quat
            3. move gripper fingers and update gripper sim according to obj.x_radius
        - y_radius grip if x_radius * 2 > long_max
            1. lower obj height by subtracting offset from half of obj.vertical_radius/2 to prevent obj & grip base collision
            2. rotate obj_quat 90 deg in x-axis
            3. move gripper fingers and update gripper sim according to obj.y_radius
        - vertical_radius grip
            1. rotate obj_quat 90 deg in z-axis
            2. lower obj height by subtracting offset from of obj.y_radius to prevent obj & grip base collision
            3. move gripper fingers and update gripper sim according to obj.y_radius
        """
        # Set HER 50% of the time
        HER = np.random.uniform() < 0.0#0.50
        # introduce offset between grip site and gripper mount to prevent collision
        offset = 0.03
        # maximum gripping space
        longitude_max = 0.07
        # HER flag for activating HER 100% all the time
        # HER = True
        if HER:
            # Rename goal object pos as eef pos, goal object quat
            HER_pos = self._eef_xpos
            HER_quat = obj_quat
            min_longitude = min(obj.x_radius * 2, obj.y_radius * 2, obj.vertical_radius)
            # Gripping strategy if horizontal radius is the shorter side
            if min_longitude == (obj.x_radius * 2) or min_longitude == (obj.y_radius * 2):
                # Check for offset
                if (obj.vertical_radius > offset):
                    HER_pos[2] -= (obj.vertical_radius / 2 - offset)
                # Rotate if current orientation is too long
                if (obj.x_radius * 2 >= longitude_max):
                    # y_radius gripping strategy
                    # quat = (x,y,z,w)
                    # rx 90 degrees
                    HER_quat = [0.98, 0, 0, 0]
                    # Set left & right fingers to reach the goal obj
                    self.sim.data.set_joint_qpos('gripper0_finger_joint1', obj.y_radius)
                    self.sim.data.set_joint_qpos('gripper0_finger_joint2', -obj.y_radius)
                # Otherwise revert back to obj default orientation
                else:
                    # x_radius gripping strategy
                    HER_quat = [0, 0, 0, 1]
                    # Set left & right fingers to reach the goal obj
                    self.sim.data.set_joint_qpos('gripper0_finger_joint1', obj.x_radius)
                    self.sim.data.set_joint_qpos('gripper0_finger_joint2', -obj.x_radius)
            # Gripping strategy if the vertical radius is the shorter side
            else:  # rz 90 degreez
                HER_quat = [0, 0, 0.7, -0.7]
                # Check for offset
                if obj.y_radius > offset:
                    HER_pos[2] -= (obj.y_radius - offset)
                # Set left & right fingers to reach the goal obj
                self.sim.data.set_joint_qpos('gripper0_finger_joint1', obj.vertical_radius / 2)
                self.sim.data.set_joint_qpos('gripper0_finger_joint2', -obj.vertical_radius / 2)

            # Update goal_object with (HER_pos, HER_quat) on the simulation
            self.object_placements[self.goal_object['name']]=(HER_pos, HER_quat, obj)
            self.goal_object['pos'] = HER_pos
            self.goal_object['quat'] = HER_quat
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(HER_pos), np.array(HER_quat)]))
        else:
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _reset_internal(self):
        """
        Resets the simulation's internal configuration and object positions and robot eef according to object_reset_strategy. 
        [organized, jumbled, wall, random].  Recall that we have a multi-object environment. 
        
        Note that objects are not reset until all objects have been successfully picked. What objects are loaded after reset
        depend on the object_randomization flag. If set to True, a new set of objects are loaded, otherwise, the same objects 
        are used.The number of objects to load is controlled by num_obs_to_load. 

        Since we are using a GNN model for DRL, we use num_objs to denote which objs to consider for graph modeling as nodes. 
        This results in two sets of lists (of objects, visual objects, and their corresponding names). i.e. self.objects vs 
        self.not_yet_considered_bojects. 

        Everytime an object is picked, a new goal object is picked from within the modelled objects while at the same time a 
        not_yet_considered_object is moved into the modeled group. 
        Note: at the beginning of the program reset() may be called upto 3 times before actually starting rollouts: by Picking.__init__,  GymWrapper.__init__, and by RLAlgorithm._start_new_rollout()
        
        Object Placement Strategies (activated in self.placement_initializer)
        - 'organized': nicely stacked side-by-side, if no more fit, on-top of each other. left-to-right.
        - 'jumbled':   just dropped in from top at center, let them fall freely.
        - 'wall':      align objects with wall. start on far wall on the left and move clock-wise.
        - 'random':    select above strategy randomly with each new reset.        

        """
        global _reset_internal_after_picking_all_objs

        # if we have not finished picking_all_objs, calling _reset_internal will do nothing
        if not _reset_internal_after_picking_all_objs:
            return

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset: # i.e. stochastic (flag set in base.py)

            if self.first_reset:
                super()._reset_internal() # action_dim|controllers|cameras|model|render
                self.first_reset = False # reset() is called during base.py:MujocoEnv.__init__(), then by robosuite/wrappers/gym_wrappery.py:GymWrapper.__init__, and then when starting to train (batch/online) rlkit/core/rl_algorithm.py:RLAlgorithm._start_new_rollout 

                # After first reset, if object_randomization is true, turn on the self.hard_reset flag to be used in the next reset
                if self.object_randomization:
                    self.hard_reset = True

            # if not first reset but on obj rand, clear fallen_objs, and turn off flag
            elif not self.first_reset and self.object_randomization:
                super()._reset_internal()
                self.objects_in_target_bin.clear()
                self.fallen_objs.clear()    
                self.fallen_objs_flag = False


            # II. Not Object Randomizations. 
            # Continuing Reset. 
            # Copy objects in target bin back to object names and then clear the former. 
            # After that, both (i) and (ii) require us to collect the object's positions and orientation. And for collision objects, set the HER strategy.
            else:
                super()._reset_internal()
                if self.objects_in_target_bin != []:

                    # Print object info before reset
                    print('Current objects in target bin are: ')
                    for obj in self.objects_in_target_bin:
                        print(f'{obj}    ')
                    # Bring back objects from target bin to object_names and not_yet_consiered_object_names
                    # Use slicing so that you create new objects (no need to copy)
                    if self.object_names == []:
                        self.object_names += self.objects_in_target_bin[:self.num_objects]
                        self.not_yet_considered_object_names += self.objects_in_target_bin[self.num_objects:]
                        self.objects_in_target_bin.clear()

                    # Print object info after the reset
                    print('After the reset, the modeled object names are: ')
                    for obj in self.object_names:
                        print(f'{obj} ')
                    
                    if len(self.not_yet_considered_object_names) != 0:
                        print('And thet unmodeled or not yet considered objects are: ')
                        for obj in self.not_yet_considered_object_names:
                            print(f'{obj} ')

                    # Proceed to place objects at the self.object_placements location.
                
            if not self.object_randomization and self.fallen_objs_flag:
                self.object_names = [name[:5]+'Object' for name in self.visual_object_names]
                self.not_yet_considered_object_names = [name[:5]+'Object' for name in self.not_yet_considered_visual_object_names]
                self.fallen_objs.clear()
                # C> Turn off flag
                self.fallen_objs_flag = False

            # Sample from the "placement initializer" for all objects (regular and visual objects)
            self.object_placements = self.placement_initializer.sample()
            
            # Set goal object to pick up and sort closest objects to model
            self.goal_object, self.other_objs_than_goals = self.get_goal_object()               
            
            # Position the objects
            for obj_pos, obj_quat, obj in self.object_placements.values():
                if obj.name == [] or self.goal_object['name'] == []:
                    break
                # Set the visual object body locations
                if "visualobject" in obj.name.lower():                             # switched "visual" for "v"
                    self.sim.model.body_pos[self.obj_body_id[obj.name]]  = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat

                    ## Addition ---
                    # self.object_placements is a place holder for all objects. However:
                    # Under HER paradigm, we have a self.goal variable for the desired goal +
                    # Under our current architecture we set self.goal_object as a single goal until that object is placed. 
                    # Use this to fill self.goal which will be used in _get_obs to set the desired_goal.
                    if obj.name.lower() == self.goal_object['name'][:5] + 'visualobject':
                            self.goal_object['pos'] = obj_pos
                            self.goal_object['quat'] = obj_quat

                # Set the position of 'collision' objects:
                elif obj.name.lower() == self.goal_object['name'].lower():
                    self._activate_her(obj_pos=obj_pos, obj_quat=obj_quat, obj=obj)
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # Set the bins to the desired position
            self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
            self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        return True

    def return_sorted_objs_to_model(self, goal, others):
        '''
        The goal of this method is to return a dictionary num_objects in length that are closest to self.goal_object ['name'] that we want to model in the graph as nodes
        1) Access all objects that are not the goal (i.e. self.other_objs_than_goals)
        2) Compute their norm with the goal
        3) Sort them
        4) Only keep the first num_object-1 entries (includes the goal)
        5) Place the goal at the front

        Example:
        : objs_to_load = [goal_obj, obj2, obj3, obj4, obj5] (assumed already sorted)
        : num_objects (to model) = 3

        : sorted_obj_dist = [obj2, obj3] # keep num_objects -1
        : sorted_obj_dist = [goal_obj, obj2, obj3] # place goal in the front

        Args:   
            obs:        dict of observables with name/pos/quat                    
            goal:       goal dict with name/pos/quat keys
            others:     list of object_names to be sorted

        Returns:
            sorted_obj_dist:    dictionary with name as key and distance as value.

        '''

        # Check for an empty goal
        if goal == {} or goal['name'] == []:
            return {}

        obj_dist        = {}
        sorted_obj_dist = {}

        # 1) Compute norm between goal_object and objects_to_consider
        for other in others:

            # Get pos from observables
            val = np.linalg.norm(self._observables[self.goal_object['name']+'_pos'].obs - self._observables[other+'_pos'].obs)
            obj_dist[other] = val


        # 2) Sort the dictionary by norm value
        sorted_obj_dist_tuple = sorted(obj_dist.items(), key = lambda item: item[1])
        sorted_obj_dist = {k: v for k, v in sorted_obj_dist_tuple[:self.num_objects-1]} # notice self.num_objects-1. This indicates the number of obj we wish to model excluding the goal. 

        # 3) Place goal object at the front using OrderedDict move_to_end(,last=False)
        sorted_obj_dist = OrderedDict(sorted_obj_dist)
        
        sorted_obj_dist[ self.goal_object['name'] ] = 0 # the distance value of goal is zero
        sorted_obj_dist.move_to_end( self.goal_object['name'], last=False) # move to FRONT

        return sorted_obj_dist

    def return_fallen_objs(self):
        """
        return list of fallen objs names if lower than table height and update obj lists/dicts accordingly
        -when modelled obj fell, remove from obj_names, sorted_objs_to_model, then add 1 unmodelled to both
        -when unmodelled obj fell, remove from not_yet_consd_obj_names
        -if goal obj fell get new goal obj
        -if other_obj_than_goals fell keep same goal obj and remove obj from other_obj_than_goals list
        """
        if self.object_names == []:
            return []
        # fallen_objs = []

        # 1. Check for fallen objs if obj height is less than table surface
        fallen_objs = [name for name in self.object_names + self.not_yet_considered_object_names
                       if self._observables[name+'_pos'].obs[2] < self.bin1_pos[2] and name not in self.fallen_objs]
        # self.object_names = [name for name in self.object_names if name not in fallen_objs]
        # self.not_yet_considered_object_names = [name for name in self.not_yet_considered_object_names if name not in fallen_objs]
        # if self.not_yet_considered_object_names != []:
        #     self.object_names += [self.not_yet_considered_object_names.pop()
        #                           for name in range ( self.num_blocks - len(self.object_names) ) ]

        # if there is a fallen obj
        # get new goal, other_objs than goals if there is a fallen object
        # if there is no fallen objs, do nothing
        # if there is a fallen goal obj, call get goal obj
        # if there is a fallen not goal obj, keep goal obj, remove fallen obj from self other obj than goal
        if fallen_objs:
            
            # if self.goal_object['name'] in fallen_objs:
            #     self.goal_object, self.other_objs_than_goals = self.get_goal_object()
            # elif self.goal_object['name'] not in fallen_objs:
            #     self.other_objs_than_goals = [name for name in self.other_objs_than_goals if name not in fallen_objs]
            
            # bring goal obj to front
            self.sorted_objects_to_model = self.return_sorted_objs_to_model(self.goal_object, self.other_objs_than_goals)
            print("fallen is {}, pos is {}, goal is {}, other obj is {}".format(fallen_objs, self._observables[fallen_objs[0]+'_pos'].obs, self.goal_object,
                                                                     self.other_objs_than_goals))
            # Turn on flag if we detect 1 fallen obj
            self.fallen_objs_flag = True

        return fallen_objs

    def _is_success(self, achieved_goal, desired_goal):
        """
        01 Success Determination
        HER-Specific check success method comparing achieved and desired positions .
        Currently the achieved_goal (current position of goal object) and desired_goal are numpy arrays with [pos] shape (3,) 
            TODO: currently we do not analyze orientation. Test good performance with position only first. 
            TODO: improve check_grasp construction. Currently finger geometries include the whole pad, which can lead to push behaviors vs picks. 
        
        02 Object handling 
            Assuming that there are n objects in a bin and m modelled objects where m<=n then if success, do:
            - remove goal_object from self.object_names
            - choose a new goal from the remaining modeled objects
            - add a new object from the remaining non-modelled objects (if available)

            - If no more objects anywhere set done to true. 
        
        03 Reactivity
        TODO: consider modifing the definition of is_success according to QT-OPTs criteria to increase reactivity
        requires reaching a certain height... see paper for more. also connected with one parameter in observations.

        04 End-effector
        TODO: after succeeding, in any occurrence, move the end-effector to a starting position

        Args:
            param: goal (dict with name, pos, quat keys)
            other: dict with entries of other objects to be sorted.
        
        Returns:
            bool: True if object placed correctly
            sorted dict: a dict of sorted objects. can be empty. 
        """
        global _reset_internal_after_picking_all_objs

        # 01 Check if Successfull
        # Subtract obj_pos from goal and compute that error's norm:
        target_dist_error = np.linalg.norm(achieved_goal - desired_goal)

        # Include checking whether any pad of the fingers is touching the goal object
        check_grasp = False
        if self.goal_object['name'] == [] or self.goal_object == {}:
            check_grasp = False
        else:
            check_grasp = self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for g in self.object_placements[self.goal_object['name']][2].contact_geoms])

        self.is_grasping = np.asarray(check_grasp).astype(np.float32)

        # If successfully placed
        if achieved_goal[2]>0.90 and self.is_grasping==1: #target_dist_error <= self.goal_pos_error_thresh:

            print("Successfully picked {}". format(self.goal_object['name']))
            # 02 Object Handling
            # Add the current goal object to the list of target bin objects
            assert self.goal_object != {}, 'checking Picking._is_successful(). Your goal_object is empty.'

            self.objects_in_target_bin.append(self.goal_object['name'])
            print("The current objects in the target bin are:")
            for object in self.objects_in_target_bin:
                print(f"{object} ")    
                                   
            # Remove goal from the list of modeled names for the next round            
            self.object_names.remove(self.goal_object['name'])
            self.sorted_objects_to_model.popitem(self.goal_object['name'])
            self.goal_object.clear()

            # Get new goal (method checks if objs available else returns empty)
            if self.object_names != []:
                self.goal_object, self.other_objs_than_goals = self.get_goal_object()
                #self.sorted_objects_to_model.update(self.goal_object['name'])

                # Add one new unmodeled object to self.object_names, the closest one to the goal, if available from the self.not_yet_considered_object_names
                if self.not_yet_considered_object_names:
                    sorted_non_modeled_elems = self.return_sorted_objs_to_model(self.goal_object, self.not_yet_considered_object_names) # returns dict of sorted objects
                    closest_obj_to_goal = list(sorted_non_modeled_elems.items())[1]        # Extract first dict item
                    self.object_names.append( closest_obj_to_goal[0] )                          # Only pass the name
                    self.sorted_objects_to_model[closest_obj_to_goal[0]] = closest_obj_to_goal[1]
                    self.not_yet_considered_object_names.remove(closest_obj_to_goal[0])
                print(f"Computing new object goal. New goal obj is {self.goal_object['name']} with location {self.goal_object['pos']}.")

            else: # len(self.object_names) == 0 and len(self.objects_in_target_bin) == self.num_objs_to_load:
                _reset_internal_after_picking_all_objs = True                                
                print("Finished picking and placing all objects, can call reset internal again")

            return True
        else:
            return False

    def check_success(self):
        """
        **Standard robosuite method. Not used with HER. **

        General check success method based on where the goal object is placed. 

        Check if self.goal_object is placed at target position. 
        To decide: check if a single object has been placed successfully, or if all objects have been placed successfully, or both. 
        
        General structure of method:
            1. check for success test
            2. remove current goal_object form list at next iteration
            3. select new next object_goal 

        Returns:
            bool: True if object placed correctly

        TODO: consider modifing the definition of is_success according to QT-OPTs criteria to increase reactivity
        requires reaching a certain height... see paper for more. also connected with one parameter in observations.
        """
        # Test
        error_threshold = 0.05 # hard-coded to 5cm only for position

        obj_pos = self.sim.data.body_xpos[ self.obj_body_id[ self.goal_object['name']]]

        # Subtract obj_pos from goal and compute that error's norm:
        target_dist_error = np.linalg.norm( self.goal_object['pos'] - obj_pos)

        if target_dist_error <= 0.05 and self.object_names!=0:

            # After successfully placing self.goal_object, remove this from the list of considered names for the next round
            self.object_names.remove(self.goal_object['name'])

            # Get a new object_goal if objs still available
            self.goal_object,_ = self.get_goal_object() 
            print(f"Successful placement. New object goal is {self.goal_object['name']}") 

            # Add the current goal object to the list ob objects in target bins
            self.objects_in_target_bin.append(self.goal_object['name'])            
            

        return True

    def _is_inside_workspace(self, robot0_proprio_obs):
        """
        Check if the robot end-effector is inside a box-like workspace.

        For x- and y-axes, the limits of the workspace match those of the bin.
        For z-axes, the lower limit coincides with the bin's bottom surface and the upper limit is located 50cm above that.

        Returns:
            bool: True if robot end-effector is inside the workspace.

        """
        robot0_gripper_position = robot0_proprio_obs[21:24] # extract end-effector position for robot propoprioception observation vector

        # bin size
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05  # half of bin - edges (2*0.025 half of each side of each wall so that we don't hit the wall)
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        workspace_min = np.array([self.bin1_pos[0]-bin_x_half, 
                                  self.bin1_pos[1]-bin_y_half, 
                                  self.bin1_pos[2]+self.bin1_surface[2]+0.01])

        workspace_max = np.array([self.bin1_pos[0]+bin_x_half, 
                                  self.bin1_pos[1]+bin_y_half, 
                                  self.bin1_pos[2]+self.bin1_surface[2]+0.3])
        
        # True if inside
        x_inside = np.greater(robot0_gripper_position[0], workspace_min[0]) and np.less(robot0_gripper_position[0], workspace_max[0])
        y_inside = np.greater(robot0_gripper_position[1], workspace_min[1]) and np.less(robot0_gripper_position[1], workspace_max[1])
        z_inside = np.greater(robot0_gripper_position[2], workspace_min[2]) and np.less(robot0_gripper_position[2], workspace_max[2])

        if not x_inside: print(f'Robot Gripper with x pos: ({robot0_gripper_position[0]:.2f}) has surpassed workspace with max limits {workspace_max[0]:.2f} and min limits {workspace_min[0]:.2f}')
        if not y_inside: print(f'Robot Gripper with y pos: ({robot0_gripper_position[1]:.2f}) has surpassed workspace with max limits {workspace_max[1]:.2f} and min limits {workspace_min[1]:.2f}')
        if not z_inside: print(f'Robot Gripper with z pos: ({robot0_gripper_position[2]:.2f}) has surpassed workspace with max limits {workspace_max[2]:.2f} and min limits {workspace_min[2]:.2f}')
        
        return x_inside and y_inside and z_inside

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=obj.root_body,
                    target_type="body",
                    return_distance=True,
                ) for obj in self.objects
            ]
            closest_obj_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.objects[closest_obj_id].root_body,
                target_type="body",
            )

    def load_objs_to_simulate(self, num_objs_in_db, num_objs_to_load):
        '''
        Assumes preamble has loaded objets (ObjectXXXX) and visual objects (VisualObjectXXXX) (could also be oXXXX and oXXXXv). 
        Although check for other possible names as well.

            01. Given the global set of objs_in_db and the num_objs_to_load, we sample num_objs_to_load from objs_in_db and place them in object_names. 
            
            02. Further, we use the self.num_objects (which indicates how many objects we will model in the graph) and keep the first num_objects in object_names. 
            The rest will go to another list called not_yet_modelled_objs. 
            
            The latter will be used when the robot successfully picks up an object. Then, object_names can be updated with one more not_modelled_yet_object.
            
            03. We return lists of object names and visual objects as well as a dict of object_to_id for modelled and not_yet_modelled objects.
            These lists/dict are then used to tell the simulation what to load for modeling. 
        
        Args:
            num_objs_in_db (int) number of objects loaded as python modules
            num_objs_to_load (int)  how many objects to insert inthe simulation

        Returns:
        '''

        # Build random list of objects to load. Ideally named o00XX.
        digits   = []
        obj_wo_o = []
        counter  = 1

        object_to_id           = {}
        object_names           = []
        visual_object_names    = []

        not_yet_modelled_object_names = []
        not_yet_modelled_visual_object_names= []

        all_objects = list(range(num_objs_in_db))
        # objs_to_consider = random.sample( all_objects, num_objs_to_load) # i.e.objs_to_consider = [69, 66, 64, 55, 65]
        #---------------------------------------------
        # cyl     = [1,2,5,7]
        # box     = [3,4,8,9,10,68]
        # oblong  = [6,11,12,20,21,]
        # round   = [13,14,15,16,17,18,47,48,49,50,51]
        # spec_objs = box # cyl + box + oblong + round
        #---------------------------------------------
        #objs_to_consider = random.sample(spec_objs, num_objs_to_load)
        objs_to_consider = [13] # boxed objects [3,4,8,9,10,69]         
                                 # 32 is half cylinder # 33 is tennis shoe # 34 is bar clamp        
        
        # 01 Sample number of objects to load
        for idx, val in enumerate(objs_to_consider):
            # Collect all objects whose file name starts with an 'o' and contain 'Object' as in OXXXXObject (substract idx by 1 since list obj1 is indexed at 0)
            if objs_in_db[ objs_to_consider[idx]-1 ][0] == 'o' and "Object" in objs_in_db[ objs_to_consider[idx]-1 ]:
                digit = objs_in_db[ objs_to_consider[idx] -1 ]
                digits.append(digit)                                            # Keep list of existing objects
                
                # Create map name:id
                object_to_id.update({digit:idx+1})                              # idx starts from 1 for diff objs: o00X8:1, oOOX3:2, oOOX9:3
                object_names.append(digit)                                      # o0001, o0002,...,o0010...
                visual_object_names.append(digit[:5]+'VisualObject')            # o0001VisualObject
            
            # Otherwise keep a list of faulty object files 
            else:
                obj_wo_o.append(idx)


        ## Do a second sweep to deal with objects that do not start with 'o'. Compare with list of registered objects
        for idx in obj_wo_o:
            temp = 'o' + str(counter).zfill(4) + 'Object'
            
            # If this number exists, increment and try again before registering. 
            while temp in digits:
                counter += 1
                temp = 'o' + str(counter).zfill(4) + 'Object'
                
            digit = temp
            digits.append(digit)            

            # Create map name:id
            object_to_id.update({digit:idx})                                # idx starts from 1 for diff objs: o00X8:1, oOOX3:2, oOOX9:3
            object_names.append(digit)                                      # o0001Object, o0002Object,...,o0010Object...
            visual_object_names.append(digit[:5]+'VisualObject')            # o0001VisualObject

        # 02. Keep only self.num_objs in objent_names. The rest go to the objs_not_yet_considered
        obj_idx_not_yet_modelled            = num_objs_to_load - self.num_objects
        
        if obj_idx_not_yet_modelled != 0:
            not_yet_modelled_object_names       = object_names[-obj_idx_not_yet_modelled:]
            not_yet_modelled_visual_object_names= visual_object_names[-obj_idx_not_yet_modelled:]

        object_names        = object_names[:self.num_objects]
        visual_object_names = visual_object_names[:self.num_objects]

        return (object_names, visual_object_names, 
                not_yet_modelled_object_names, not_yet_modelled_visual_object_names, object_to_id,)

    def extract_obj_classes(self):
        '''
        Given that self.object_names and self.visual_object_names are available from load, return corresponding class objects 
        '''

        # Check for pre-required object names
        assert self.object_names is not None
        assert self.visual_object_names is not None

        # Create empty lists to collect objects
        objects         = []
        visual_objects  = []

        not_considered_objs     = []
        not_considered_vis_objs = []  

        # Extract visual classes and objects 
        for cls in self.visual_object_names:

            vis_obj = getattr(sys.modules[__name__], cls)           # extract class
            vis_obj = vis_obj(name=cls)                             # Now instantiate class by passing needed constructor argument: name of class
            visual_objects.append(vis_obj)                     

        # Repeat for non-visual objects
        for cls in self.object_names:
            obj = getattr(sys.modules[__name__], cls)          
            obj = obj(name=cls)
            objects.append(obj)  

        # Extract not yet considered visual classes and objects 
        for cls in self.not_yet_considered_visual_object_names:

            vis_obj = getattr(sys.modules[__name__], cls)           # extract class
            vis_obj = vis_obj(name=cls)                             # Now instantiate class by passing needed constructor argument: name of class
            not_considered_vis_objs.append(vis_obj)                     

        # Repeat for non-visual not considered objectsobjects
        for cls in self.not_yet_considered_object_names:
            obj = getattr(sys.modules[__name__], cls)          
            obj = obj(name=cls)
            not_considered_objs.append(obj)              

        return visual_objects, objects, not_considered_vis_objs, not_considered_objs
        
    def _get_obs(self, force_update=False):
        '''
        This declaration comes from the Fetch Block Construction environment in rlkit_relational. In our top class: MujocoEnv
        we have the _get_observations declaration. It returns an Ordered dictionary of observables. 
        This method will get such dict of observations and manipulate them to return a formate amenable to rlkit_relational+HER,
        namely return a dictionay with keys: 'obsevations', 'achieved_goals', and 'desired_goals' composed of numpy arrays.

        We keep this declaration as is for two reasons:
        (1) Facilitate the migration of the rlkit_relational code by keeping the same method name
        (2) Keeping the same contents of the original method: not limited to creating achieved_goals and desired_goal.

        Method executes the following:
        1. Compute observations as np arrays [grip_xyz, f1x, f2x, grip_vxyz, f1v, f2v, obj1_pos, rel_pos_wrt_gripp_obj, obj1_theta, obj1_vxyz, obj1_dtheta obj2... ]
        2. Achieved goal: [goal_obj.pos]                        # First iteration, testing placing only with pos and w/out orientation.
        3. Desired goal: goal pos obtained in ._sample_goal()   # same as achieved_goal 

        Notes: 

        Observable Modalities:
        Currently we do not consider the observable's modalities in this function. 
        The GymWrapper uses them in its constructor... So far I don't think it will be a problem but need to check. 
        
        Orientations:
        For combined pick and place, we will collect quaternions in the robot and object observations to help with pick. 
        However, we will not include the quat's for achieved/desired goals used in placement for now. Important to keep correct dims into account in relationalRL/graph code.
        
        
        Impact on Actions:
        Action dims are set by the controller used (i.e. robosuite's Operational Space Controller (OSC)).
        OSC uses xyz rpy updates for the robot. 
        
        Consideration: will the use of quat's in observations and rpy in actions make learning more difficult for the NN?
        TODO: may need to test performance between in-quat + out-rpy and in-rpy + out-rpy or in-quat + out-quat.
        
        Additional Observations:
        ## TODO: Additional observations
            # (1) End-effector type: use robosuites list to provide an appropriate number to these
            # (2) QT-OPTs DONE parameter for reactivity.
        '''

        # Init achieved_goal
        achieved_goal = []

        # Get robosuite observations as an Ordered dict. keep a local obs reference for convencience (vs. self_observables)
        obs = self._get_observations(force_update) # if called by reset() [see base class] this will be set to True.

        if self.use_camera_obs:            
            if not self.use_depth_obs:

                rgb_image     = obs[self.camera_names[0]+'_image']
                seg_image_obs = obs[self.camera_names[0]+'_segmentation_instance'] # robot0_eye_in_hand_segmentation_instance

                # Process seg image to only retain instances for gripper and object
                proc_image_obs = img.process_gray_mask(rgb_image, seg_image_obs, output_size=(self.camera_image_height, self.camera_image_width))
                #proc_image_obs = img.process_seg_image(seg_image_obs, output_size=(self.camera_image_height, self.camera_image_width))

                # Keep two channels of the image
                proc_image_obs = cv2.merge([proc_image_obs,proc_image_obs])

                # Need another slightly different copy only for major-axis and orientation extraction for the object
                seg_obj_img = img.process_seg_obj_image(seg_image_obs, output_size=(self.camera_image_height, self.camera_image_width))
                self.blob_ori = img.compute_blob_orientation(seg_obj_img)
           
            else:
                # Process segmented instance and depth
                seg_image_obs  = obs[self.camera_names[0]+'_segmentation_instance']
                proc_seg_image = img.process_seg_image(seg_image_obs, output_size=(self.camera_image_height, self.camera_image_width))

                depth_image_obs  = obs[self.camera_names[0]+'_depth']
                proc_depth_image = img.process_depth_image(depth_image_obs, output_size=(self.camera_image_height, self.camera_image_width))

                proc_image_obs = cv2.merge([proc_seg_image, proc_depth_image])

        # Get prefix for robot to extract observation keys
        pf = self.robots[0].robot_model.naming_prefix
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep  # dt is equivalent to the amount of time across number of substeps. But xvelp is already the velocity in 1 substep. It seems to make more sense to simply scale xvel by the number of substeps in 1 env step.        
        
        #--------------------------------------------------------------------------
        # 01a) EEF: pos and vel 
        #--------------------------------------------------------------------------
        grip_pos  = obs[f'{pf}eef_pos']
        grip_quat = obs[f'{pf}eef_quat']        

        grip_velp = obs[f'{pf}eef_velp'] * dt   
        grip_velr = obs[f'{pf}eef_velr'] * dt   

        # Finger: Extract robot pos and vel to get finger data
        gripper_state = obs[f'{pf}gripper_qpos']        
        gripper_vel   = obs[f'{pf}gripper_qvel'] * dt   

        grip_height_from_bin = np.array(grip_pos[2]-self.bin1_pos[2]).astype(np.float32)

        # Concatenate and place in env_obs (1)
        # Note: when we change these, also update the robot/object/goal dims of:
        #  rlkit-relational/examples/relationalrl/train_binPicking and train_binPicking_basic
        #  rlkit-relational/rlkit/torch/relational/modules.py:FetchInputPreprocessing.forward()
        #  rlkit-relational/rlkit/torch/relational/relational_util.py.fetch_preprocessing
        env_obs = np.concatenate([  # 17 dims
            # grip_pos.ravel(),       # 3
            # grip_quat.ravel(),      # 4

            # grip_velp.ravel(),      # 3
            # grip_velr.ravel(),      # 3

            # grip_height_from_bin.ravel(),
            self.is_grasping.ravel(),
            # self.blob_ori.ravel(),      # 2 (minor-axis)

            # gripper_state.ravel(),  # 2
            # gripper_vel.ravel(),    # 2
        ])

        #-------------------------------------------------------------------------- 
        # 01b) Object observations *We do not follow relationalRL here and do not add all objects + grip to achieved goals. Instead just add goal_object
        #-------------------------------------------------------------------------- 

        # Observations for Objects
        # *Note: there are three quantities of interest: (i) (total) num_objs_to_load, (ii) num_objs (to_model), and (iii) goal object. 
        # We report data for num_objs that are closest to goal_object and ignore the rest. This list is updated when is_success is True.
        # We only consider the relative position between the goal object and end-effector, all the rest are set to 0.
        # Check, remove & update fallen objs list/dicts
        self.fallen_objs = self.return_fallen_objs() # remove obj from self.obj_names
        
        # # Place goal object at the front
        if self.fallen_objs == []:
            self.sorted_objects_to_model = self.return_sorted_objs_to_model(self.goal_object, self.other_objs_than_goals)
        
        # TODO: sorted_objects should be updated when an object is successfully picked. Such that when there is one object less, 
        # the new dimensionality is reflected in these observations as well.

        # Initialize obj observations with dim 20 3 pos, 4 quat, 3 velp, 3 velr, 3 obj rel pos, 4 obj rel quat
        object_i_pos = np.zeros(3*self.num_objects)
        object_i_quat = np.zeros(4*self.num_objects)
        object_velp = np.zeros(3*self.num_objects)
        object_velr = np.zeros(3*self.num_objects)
        object_rel_pos = np.zeros(3*self.num_objects)
        object_rel_rot = np.zeros(4*self.num_objects)
        achieved_goal = np.zeros(3)

        for i in range(self.num_objects) :

            name_list = list(self.sorted_objects_to_model)
            # if not empty fill from obs, else leave entries as zeros
            if i <= len(name_list)-1:
                # Pose: pos and orientation
                object_i_pos[3*i:3*(i+1)]  = obs[name_list[i] + '_pos']
                object_i_quat[4*i:4*(i+1)] = obs[name_list[i] + '_quat']

                # Vel: linear and angular
                object_velp[3*i:3*(i+1)] = obs[name_list[i] +'_velp'] * dt
                object_velp[3*i:3*(i+1)] = object_velp[3*i:3*(i+1)] - grip_velp # relative velocity between object and gripper

                object_velr[3*i:3*(i+1)] = obs[name_list[i] +'_velr'] * dt

                # Relative position wrt to gripper:
                # *Note: we will only do this for the goal object and set the rest to 0.
                # By setting to 0 all calculations in the network will be cancelled. Robot should reach only to the goal object.
                # Goal object to be modified if successful (without repeat)
                if i == 0:
                     object_rel_pos[3*i:3*(i+1)] = object_i_pos[:3] - grip_pos
                     object_rel_rot[4*i:4*(i+1)] = T.quat_distance(object_i_quat[:4] ,grip_quat) # quat_dist returns the difference

                    # 02) Achieved Goal: the achieved state will be the object(s) pose(s) of the goal (1st) object
                    #--------------------------------------------------------------------------
                    # TODO: double check if this works effectively for our context + HER. Otherwise can add objects and grip pose.
                    #--------------------------------------------------------------------------
                     achieved_goal = np.concatenate([    # 3          # 7
                        object_i_pos[:3].copy(),    # 3      # Try pos only first.
                        # object_i_quat.copy(), # 4
                    ])

                else:
                    # Fill these rel data with fixed nondata
                    object_rel_pos[3*i:3*(i+1)] = np.zeros(3)
                    object_rel_rot[4*i:4*(i+1)] = np.zeros(4)

            # # Augment observations      Dims:
            # env_obs = np.concatenate([  # 17 + (20 * num_objects)
            #     env_obs,
            #     object_i_pos.ravel(),   # 3
            #     object_i_quat.ravel(),  # 4
            #
            #     object_velp.ravel(),    # 3
            #     object_velr.ravel(),    # 3
            #
            #     object_rel_pos.ravel(), # 3
            #     object_rel_rot.ravel()  # 4
            # ])
            # env_obs = np.concatenate([  # 17 + (20 * num_objects)
            #     env_obs,
            #     object_i_pos[3*i:3*(i+1)].ravel(),  # 3
            #     object_i_quat[4*i:4*(i+1)].ravel(),  # 4

            #     object_velp[3*i:3*(i+1)].ravel(),  # 3
            #     object_velr[3*i:3*(i+1)].ravel(),  # 3

            #     object_rel_pos[3*i:3*(i+1)].ravel(),  # 3
            #     object_rel_rot[4*i:4*(i+1)].ravel()  # 4
            # ])

            ## TODO: Additional observations
            # (1) End-effector type: use robosuites list to provide an appropriate number to these
            # (2) QT-OPTs DONE parameter for reactivity.
        
        # --------------------------------------------------------------------------------------
        # Removed from here:   Finally, append the robot's grip xyz 
        # --------------------------------------------------------------------------------------
        # TODO: should we differentiate between object in hand or not like original fetch?
        # achieved_goal = np.concatenate([
        #     achieved_goal, 
        #     grip_pos.copy(),
        #     grip_quat.copy()
        #     ])
        achieved_goal = np.squeeze(achieved_goal)

        #--------------------------------------------------------------------------
        # 03 Desired Goal
        #--------------------------------------------------------------------------
        # desired_goal = []
        # desired_goal = np.concatenate([ # 3             # 7
        #     self.goal_object['pos'],    # 3             # Try pos only first.
        #     # self.goal_object['quat']    # 4
        # ])
        desired_goal = np.array([0,0,0]).astype(np.float32)

        # Returns obs, ag, and also dg
        return_dict = {
            'observation':   env_obs.copy(),
            'achieved_goal': achieved_goal.copy(),  # [ag_ob0_xyz, ag_ob1_xyz, ... rob_xyz]
            'desired_goal':  desired_goal.copy(),   # [goal_obj_xyz, goal_obj_quat]
            # self.camera_names[0]+'_image': image_obs,
            'image_'+self.camera_names[0]: proc_image_obs.copy(),
            # 'depth_image_'+self.camera_names[0]: depth_image_proc.copy(),

            # TODO: Should we also include modalities [image-state, object-state] from observables? 
            # GymWrapper checks for it, but we may not need GymWrapper.
            # Alternatively we could change GymWrapper to look for the 'obsevation' key and these last two. 
            pf+'proprio-state': obs[pf+'proprio-state'],
            'object-state': obs['object-state'],
        }

        # Images
        #TODO: add image representation
        #if obs[]
        return return_dict

    def step(self, action):
        '''
        Takes a step in simulation with control command @action:

        01 Call sim.forward() 
            sim.forwrad() performs steps 2-21 (http://mujoco.org/book/computation.html#piForward) of a regular sim.step() call. Step 2-21 summarized below.
            fk->poses bodies/geoms/sites/cams
            body inertias+joint axes in global frames centered at CoM
            tendon lengths/moment arms
            composite rigid body inertias -> joint-space inertia matrix
            list of active contacts
            constraint jacobian and residuals
            sensor data/potential energy
            tend/actuator vels
            body vels and rates of change of the joint axes in global frames at CoM
            passive forces: spring-dampers and fluid dynamics
            sensor data that depends on velocity/kinetic energy
            constraint acceleration
            coriolis/centrigual/gravitational forces
            actuator forces/activation dynamics
            compute joint acceleration from all forces except still unknown constraint forces
            compute constraint forces with selected solver. update joint acceleration in mjData.aqcc main out of FwdDyns
            compute sensor data that dpeends on force/acceleration
            
        02 Clip action 
            Note: not necessary when we wrap the env with the NormalizedBoxEnv class). TODO NormalizedBoxEnv currently clips at [-1,+1] and all the same for eef and fingers. Need to fix.

        03 Set action
            :3  -> dx dy dz
            3:6 -> angle-axis representations (ax ay az)
            6   -> gripper command. 
            Passed in by self.controller.set_goal(arm_action) --> 
                self.interpolator_pos.set_goal
                self.interpolator_ori.set_goal 

        04 Step (steps 1-24)
            
            check pos/vels for unacceptably large vals indicating divergence.
            ...
            ... steps from sim.forward() above
            ... 
            check for unacceptably lare values, if so reset
            compare FwdDyn | InvDyn to diagnose poor solver conv
            advance sim state by one time step with selected integrator

        05 Update observables and get new observations
        06 Update 'Done'
        07 Process info 
        08 Process reward

        return obs, reward, done, info 

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            4-tuple:

                - (dict) with 'observations', 'achieved_goal', and 'desired_goal' 
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information

        Raises:
            ValueError: [Steps past episode termination]        
        '''
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy_step update
        policy_step = True

        self._update_observables()
        first_image = self._get_obs(force_update=True)['image_'+self.camera_names[0]][:,:,0]

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):

            # 01. sim.forward()
            self.sim.forward()            
        
            # 02 Copy action to sim.data.ctrl (no mocaps used currently. differs from FetchEnv step approach)
            self._pre_action(action, policy_step)

            # 03 sim.step
            try:
                self.sim.step()                             # Advance simulation
            
            except mujoco_py.builder.MujocoException as e:
                print(e)
                print(F"action {action}") 
            
            policy_step = False

        # 04 Update observables and get new observations. Need force_update=True to get latest image update
        self._update_observables()
        env_obs = self._get_obs(force_update=True)
        
        second_image = env_obs['image_'+self.camera_names[0]][:,:,0]
        env_obs['image_'+self.camera_names[0]] = cv2.merge([first_image, second_image])

        # 05 Render: TODO move this to utils/img_processing.py
        img.render_images(self,env_obs,second_image)
        
        # 06 Process info
        info = { 'is_success': self._is_success(env_obs['achieved_goal'], env_obs['desired_goal']),
                 'is_inside_workspace': self._is_inside_workspace(env_obs['robot0_proprio-state']) }

        # 06b Process Reward * Info
            # TODO: design a manner to describe observations in our graph node setting. currently just 'state', but later will use images in nodes, and can extend beyond.
            # if "image" in self.obs_type:
            #     reward = self.compute_reward_image()
            #     if reward < .05:
            #         info = { 'is_success': True }
            #     else:
            #         info = { 'is_success': False }
            # elif "state" in self.obs_type: ...
            # else:
            #     raise ("Obs_type not recognized")

        # 07 Process Done: 
        # If (i) time_step is past horizon OR (ii) we have succeeded, set to true OR (iii) end-effector moves outside the workspace
        done = (self.timestep >= self.horizon) and not self.ignore_done or info['is_success'] and self.object_names == [] \
               or self.fallen_objs_flag or not info['is_inside_workspace']
        
        # 08 Rewards: for grasping and placing
        reward =  self.compute_reward(env_obs['achieved_goal'], env_obs['desired_goal'], self.is_grasping, info)
    
        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep     

        return env_obs, reward, done, info       

    # -----Serialization------
    def __getstate__(self):
        '''
        Saves key attributes needed to reinstantiate the class. Called on pickle.dumps.
        
        Method ideally would save object. Could do via the Serializable class. 
        However, there is an offending class; namely, self.robots. If you try to pickle this class an exception occurs stating that in mujoco_py/mjbatchrenderer.pyx, L2 import pycuda.driver as drv, no default __reduce__ is found due to a non-trivial __cinit__
        Below, in the commented out section, we tried include the offending class but deleting sub classes... have not yet succeeded. 
        It would be desirable to solve this as it facilitates the re-use of the environment. 
        
        Right now, we save everything except self.robots but then actually need to re-construct the class. 
        Note that the reconstruction is not done directly in __setstate__, we do it outside in a script like rlkit-relational/scripts/sim_goal_conditional_policy.py to allow for customization needed for simulation         
        '''
        # Extract all kwargs        
        d = dict()        
        d['robots'] = self.robot_names                       
        d['reward_scale'] = self.reward_scale
        d['hard_reset'] = self.hard_reset
        d['ignore_done'] = self.ignore_done
        d['object_reset_strategy'] = self.object_reset_strategy
        d['num_blocks'] = self.num_blocks
        d['num_objs_to_load'] = self.num_objs_to_load
        d['object_randomization'] = self.object_randomization
        d['use_object_obs'] = self.use_object_obs
        d['use_camera_obs'] = self.use_camera_obs
        d['reward_shaping'] = self.reward_shaping
        d['top_down_grasp'] = self.top_down_grasp

        # Controller configuration
        d['controller_config'] = self.robot_configs[0]['controller_config']

        d['variant'] = self.variant
        # d['control_freq'] = self.robot_configs[0]['control_freq'] # not needed. inside robot_configs[0]

        # May not need these as you will select custom values to display policy
        d['horizon'] = self.horizon
        d['has_renderer'] = self.has_renderer

        # d = self.__dict__.copy()
        # Keep the last portion of the module string name as the name of the environment
        #d['env_name'] = type(self).__name__
        d['env_name'] = self.variant['expl_environment_kwargs']['env_name']

        # Note:
        # This pickling fails if we save self.robots, i.e.:
        # d['robots'] = self.robots               # list containing robot objects
        # I have not been able to solve this even if I:
        # - immediately later del objects within self.robots
        # - immeidately later del d['robots'] itself

        # Try to remove offending class
        # del d['robots'] 


        return d 
    
    def __setstate__(self, d):
        '''
        __setstate_ will properly extract all args/kwargs and then pass them to the environment's constructure to re-insantiate the object.
        '''
        #Serializable.__setstate__(self, d)   
         
        #self.robot_names                        = d['robots']
        self.robots                             = d['robots']
        # self.robot_configs = list()
        # self.robot_configs.append( d['controller_configs] ) 

        self.reward_scale                       = d['reward_scale']
        self.hard_reset                         = d['hard_reset']
        self.ignore_done                        = d['ignore_done']
        self.object_reset_strategy              = d['object_reset_strategy']
        self.num_blocks                         = d['num_blocks']
        self.num_objs_to_load                   = d['num_objs_to_load']
        self.object_randomization               = d['object_randomization']
        self.use_object_obs                     = d['use_object_obs']
        self.use_camera_obs                     = d['use_camera_obs']
        self.reward_shaping                     = d['reward_shaping']        
        self.variant                            = d['variant']
        self.top_down_grasp                     = d['top_down_grasp']

        # May not need these as you will select custom values to display policy
        self.horizon                            = d['horizon']
        self.has_renderer                       = d['has_renderer']

        # Controller Configs (need the 's' below)
        self.controller_configs                  = d['controller_config']

        # environment name
        env_name = d['env_name']
        del d['env_name'] # without deleting it shows up as a double attribute

        # Remake the picking environment via make in base.py? 
        # No. Opted to rebuild outside to allow to customize some params.    
        #env = suite.make(env_name, *(), **d) 
        #self = env # NormalizedBoxEnv(GymWrapper(env)) 