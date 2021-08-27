#core
import sys
import random
import os.path

import math
import numpy as np
from collections import OrderedDict

# Utilities
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler, robotUniformRandomSampler, UniformWallSampler

# 01 Objects
# Import desired|all objects (and visual objects). Use * to import large number of objects. 
from robosuite.models.objects import * 

# After importing we can extract objects by getting the modules via dir()
mods = dir()

# Name Assumptions:
# object names: oXXXX
# visual object names: oXXXXv
# class names will follow the same name as object names and visual object names. This is a departure from robosuite which sets them as: MilkObject, MilkVisualObject, Milk, and Milkvisual for object name, visual object name, object class, and visual object class name respectively. 

# Assumes names have VisualObject TODO: this may change to o00XXv
visual_objs_in_db   = [ item for item in mods if 'VisualObject' in item]
objs_in_db          = [ item.replace('Visual','') for item in visual_objs_in_db ] #
num_objs_in_db = len(objs_in_db)
# (
#     MilkObject,
#     BreadObject,
#     CerealObject,
#     CanObject,
#     MasterChefCanObject,
# )
# from robosuite.models.objects import (
#     MilkVisualObject,
#     BreadVisualObject,
#     CerealVisualObject,
#     CanVisualObject,
#     MasterChefCanVisualObject,
# )

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

# Globals
object_reset_strategy_cases = ['organized', 'jumbled', 'wall', 'random']
_reset_internal_after_picking_all_objs = True


class Picking(SingleArmEnv):
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

        self_object_to_use (str):    contains a string to the name of the object that would be set as a goal

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

        # move bins
        bin1_pos=(-0.1, -0.25, 0.8),  # Follows xml
        bin2_pos=(-0.1, 0.28, 0.8),
        bin_thickness=(0, 0, 0.02),
        
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

        object_reset_strategy = "random",   # [organized, jumbled, wall, random]. random will randomly choose between first three options. 
        object_randomization  = True,       # Randomly select new objects from database after reset or not

        # Reset
        hard_reset            = True,       # If True, re-loads model|sim|render object w reset call. Else, only call sim.reset and reset all robosuite-internal variables

        # Goals
        goal                    = 0,
        objects_in_target_bin   = [],

        goal_pos_error_thresh   = 0.05,     # Used to determine if the current position of the object is within a threshold of goal position

        # Camera: RGB
        camera_names            = "agentview",
        camera_heights          = 256,
        camera_widths           = 256,
        camera_depths           = False,

        has_renderer            = False,
        has_offscreen_renderer  = True,
        
        render_camera           = "birdview", #TODO: may need to adjust here for better angle for our work
        render_collision_mesh   = False,
        render_visual_mesh      = True,
        render_gpu_device_id    = 0,            # was -1 

        # Noise
        initialization_noise    ="default",
        suite_path              = "",        
    ):
        print('Generating Picking class.\n')
        # Task settings

        # (A) Objects: 
        # 1. Set num_objs_in_db
        # 2. set num_objs_to_load, but first check this number is less than num_objs_in_db
        # 3. set num_objects to represent in graph
        # 4. generate random list of objects to load: more involved as several checks required. 

        # Init structs
        self.object_names           = []
        self.visual_object_names    = []

        self.object_to_id           = {}
        self.object_id_to_sensors   = {}                        # Maps object id to sensor names for that object         

        # Organize object information

        # num_objs_to_load obtained from preamble via reading dir() modules correctly instead of files as below. 
        # obj_dir = os.path.join(suite_path, './models/assets/objects')
        # self.num_objs_in_db = len([name for name in next(os.walk(obj_dir))[2] if 'visual' not in name]) # looks for files without visual        
        self.num_objs_in_db = num_objs_in_db

        if num_objs_to_load < self.num_objs_in_db:
            self.num_objs_to_load  = num_objs_to_load           # tot num of object to load 
        else:
            self.num_objs_to_load = self.num_objs_in_db                   

        if num_blocks > self.num_objs_to_load:
            self.num_blocks  = self.num_objs_to_load            # We cannot model more objects in the graph that what we have loaded
            self.num_objects = self.num_blocks
        else:     
            self.num_blocks  = num_blocks
            self.num_objects = num_blocks                       # tot num of objects to represent in graph (we use this more descriptive name here, but keep num_blocks for RelationalRL)
        
        # Strategies
        self.object_reset_strategy  = object_reset_strategy     # organized|jumbled|wall|random
        self.object_randomization   = object_randomization      # randomize with new objects after reset (bool)
        print(f"A total of {self.num_objs_to_load} objects are being loaded. A total of {self.num_objects} will be modeled as nodes.")
        print(f"The object reset strategy is: {self.object_reset_strategy} and new objects will be randomly picked after reset\n")
        #-----------

        # (B) Objects and Goals
        # Given the available objects, randomly pick num_objs_to_load and return names, visual names, and name_to_id
        self.object_names, self.visual_object_names, self.object_to_id = self.load_objs_to_simulate(self.num_objs_in_db,self.num_objs_to_load)

        # Objects
        self.sorted_objects_to_model = {}                    # closes objects to the self.goal_object based on norm2 dist
        self.object_placements       = {}                    # placements for all objects upon reset    
        self.other_objs_than_goals   = []
        self.objects_in_target_bin   = objects_in_target_bin # list of object names (str) in target bin        

        # Goal pose for HER setting
        self.goal_object            = {}                     # holds name, pos, quat
        self.goal_pos_error_thresh  = goal_pos_error_thresh  # set threshold to determine if current pos of object is at goal.

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # (C) reward configuration
        self.reward_scale   = reward_scale                      # Sets a scale for final reward
        self.reward_shaping = reward_shaping

        self.distance_threshold = 0.05                          # Determine whether obj reaches goal

        # (D) Arena: bins_arena.xml

        # Table---
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction  = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)
        self.bin_thickness = np.array(bin_thickness)

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
            camera_heights          = camera_heights,
            camera_widths           = camera_widths,
            camera_depths           = camera_depths,

            has_renderer            = has_renderer,
            has_offscreen_renderer  = has_offscreen_renderer,
            render_camera           = render_camera,
            render_collision_mesh   = render_collision_mesh,
            render_visual_mesh      = render_visual_mesh,
            render_gpu_device_id    = render_gpu_device_id,            

            initialization_noise    = initialization_noise,            
        )

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_goal_object(self):
        '''
        Plays the role of selecting a desired object for order fulfillment (name,pose)
        Currently, randomly choose an object from the list of self.object_names which has num_objs_to_load.

        Assumes that (visual) object placements from reset_sim() are available: 
        self.object_placements = self.placement_initializer.sample()

        If objects are unavailable, return none
        
        Could make more sophisticated in the future
        '''
        assert self.num_objs_to_load >= 0, "There are no objects to load!! Success."

        # Select a goal obj
        goal_obj = {}
        goal_obj['name'] = np.random.choice(self.object_names)

        # Extract the pos and quat from the (pos,quat) tuple of the object_placements
        goal_obj['pos']  = self.object_placements[goal_obj['name']][0]
        goal_obj['quat'] = self.object_placements[goal_obj['name']][1]
        
        # Prepare a list of other objs not containing goal
        other_objs_to_consider = self.object_names.copy()
        other_objs_to_consider.remove(goal_obj['name'])

        return goal_obj.copy(), other_objs_to_consider.copy()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
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
        reward = -(dist > self.distance_threshold).astype(np.float32)
        # reward = np.min([-(dist > self.distance_threshold).astype(np.float32) for d in dist], axis=0)

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
        for i, obj in enumerate(self.objects):
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

        Create 3 samplers: picking, placing, and for setting the robot(s) eef upon a reset. 
        
        Note: 
        - Each (visual) objects/robot will get a instantiated sampler. 
        - Samplers can be accessed via the samplers object. Each sub-sampler obj is accessed as a dict key self.placement_initializer.samplers[sampler_obj_name]

        %---------------------------------------------------------------------------------------------------------
        TODO: 1) extend this function to place objects according to strategy: wall, organized.
        %---------------------------------------------------------------------------------------------------------
        """
        if self.object_reset_strategy == 'jumbled':
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")  # Samples position for each object sequentially. Allows chaining multiple placement initializers together - so that object locations can be sampled on top of other objects or relative to other object placements.

            # can sample anywhere in bin
            bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05 # half of bin - edges (2*0.025 half of each side of each wall so that we don't hit the wall)
            bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

            # pickObjectSampler: (non-visual) objects are sampled within the bounds of the picking bin #1 (with some tolerance) and outside the object radiuses
            self.placement_initializer.append_sampler(
                sampler = UniformRandomSampler(
                    name                            = "pickObjectSampler",
                    mujoco_objects                  = self.objects,
                    x_range                         = [-bin_x_half, bin_x_half],    # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    y_range                         = [-bin_y_half, bin_y_half],
                    rotation                        = None,                         # Add uniform random rotation
                    rotation_axis                   = 'z',                          # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    reference_pos                   = self.bin1_pos+self.bin_thickness,
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
                    mujoco_objects                  = self.objects,
                    x_range                         = [-bin_x_half, bin_x_half],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    y_range                         = [-bin_y_half, bin_y_half],
                    rotation                        = None,                             # Add uniform random rotation
                    rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    reference_pos                   = self.bin1_pos+self.bin_thickness,
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
                    mujoco_objects                  = self.objects,
                    x_range                         = [-bin_x_half, bin_x_half],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                    y_range                         = [-bin_y_half, bin_y_half],
                    rotation                        = None,                             # Add uniform random rotation
                    rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                    ensure_object_boundary_in_range = True,
                    ensure_valid_placement          = True,
                    reference_pos                   = self.bin1_pos+self.bin_thickness,
                    z_offset                        = 0.,
                )
            )

        # placeObjectSamplers: each visual object receives a sampler that places it in the TARGET bin
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name                            = "placeObjectSampler",             # name for object sampler for each object
                mujoco_objects                  = self.visual_objects,
                x_range                         = [-bin_x_half, bin_x_half],        # This (+ve,-ve) range goes from center to the walls on each side of the bin
                y_range                         = [-bin_y_half, -bin_y_half * 0.8],
                rotation                        = None,                             # Add uniform random rotation
                rotation_axis                   = 'z',                              # Currently only accepts one axis. TODO: extend to multiple axes.
                ensure_object_boundary_in_range = True,
                ensure_valid_placement          = True,
                reference_pos                   = self.bin2_pos,
                z_offset                        = 0.20,                             # Set a vertical offset of XXcm above the bin
                z_offset_prob                   = 0.50,                             # probability with which to set the z_offset
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
            reference_pos=self.bin1_pos+self.bin_thickness,
            )

    def _load_model(self):
        """
        Create a manipulation task object. 
        Requires a (i) mujoco arena, (ii) robot (+gripper), and (iii) object + visual objects. 
        
        Return an xml model under self.model
        """
        super()._load_model()

        # Extract hard coded starting pose for your robot
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"] # Access your robot's (./model/robots/manipulator/robot_name.py) base_xpose_offset hardcoded position as dict
        # Place your robot's base at the indicated position. 
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top/bins workspace
        mujoco_arena = BinsArena(
                                bin1_pos        = self.bin1_pos,
                                table_full_size = self.table_full_size,
                                table_friction  = self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size # bin_size is really the area covered by the two bins

        # Create class names (must match those in ./robosuite/models/objects/xml_objects.py ) and call them with the name of the MujocoXMLObject
        self.objects        = []
        self.visual_objects = []

        # Given that self.object_names and self.visual_object_names are available from load, return classes
        self.visual_objects, self.objects = self.extract_obj_classes()

        # insantiate object model: includes arena, robot, and objects of interest. merges them to return a single model. 
        self.model = ManipulationTask(
            mujoco_arena    = mujoco_arena,
            mujoco_robots   = [robot.robot_model for robot in self.robots], 
            mujoco_objects  = self.visual_objects + self.objects,
        )

        # Create placement initializers for each existing object (self.placement_initializer): will place according to strategy
        self._get_placement_initializer()

    # Next 3 methods all handle observables
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.visual_objects + self.objects):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in the target bins
        self.objects_in_target_bin =[]

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
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
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

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

            # Create sensors for objects
            for i, obj in enumerate(self.objects):
                # Create object sensors
                using_obj = True #(self.single_object_mode == 0 or self.object_id == i)
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
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
        HER = np.random.uniform() < 0.50
        # introduce offset between grip site and gripper mount to prevent collision
        offset = 0.03
        # maximum gripping space
        longitude_max = 0.07
        # HER flag for activating HER 100% all the time
        HER = True
        if HER:
            # Rename goal object pos as eef pos, goal object quat
            HER_pos = self._eef_xpos
            HER_quat = obj_quat
            # print("Original object pos is {}".format(HER_pos))
            # print("Original obj_quat is {}".format(HER_quat))
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
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(HER_pos), np.array(HER_quat)]))
            # print("Update HER pos for {} to {}".format(self.goal_object['name'], HER_pos))
            # print("Update HER pose for {} to {}".format(self.goal_object['name'], HER_quat))
        else:
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _reset_internal(self):
        """
        Resets the simulation's internal configuration and object positions and robot eef
        TODO: should use IKs to test if positions are admissible. 

        Upon a hard reset, we will place num_obs_to_load according to object_reset_strategy. 
        [organized, jumbled, wall, random]. 
        
        Where, 
        - 'organized': nicely stacked side-by-side, if no more fit, on-top of each other. left-to-right.
        - 'jumbled':   just dropped in from top at center, let them fall freely.
        - 'wall':      align objects with wall. start on far wall on the left and move clock-wise.
        - 'random':    select above strategy randomly with each new reset.

        Note, that under these strategies we could keep the same objects or change them. This is configured by 
        the object_randomization flag. The latter depends on the num_objs_in_db available in the db, and the num_obs_to_load. 

        # TODO: need to decide when the locations of objects should be updated. if arm does not finish picking everything, do we want to move things around?
        The goal object should also not be changed for this time. Should this only happen in a hard reset?
        """
        global _reset_internal_after_picking_all_objs

        # if we have not finished picking_all_objs, calling _reset_internal will do nothing
        if not _reset_internal_after_picking_all_objs:
            return

        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset: # i.e. stochastic (flag set in base.py)

            if self.object_reset_strategy == 'random':
                self.object_reset_strategy = random.choice(object_reset_strategy_cases[0:3]) # Do not include random in selection

            if self.object_reset_strategy == 'organized':
                pass

            if self.object_reset_strategy == 'jumbled':

                # A) If we want to randomize objects with new reset: select new objects, and update the placement samplers (as done in _load_model). 
                if self.object_randomization == True:

                    # A. Select new objects
                    # Given the available objects, randomly pick num_objs_to_load and return names, visual names, and name_to_id
                    self.object_names, self.visual_object_names, self.object_to_id = self.load_objs_to_simulate(self.num_objs_in_db,self.num_objs_to_load)

                    # B. Extract class names (must match those in ./robosuite/models/objects/xml_objects.py ) and call them with the name of the MujocoXMLObject
                    self.objects        = []
                    self.visual_objects = []

                    # Given that self.object_names and self.visual_object_names are available from load, return classes
                    self.visual_objects, self.objects = self.extract_obj_classes()

                    # C. Update the model's mujoco objects
                    self.model.mujoco_objects  = self.visual_objects + self.objects

                    # Create placement initializer objects for each existing object (self.placement_initializer): will place according to strategy
                    self._get_placement_initializer()   

                #----------------- Continue to update placement of objects (pick, place, eef)
                ##TODO: need to decide WHEN these locs + GOAL should be updated. The main considerations to discuss are:
                # If arm does not finish picking goal object do we want to set a new goal or the same?
                # If all objects are not placed, do we want to move them or keep them in the same place?

                # Sample from the placement initializer for all objects (regular and visual objects)
                self.object_placements = self.placement_initializer.sample()

                # Set goal object to pick up and sort closest objects to model
                self.goal_object, self.other_objs_than_goals = self.get_goal_object()

                #eef_pos = self._eef_xpos
                #print("Testing {}".format(self._observables))
                # # Move eef
                # if hasattr(self, 'robot_placement_initializer'):
                #     robot_placements = self.robot_placement_initializer.sample()
                #     # 2. set pose of robot. (data.ctrl is for joint angles not pose).
                #     for robot in robot_placements.keys():
                #         if self.sim.data.ctrl is not None:
                #             print(f"Starting eef_xpos: {self._eef_xpos}. \nDesired xpos {robot_placements[robot][0][:3]}")
                #             self.sim.data.site_xpos[self.robots[0].eef_site_id] = robot_placements[robot][0][:3]
                #             #self.sim.data_site_
                #             self.sim.data.site_xpos[2] = robot_placements[robot][0][:3]
                #             self.sim.data.site_xpos[3] = robot_placements[robot][0][1]
                #             self.sim.data.site_xpos[4] = robot_placements[robot][0][2]
                #
                #             self.sim.data.set_joint_qpos('robot0_joint1', robot_placements[robot][0][0])
                #             self.sim.data.set_joint_qpos('robot0_joint2', robot_placements[robot][0][1])
                #             self.sim.data.set_joint_qpos('robot0_joint3', robot_placements[robot][0][2])
                #             self.sim.data.set_joint_qpos('robot0_joint4', robot_placements[robot][1][0])
                #             self.sim.data.set_joint_qpos('robot0_joint5', robot_placements[robot][1][1])
                #             self.sim.data.set_joint_qpos('robot0_joint6', robot_placements[robot][1][2])
                #             self.sim.data.set_joint_qpos('robot0_joint7', robot_placements[robot][1][3])
                #
                #             self.sim.data.ctrl[:3]  =  np.asarray(robot_placements[robot][0])   # pos
                #             self.sim.data.ctrl[3:7] =  np.asarray(robot_placements[robot][1])   # quat
                #             self.sim.data.ctrl[7:9] =  np.array([0,0])                          # two fingers
                #
                #         for i in range(10):
                #             self.sim.step()
                #             self._update_observables()
                #     print(f"Updated eef_xpos: {self._eef_xpos}")

                # Available "joint" names = ('robot0_joint1', 'robot0_joint2', 'robot0_joint3',
                # 'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7',
                # 'gripper0_finger_joint1', 'gripper0_finger_joint2'
                # left & right finger
                # self.sim.data.set_joint_qpos('gripper0_finger_joint1', 0.04)
                # self.sim.data.set_joint_qpos('gripper0_finger_joint2', -0.04)

                for obj_pos, obj_quat, obj in self.object_placements.values():
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

            # Robot EEF: TODO not sure this is the right way to move the end-effector. Needs follow-up study.
            # 1. Call robot placement_initializer to sample eef poses above bin1. Ret's dict of robots with (pos,quat,obj)
            #----------------------
            # if hasattr(self,'robot_placement_initializer'):
            #     robot_placements = self.robot_placement_initializer.sample()

            #     # 2. set pose of robot. (data.ctrl is for joint angles not pose).
            #     # for robot in robot_placements.keys():
            #     #     if self.sim.data.ctrl is not None:
            #     #         print(f"Starting eef_xpos: {self._eef_xpos}. \nDesired xpos {robot_placements[robot][0][:3]}")
            #     #         self.sim.data.site_xpos[self.robots[0].eef_site_id] = robot_placements[robot][0][:3]
            #     #         #self.sim.data_site_
            #     #             # self.sim.data.site_xpos[2] = robot_placements[robot][0][:3]
            #     #             # self.sim.data.site_xpos[3] = robot_placements[robot][0][1]
            #     #             # self.sim.data.site_xpos[4] = robot_placements[robot][0][2]

            #     #             # self.sim.data.set_joint_qpos('robot0_joint1', robot_placements[robot][0][0])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint2', robot_placements[robot][0][1])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint3', robot_placements[robot][0][2])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint4', robot_placements[robot][1][0])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint5', robot_placements[robot][1][1])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint6', robot_placements[robot][1][2])
            #     #             # self.sim.data.set_joint_qpos('robot0_joint7', robot_placements[robot][1][3])

            #     #             # self.sim.data.ctrl[:3]  =  np.asarray(robot_placements[robot][0])   # pos
            #     #             # self.sim.data.ctrl[3:7] =  np.asarray(robot_placements[robot][1])   # quat
            #     #             # self.sim.data.ctrl[7:9] =  np.array([0,0])                          # two fingers
                
            #     #     for i in range(10):
            #     #         self.sim.step() 
            #     #         #self._update_observables()
            #     # print(f"Updated eef_xpos: {self._eef_xpos}")
            #-------------------------

            if self.object_reset_strategy == 'wall':

                # A) If we want to randomize objects with new reset: select new objects, and update the placement samplers (as done in _load_model). 
                if self.object_randomization == True:
                    # A. Select new objects
                    # Given the available objects, randomly pick num_objs_to_load and return names, visual names, and name_to_id
                    self.object_names, self.visual_object_names, self.object_to_id = self.load_objs_to_simulate(
                        self.num_objs_in_db, self.num_objs_to_load)

                    # B. Extract class names (must match those in ./robosuite/models/objects/xml_objects.py ) and call them with the name of the MujocoXMLObject
                    self.objects = []
                    self.visual_objects = []

                    # Given that self.object_names and self.visual_object_names are available from load, return classes
                    self.visual_objects, self.objects = self.extract_obj_classes()

                    # C. Update the model's mujoco objects
                    self.model.mujoco_objects = self.visual_objects + self.objects

                    # Create placement initializer objects for each existing object (self.placement_initializer): will place according to strategy
                    self._get_placement_initializer()

                # ----------------- Continue to update placement of objects
                ##TODO: need to decide WHEN these locs + GOAL should be updated. The main considerations to discuss are:
                # If arm does not finish picking goal object do we want to set a new goal or the same?
                # If all objects are not placed, do we want to move them or keep them in the same place?

                # Sample from the placement initializer for all objects (regular and visual objects)
                self.object_placements = self.placement_initializer.sample()

                # Set goal object to pick up and sort closest objects to model
                self.goal_object, self.other_objs_than_goals = self.get_goal_object()

                # Sample from the placement initializer for all objects (regular and visual objects)
                object_placements = self.placement_initializer.sample()

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    # Set the visual object body locations
                    if "v" in obj.name.lower():  # switched "visual" for "v"
                        self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                        self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                    else:
                        # Set the collision object joints (setting pose for obj)
                        self.sim.data.set_joint_qpos(obj.joints[0],
                                                     np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

                # Set the bins to the desired position
                self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
                self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

                # flag to run _reset_internal for the very first time only
                _reset_internal_after_picking_all_objs = False
        return True

    def return_sorted_objs_to_model(self,obs):
        '''
        The goal of this method is to return a dictionary num_objects in length that are closest to self.goal_object ['name'] that we want to model in the graph as nodes
        1) Access all objects that are not the goal
        2) Compute their norm with the goal
        3) Sort them
        4) Only keep the first num_object-1 entries (includes the goal)
        5) Place the goal at the front

        Example:
        : objs_to_load = [goal_obj, obj2, obj3, obj4, obj5] (assumed already sorted)
        : num_objects (to model) = 3

        : sorted_obj_dist = [obj2, obj3] # keep num_objects -1
        : sorted_obj_dist = [goal_obj, obj2, obj3] # place goal in the front

        Params:
            self.goal_object:               class level param indicates the name of the object to pick.
            self.other_objs_than_goals:     class level list with names of objects to model. Recall we may choose to model fewer graph nodes than the total number of objects loaded into the simulation at one instance


        '''
        assert len(self.other_objs_than_goals) > 0
        obj_dist        = {}
        sorted_obj_dist = {}

        # 1) Compute norm between goal_object and objects_to_consider
        for other in self.other_objs_than_goals:

            # Get pos from observables
            val = np.linalg.norm(obs[self.goal_object['name']+'_pos'] - obs[other+'_pos'])
            obj_dist[other] = val


        # 2) Sort the dictionary by norm value
        sorted_obj_dist_tuple = sorted(obj_dist.items(), key = lambda item: item[1])
        sorted_obj_dist = {k: v for k, v in sorted_obj_dist_tuple[:self.num_objects-1]} # notice self.num_objects-1. This indicates the number of obj we wish to model excluding the goal. 

        # 3) Place goal object at the front using OrderedDict move_to_end()
        sorted_obj_dist = OrderedDict(sorted_obj_dist)
        
        sorted_obj_dist[ self.goal_object['name'] ] = 0 # the distance value of goal is zero
        sorted_obj_dist.move_to_end( self.goal_object['name'], last=False) # move to front

        return sorted_obj_dist
    
    def return_fallen_objs(self, obs):
        """
        return list of fallen objs names if lower than table height
        """
        fallen_objs = []

        # for obj_pos, obj_quat, obj in self.object_placements.values():
        for placed_pos , placed_quat, obj in self.object_placements.values():
            # Get real-time pos from observables
            
            # print(obs[obj.name + '_pos'][2])
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj.name]]
            print("can we read obj observations? {}".format(obj_pos))
            # check if obj has fallen below bin
            if obj_pos[2] < self.bin1_pos[2]+self.bin_thickness[2] and obj.name in self.object_names:
                print("new fallen obj !!! {}, pos is {}".format(obj.name, obj_pos))
                fallen_objs.append(obj.name)
                # if fallen obj, remove from list
                print("initially we have {} in object names list".format(self.object_names))
                self.object_names.remove(obj.name)
                print("removed {} now we have {}".format(obj.name, self.object_names))

                # Check whether this is necessary.
                # Also remove from sorted_object list so that it is no longer considered in computing
                # the observations in the next iteration
                if obj.name in self.sorted_objects_to_model:
                    print("initially we have {} in sorted object to model list")
                    self.sorted_objects_to_model.__delitem__(obj.name)

        return fallen_objs

    def _is_success(self, achieved_goal, desdired_goal):
        """
        HER-Specific check success method comparing achieved and desired positions 
        TODO: currently we do not analyze orientation. Test good performance with position only first. 
        TODO: Should also add an additional check to see if the object is in fact touching the fingers. This check is done in standard robosuite and should be integrated here. 

        Currently the achieved_goal (current position of goal object) and desired_goal are numpy arrays with [pos|quat] shape (7,) 

        Check if self.goal_object placed at target location. 
        TODO: another possible thing to do is instead of checking whether one object reached the target, check for all objects in the target, or both.
        
        General structure of method:
            1. check for success test
            2. remove current goal_object form list at next iteration
            3. select new next object_goal 

        Returns:
            bool: True if 1 / all object(s) placed correctly

        TODO: consider modifing the definition of is_success according to QT-OPTs criteria to increase reactivity
        requires reaching a certain height... see paper for more. also connected with one parameter in observations.

        TODO: after succeeding, in any occurrence, move the end-effector to a starting position
        """
        global _reset_internal_after_picking_all_objs
        
        # Subtract obj_pos from goal and compute that error's norm:
        target_dist_error = np.linalg.norm(achieved_goal - desdired_goal)

        if target_dist_error <= self.goal_pos_error_thresh and len(self.object_names) != 0:
            # After successfully placing self.goal_object, remove this from the list of considered names for the next round
            self.object_names.remove(self.goal_object['name'])

            # TODO: double check if the above line is enough, or we aldso need the line below. Also remove from sorted_object list so that it is no longer considered in computing the observations in the next iteration
            self.sorted_objects_to_model.__delitem__(self.goal_object['name'])

            # Get a new object_goal if objs still available
            self.goal_object,_ = self.get_goal_object() 
            print(f"Successful placement. New object goal is {self.goal_object['name']}") 

            # Add the current goal object to the list ob objects in target bins
            self.objects_in_target_bin.append(self.goal_object['name'])
            print("Picked {}". format(self.goal_object['name']))
            return True
        elif len(self.object_names) == 0 and len(self.objects_in_target_bin) == self.num_objs_to_load:
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
        Assumes preamble has loaded objets (oXXXX) and visual objects (oXXXXv). Although check for other possible names as well.

        Given the global set of objs_in_db and the num_objs_to_load, we sample objs to consider and carefully
        look at file names to load. 

        We create lists of regular object names and visual objects as well as a dict of object_to_id. 
        These lists/dict are then used to tell the simulation what to load for modeling. 
        '''

        # Build random list of objects to load. Ideally named o00XX.
        digits   = []
        obj_wo_o = []
        counter  = 1

        object_to_id           = {}
        object_names           = []
        visual_object_names    = []

        objs_to_consider = random.sample( range(num_objs_in_db), num_objs_to_load) #objs_to_consider = [69, 66, 64, 55, 65]
        for idx, val in enumerate(objs_to_consider):

            # Collect all objects whose file name starts with an 'o' and contain 'Object' as in OXXXXObject
            if objs_in_db[ objs_to_consider[idx] ][0] == 'o' and "Object" in objs_in_db[ objs_to_consider[idx] ]:
                digit = objs_in_db[ objs_to_consider[idx] ]
                digits.append(digit)            # Keep list of existing objects
                # Create map name:id
                object_to_id.update({digit:idx+1})             # idx starts from 1 for diff objs: o00X8:1, oOOX3:2, oOOX9:3
                object_names.append(digit)                     # o0001, o0002,...,o0010...
                visual_object_names.append(digit[:5]+'VisualObject')          # o0001VisualObject
            
            # Otherwise keep a list of those that do not
            else:
                obj_wo_o.append(idx)


        # Do a second sweep to deal with objects that do not start with 'o'. Compare with list of registered objects
        for idx in obj_wo_o:
            temp = 'o' + str(counter).zfill(4) + 'Object'
            
            # If this number exists, increment and try again before registering. 
            while temp in digits:
                counter += 1
                temp = 'o' + str(counter).zfill(4) + 'Object'
                
            digit = temp
            digits.append(digit)            

            # Create map name:id
            object_to_id.update({digit:idx})               # idx starts from 1 for diff objs: o00X8:1, oOOX3:2, oOOX9:3
            object_names.append(digit)                     # o0001Object, o0002Object,...,o0010Object...
            visual_object_names.append(digit[:5]+'VisualObject')          # o0001VisualObject

        return (object_names, visual_object_names, object_to_id)

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

        return visual_objects, objects   
        
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

        # Get robosuite observations as an Ordered dict
        obs = self._get_observations(force_update) # if called by reset() [see base class] this will be set to True.

        # Get prefix for robot to extract observation keys
        pf = self.robots[0].robot_model.naming_prefix
        dt        = self.sim.nsubsteps * self.sim.model.opt.timestep  # dt is equivalent to the amount of time across number of substeps. But xvelp is already the velocity in 1 substep. It seems to make more sense to simply scale xvel by the number of substeps in 1 env step.        
        
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

        # Concatenate and place in env_obs (1)
        env_obs = np.concatenate([  # 17 dims
            grip_pos.ravel(),       # 3
            grip_quat.ravel(),      # 4

            grip_velp.ravel(),      # 3
            grip_velr.ravel(),      # 3

            gripper_state.ravel(),  # 2
            gripper_vel.ravel(),    # 2
        ])

        #-------------------------------------------------------------------------- 
        # 01b) Object observations *We do not follow relationalRL here and do not add all objects + grip to achieved goals. Instead just add goal_object
        #-------------------------------------------------------------------------- 

        # Observations for Objects
        # *Note: there are three quantities of interest: (i) (total) num_objs_to_load, (ii) num_objs (to_model), and (iii) goal object. 
        # We report data for num_objs that are closest to goal_object and ignore the rest. This list is updated when is_success is True.
        # We only consider the relative position between the goal object and end-effector, all the rest are set to 0.
        self.sorted_objects_to_model = self.return_sorted_objs_to_model(obs)

        # Place goal object at the front
        self.sorted_objects_to_model

        # TODO: sorted_objects should be updated when an object is successfully picked. Such that when there is one object less, 
        # the new dimensionality is reflected in these observations as well.
        for i in range( len(self.sorted_objects_to_model )):

            name_list = list(self.sorted_objects_to_model)

            # Pose: pos and orientation            
            object_i_pos  = obs[name_list[i] + '_pos'] 
            object_i_quat = obs[name_list[i] + '_quat'] 

            # Vel: linear and angular
            object_velp = obs[name_list[i] +'_velp'] * dt
            object_velp = object_velp - grip_velp # relative velocity between object and gripper

            object_velr = obs[name_list[i] +'_velr'] * dt

            # Relative position wrt to gripper: 
            # *Note: we will only do this for the goal object and set the rest to 0. 
            # By setting to 0 all calculations in the network will be cancelled. Robot should reach only to the goal object.
            # Goal object to be modified if successful (without repeat)
            if i == 0: 
                 object_rel_pos = object_i_pos - grip_pos
                 object_rel_rot = T.quat_distance(object_i_quat,grip_quat) # quat_dist returns the difference
                 
                # 02) Achieved Goal: the achieved state will be the object(s) pose(s) of the goal (1st) object         
                #--------------------------------------------------------------------------
                # TODO: double check if this works effectively for our context + HER. Otherwise can add objects and grip pose.
                #--------------------------------------------------------------------------                                 
                 achieved_goal = np.concatenate([    # 3          # 7                
                    object_i_pos.copy(),    # 3      # Try pos only first.           
                    # object_i_quat.copy(), # 4
                ])

            else:
                object_rel_pos = np.zeros(3)
                object_rel_rot = np.zeros(4)
            
            # Augment observations      Dims:
            env_obs = np.concatenate([  # 17 + (20 * num_objects)
                env_obs,                
                object_i_pos.ravel(),   # 3
                object_i_quat.ravel(),  # 4

                object_velp.ravel(),    # 3
                object_velr.ravel(),    # 3

                object_rel_pos.ravel(), # 3
                object_rel_rot.ravel()  # 4
            ])

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
        desired_goal = []
        desired_goal = np.concatenate([ # 3             # 7
            self.goal_object['pos'],    # 3             # Try pos only first.
            # self.goal_object['quat']    # 4
        ])
        
        # Returns obs, ag, and also dg
        return_dict = {
            'observation':   env_obs.copy(),
            'achieved_goal': achieved_goal.copy(),  # [ag_ob0_xyz, ag_ob1_xyz, ... rob_xyz]
            'desired_goal':  desired_goal.copy(),   # [goal_obj_xyz, goal_obj_quat]

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
            Note: not necessary when we wrap the env with the NormalizedBoxEnv class)

        03 Set data to mujoco sim.data.ctrl

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


        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):

            # 01. sim.forward()
            self.sim.forward()

            # Not necessary to clip actions within robosuite as we wrap with the NormalizedBoxEnv. 
            # -->Set (clipped) action in mujoco
            # action = np.clip(action, 
            #                  self.action_spec[0],
            #                  self.action_spec[1])
            
        
            # 03 Copy action to sim.data.ctrl (no mocaps used currently. differs from FetchEnv step approach)
            self._pre_action(action, policy_step)

            # 04 sim.step
            try:
                self.sim.step()                             # Advance simulation
            
            except mujoco_py.builder.MujocoException as e:
                print(e)
                print(F"action {action}") 

            # 05 Update observables and get new observations
            self._update_observables()
            env_obs = self._get_obs()
            
            policy_step = False
                                           
        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep        

        # 06 Process info
        info = { 'is_success': self._is_success(env_obs['achieved_goal'], env_obs['desired_goal']) }

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

        # 06c Check & remove fallen objs

        fallen_objs = self.return_fallen_objs(obs=env_obs)

        # 07 Process Done: 
        # If (i) time_step is past horizon OR (ii) we have succeeded, set to true.
        done = (self.timestep >= self.horizon) and not self.ignore_done or info['is_success']
    
        # 08 Process Reward
        reward = self.compute_reward(env_obs['achieved_goal'], env_obs['desired_goal'], info)

        return env_obs, reward, done, info       

    def __reduce__(self):
        
    # #     # Return the object’s local name relative to its module; 
    # #     #return "picking_blocks1_numrelblocks3_nqh1_rewardsparse_dictstateObs" #self.__module__
        return 'Picking'
    
    def __getnewargs_ex__(self):
        '''
        The arguments needed to pass in are those used in base.py to create the new meta classes, i.e.
        def __new__(meta, name, bases, class_dict):

        Where, 
        - meta is the MujocoEnv class isntance
        - name is the name of the class, i.e. Picking
        - bases is a tuple with the <class 'robosuite.environments.manipulation.single_arm_env.SingleArmEnv'>
        - classes_dict is a dict with all the class method names and associated method objects
        '''
        args = tuple()
        meta, name, bases = None, None, None
        
        kwargs = {}
        kwargs['meta']  = self
        kwargs['name']  = suite.environments.base.EnvMeta
        kwargs['bases'] = (suite.environments.manipulation.single_arm_env.SingleArmEnv,) #(<class 'robosuite.environments.manipulation.single_arm_env.SingleArmEnv'>,)
        kwargs          =  picking_dict['picking_dict'] # self.__dict__
        return (args,kwargs)
#-------------------------------------------------------------
# Define new permutation of classes to register based on picking for relationalRL code
# *This was my original sol. in following rlkit-relational FetchBlockConstruction. However it breaks, pickle.dumps/loads used in relationalRL. 
# *Moved this to the base.py:MakeEnv and then added a __reduce__ method below to solve a __reduce__ related error, but could not. 
#  For these reasons, currently giving up on registerin different classes. Will just go with 1 class Picking.
#-------------------------------------------------------------

#-------------------------------------------------------------    
# for num_blocks in range(1, 25): # use of num_blocks indicates objects. kept for historical reasons.
#     for num_relational_blocks in [3]: # currently only testin with 3 relational blocks (message passing)
#         for num_query_heads in [1]: # number of query heads (multi-head attention) currently fixed at 1
#             for reward_type in ['incremental','sparse']: #could add sparse
#                 for obs_type in ['dictstate','dictimage','np']: #['dictimage', 'np', 'dictstate']:

#                     # Generate the class name 
#                     className = F"picking_blocks{num_blocks}_numrelblocks{num_relational_blocks}_nqh{num_query_heads}_reward{reward_type}_{obs_type}Obs"

#                     # Add necessary attributes

#                     # Generate the class type using type and set parent class to Picking
#                     pickingReNN = type(className, (Picking,), {}) # args: (i) class name, (ii) tuple of base class, (iii) dictionary of attributes

#                     # Customize the class name
#                     globals()[className] = pickingReNN

