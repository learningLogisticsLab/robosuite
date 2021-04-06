# 1. Create the world
# robosuite.models defines arenas|grippers|mounts|objects|robots|tasks
from robosuite.models import MujocoWorldBase
from robosuite.models import grippers
world = MujocoWorldBase()

# 2. Add the Robot
from robosuite.models.robots import Panda
mujoco_robot = Panda()

mujoco_robot.set_base_xpos([0, 0, 0])   # xyz
mujoco_robot.set_base_ori([0, 0, 0 ])   # rpy
world.merge(mujoco_robot)               # takes xml/list of xmls
#------------------------------------------------------------------------

# 3. Add gripper: (1) create gripper instance and (2) add it to robot
from robosuite.models.grippers import gripper_factory, gripper_model

#gripper = gripper_factory('PandaGripper')
#RethinkGripper',    'PandaGripper', 'JacoThreeFingerGripper', 'JacoThreeFingerDexterousGripper', 'WipingGripper', 'Robotiq85Gripper', 'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 'RobotiqThreeFingerDexterousGripper',
# gripper = gripper_factory('PandaGripper')
# gripper.hide_visualization()
# mujoco_robot.add_gripper(gripper)

#------------------------------------------------------------------------
# 4. Create Arena: table

# Initialize the TableArena instance taht creates a table and floorplane
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

#------------------------------------------------------------------------

# 4. Add a MujocoObject
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

# Create ball
sphere = BallObject(
                name = "sphere",
                size = [0.04],
                rgba = [0,0.5, 0.5,1]).get_obj()

sphere.set('pos','1.0 0 1.0') 
world.worldbody.append(sphere)

# Add a free joint to enable movement [deprecated: ball object already comes with a free joint by default]
#sphere.append(new_joint(name='sphere_free_joint', type='free'))

#------------------------------------------------------------------------

# 5. Run simulation

model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer
sim = MjSim(model)

# view it
viewer = MjViewer(sim) # creates viewing window
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()