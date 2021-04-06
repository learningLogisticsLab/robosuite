from robosuite.models import MujocoWorldBase
from robosuite.models import grippers
world = MujocoWorldBase()

#------------------------------------------------------------------------
from robosuite.models.robots import Panda
mujoco_robot = Panda()

mujoco_robot.set_base_xpos([0, 0, 0])   # xyz
mujoco_robot.set_base_ori([0, 0, 0 ])   # rpy
world.merge(mujoco_robot)               # takes xml/list of xmls
#------------------------------------------------------------------------
from robosuite.models.grippers import gripper_factory, gripper_model

#------------------------------------------------------------------------
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

#------------------------------------------------------------------------
# must be xml.etree


#------------------------------------------------------------------------
# model, simulate, view
model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer
sim = MjSim(model)
viewer = MjViewer(sim) 
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()