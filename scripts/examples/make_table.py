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
# 3. Create Arena: table

# Initialize the TableArena instance taht creates a table and floorplane
from robosuite.models.arenas import TableArena
#mujoco_arena = TableArena()
#mujoco_arena = TableArena(table_full_size=(0.4, 0.8, 0.05))
#mujoco_arena = TableArena(table_full_size=(0.4, 0.8, 0.05),has_legs=False)
mujoco_arena = TableArena(table_full_size=(0.4, 0.8, 0.05),table_offset=(0.4, 0, 0.1))
mujoco_arena.set_origin([0.4, 0, 0])
world.merge(mujoco_arena)

#------------------------------------------------------------------------

# 4. Run simulation
model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer
sim = MjSim(model)

# view it
viewer = MjViewer(sim) # creates viewing window
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(1000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()
    #   input()