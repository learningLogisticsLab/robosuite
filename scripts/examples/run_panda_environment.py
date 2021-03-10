import numpy as np
import robosuite as suite

# 02/2021
# Robots:       ['Baxter', 'IIWA', 'Jaco', 'Kinova3', 'Panda', 'Sawyer', 'UR5e']
# ---
# Controllers:  ['IK_POSE','JOINT_POSITION','JOINT_TORQUE',
#                'JOINT_VELOCITY','OSC_POSE','OSC_POSITION']
# ---
# Grippers:     ['RethinkGripper', 'PandaGripper', 'JacoThreeFingerGripper', 
#                'JacoThreeFingerDexterousGripper', 'WipingGripper', 'Robotiq85Gripper', 
#                'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 
#                'RobotiqThreeFingerDexterousGripper', None]
# ---
# Envs 
# ['Door',
#  'Lift',
#  'NutAssembly',
#  'NutAssemblyRound',
#  'NutAssemblySingle',
#  'NutAssemblySquare',
#  'PickPlace',
#  'PickPlaceBread',
#  'PickPlaceCan',
#  'PickPlaceCereal',
#  'PickPlaceMilk',
#  'PickPlaceSingle',
#  'Stack',
#  'TwoArmHandover',
#  'TwoArmLift',
#  'TwoArmPegInHole',
#  'Wipe']


# create environment instance
env = suite.make(
                env_name                = "PickPlace",
                robots                  = "Panda",
                has_renderer            = True,
                has_offscreen_renderer  = False,
                use_camera_obs          = False,
)

# reset the environment
env.reset()

# Sample random actions
for i in range(1000):
    action = np.random.randn(env.robots[0].dof)
    obs, reward, done, info = env.step(action)
    env.render()