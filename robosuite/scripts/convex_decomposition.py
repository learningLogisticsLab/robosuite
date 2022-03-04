"""
Script to create convex decomposition of a mesh
"""

import pybullet as p
import os

p.connect(p.DIRECT)
name_in = os.path.join("/home/jrojas/Desktop/bar_clamp.obj")
name_out = "/home/jrojas/Desktop/bar_clamp_vhacd.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000 )