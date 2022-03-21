"""
Script to save convex decomposition as individual stl files.
"""

FILE_PATH = "/home/jrojas/Desktop/bar_clamp_vhacd.obj"

SAVE_DIR_PATH = "/home/jrojas/Desktop/bar_clamp_all_meshes/"

# convert single obj to multiple objs
file = open(FILE_PATH, 'r')
content = file.read()
mesh_list = content.split('\no')

count = 0
for item in mesh_list:
    count += 1
    new_filename = SAVE_DIR_PATH +'{}.obj'.format(count)
    with open(new_filename, 'w') as f_out:
        f_out.write('{}\n'.format(item))



# convert each obj to stl
import pymeshlab
import os
from os import walk
ms = pymeshlab.MeshSet()

filenames = next(walk(SAVE_DIR_PATH), (None, None, []))[2]  # [] if no file

for f in filenames:
    ms.load_new_mesh(SAVE_DIR_PATH+f)
    ms.generate_convex_hull()
    ms.save_current_mesh(SAVE_DIR_PATH+f[:-4]+'.stl')
    os.remove(SAVE_DIR_PATH+f)


print("Done. Files saved.")


