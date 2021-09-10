import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class BinsArena(Arena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        bin1_pos (3-tuple): (x,y,z) position to place bin1
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self,
        # self, bin1_pos  = (0.1, -0.5, 0.8),
        table_full_size = (0.39, 0.49, 0.82), 
        # table_friction  = (1, 0.005, 0.0001)
    ):
        super().__init__(xml_path_completion("arenas/bins_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        # self.table_friction  = table_friction

        self.bin1_body      = self.worldbody.find("./body[@name='bin1']")
        self.bin2_body      = self.worldbody.find("./body[@name='bin2']")
        self.bin1_geom      = self.worldbody.find("./body/geom[@name='bin1_surface']")
        self.bin2_geom      = self.worldbody.find("./body/geom[@name='bin2_surface']")

        self.bin1_pos       = string_to_array(self.bin1_body.get("pos"))
        self.bin2_pos       = string_to_array(self.bin2_body.get("pos"))
        self.bin1_surface   = [0, 0, string_to_array(self.bin1_geom.get("size"))[2]]
        self.bin2_surface   = [0, 0, string_to_array(self.bin1_geom.get("size"))[2]]
        self.table_top_abs  = self.bin1_pos
        self.bin1_friction  = string_to_array(self.bin1_geom.get("friction"))
        self.bin2_friction  = string_to_array(self.bin2_geom.get("friction"))

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))
