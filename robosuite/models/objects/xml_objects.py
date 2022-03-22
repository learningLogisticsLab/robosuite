import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/bottle.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/can.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class MasterChefCanObject(MujocoXMLObject):
    """
    MasterChef Coffee can object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/master_chef_can.xml"),
                         name=name,
                         joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/lemon.xml"),
                         name=name, obj_type="all", duplicate_collision_geoms=True)


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/milk.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/bread.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/cereal.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/square-nut.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/round-nut.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/milk-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/bread-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/can-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class MasterChefCanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of masterchef can

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/master_chef_can_visual.xml"),
                         name=name,
                         joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(xml_path_completion(xml_path),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle"
        })
        return dic


# BinPicking objects and visuals (choosing to append just a 'v' for ease)
# Defs: https://github.com/learningLogisticsLab/binPicking/blob/main/docs/simulation/objects/object_categories_db.xlsx


class o0001Object(MujocoXMLObject):
    """
    o0001Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0001.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0001VisualObject(MujocoXMLObject):
    """
    o0001ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0001v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0002Object(MujocoXMLObject):
    """
    o0002Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0002.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0002VisualObject(MujocoXMLObject):
    """
    o0002ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0002v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0003Object(MujocoXMLObject):
    """
    o0003Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0003.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0003VisualObject(MujocoXMLObject):
    """
    o0003ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0003v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0004Object(MujocoXMLObject):
    """
    o0004Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0004.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0004VisualObject(MujocoXMLObject):
    """
    o0004ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0004v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0005Object(MujocoXMLObject):
    """
    o0005Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0005.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0005VisualObject(MujocoXMLObject):
    """
    o0005ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0005v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0006Object(MujocoXMLObject):
    """
    o0006Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0006.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0006VisualObject(MujocoXMLObject):
    """
    o0006ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0006v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0007Object(MujocoXMLObject):
    """
    o0007Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0007.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0007VisualObject(MujocoXMLObject):
    """
    o0007ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0007v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0008Object(MujocoXMLObject):
    """
    o0008Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0008.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0008VisualObject(MujocoXMLObject):
    """
    o0008ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0008v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0009Object(MujocoXMLObject):
    """
    o0009Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0009.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0009VisualObject(MujocoXMLObject):
    """
    o0009ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0009v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0010Object(MujocoXMLObject):
    """
    o0010Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0010.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0010VisualObject(MujocoXMLObject):
    """
    o0010ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0010v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0011Object(MujocoXMLObject):
    """
    o0011Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0011.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0011VisualObject(MujocoXMLObject):
    """
    o0011ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0011v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0012Object(MujocoXMLObject):
    """
    o0012Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0012.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0012VisualObject(MujocoXMLObject):
    """
    o0012ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0012v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0013Object(MujocoXMLObject):
    """
    o0013Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0013.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0013VisualObject(MujocoXMLObject):
    """
    o0013ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0013v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0014Object(MujocoXMLObject):
    """
    o0014Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0014.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0014VisualObject(MujocoXMLObject):
    """
    o0014ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0014v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0015Object(MujocoXMLObject):
    """
    o0015Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0015.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0015VisualObject(MujocoXMLObject):
    """
    o0015ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0015v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0016Object(MujocoXMLObject):
    """
    o0016Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0016.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0016VisualObject(MujocoXMLObject):
    """
    o0016ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0016v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0017Object(MujocoXMLObject):
    """
    o0017Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0017.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0017VisualObject(MujocoXMLObject):
    """
    o0017ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0017v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0018Object(MujocoXMLObject):
    """
    o0018Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0018.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0018VisualObject(MujocoXMLObject):
    """
    o0018ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0018v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0019Object(MujocoXMLObject):
    """
    o0019Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0019.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0019VisualObject(MujocoXMLObject):
    """
    o0019ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0019v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0020Object(MujocoXMLObject):
    """
    o0020Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0020.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0020VisualObject(MujocoXMLObject):
    """
    o0020ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0020v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0021Object(MujocoXMLObject):
    """
    o0021Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0021.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0021VisualObject(MujocoXMLObject):
    """
    o0021ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0021v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0022Object(MujocoXMLObject):
    """
    o0022Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0022.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0022VisualObject(MujocoXMLObject):
    """
    o0022ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0022v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0023Object(MujocoXMLObject):
    """
    o0023Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0023.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0023VisualObject(MujocoXMLObject):
    """
    o0023ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0023v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0024Object(MujocoXMLObject):
    """
    o0024Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0024.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0024VisualObject(MujocoXMLObject):
    """
    o0024ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0024v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0025Object(MujocoXMLObject):
    """
    o0025Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0025.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0025VisualObject(MujocoXMLObject):
    """
    o0025ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0025v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0026Object(MujocoXMLObject):
    """
    o0026Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0026.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0026VisualObject(MujocoXMLObject):
    """
    o0026ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0026v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0027Object(MujocoXMLObject):
    """
    o0027Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0027.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0027VisualObject(MujocoXMLObject):
    """
    o0027ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0027v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0028Object(MujocoXMLObject):
    """
    o0028Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0028.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0028VisualObject(MujocoXMLObject):
    """
    o0028ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0028v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0029Object(MujocoXMLObject):
    """
    o0029Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0029.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0029VisualObject(MujocoXMLObject):
    """
    o0029ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0029v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0030Object(MujocoXMLObject):
    """
    o0030Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0030.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0030VisualObject(MujocoXMLObject):
    """
    o0030ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0030v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0031Object(MujocoXMLObject):
    """
    o0031Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0031.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0031VisualObject(MujocoXMLObject):
    """
    o0031ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0031v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0032Object(MujocoXMLObject):
    """
    o0032Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0032.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0032VisualObject(MujocoXMLObject):
    """
    o0032ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0032v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0033Object(MujocoXMLObject):
    """
    o0033Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0033.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0033VisualObject(MujocoXMLObject):
    """
    o0033ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0033v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0034Object(MujocoXMLObject):
    """
    o0034Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0034.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0034VisualObject(MujocoXMLObject):
    """
    o0034ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0034v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0035Object(MujocoXMLObject):
    """
    o0035Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0035.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0035VisualObject(MujocoXMLObject):
    """
    o0035ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0035v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0036Object(MujocoXMLObject):
    """
    o0036Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0036.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0036VisualObject(MujocoXMLObject):
    """
    o0036ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0036v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0037Object(MujocoXMLObject):
    """
    o0037Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0037.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0037VisualObject(MujocoXMLObject):
    """
    o0037ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0037v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0038Object(MujocoXMLObject):
    """
    o0038Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0038.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0038VisualObject(MujocoXMLObject):
    """
    o0038ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0038v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0039Object(MujocoXMLObject):
    """
    o0039Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0039.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0039VisualObject(MujocoXMLObject):
    """
    o0039ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0039v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0040Object(MujocoXMLObject):
    """
    o0040Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0040.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0040VisualObject(MujocoXMLObject):
    """
    o0040ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0040v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0041Object(MujocoXMLObject):
    """
    o0041Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0041.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0041VisualObject(MujocoXMLObject):
    """
    o0041ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0041v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0042Object(MujocoXMLObject):
    """
    o0042Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0042.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0042VisualObject(MujocoXMLObject):
    """
    o0042ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0042v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0043Object(MujocoXMLObject):
    """
    o0043Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0043.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0043VisualObject(MujocoXMLObject):
    """
    o0043ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0043v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0044Object(MujocoXMLObject):
    """
    o0044Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0044.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0044VisualObject(MujocoXMLObject):
    """
    o0044ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0044v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0045Object(MujocoXMLObject):
    """
    o0045Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0045.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0045VisualObject(MujocoXMLObject):
    """
    o0045ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0045v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0046Object(MujocoXMLObject):
    """
    o0046Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0046.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0046VisualObject(MujocoXMLObject):
    """
    o0046ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0046v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0047Object(MujocoXMLObject):
    """
    o0047Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0047.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0047VisualObject(MujocoXMLObject):
    """
    o0047ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0047v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0048Object(MujocoXMLObject):
    """
    o0048Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0048.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0048VisualObject(MujocoXMLObject):
    """
    o0048ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0048v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0049Object(MujocoXMLObject):
    """
    o0049Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0049.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0049VisualObject(MujocoXMLObject):
    """
    o0049ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0049v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0050Object(MujocoXMLObject):
    """
    o0050Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0050.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0050VisualObject(MujocoXMLObject):
    """
    o0050ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0050v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0051Object(MujocoXMLObject):
    """
    o0051Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0051.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0051VisualObject(MujocoXMLObject):
    """
    o0051ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0051v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0052Object(MujocoXMLObject):
    """
    o0052Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0052.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0052VisualObject(MujocoXMLObject):
    """
    o0052ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0052v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0053Object(MujocoXMLObject):
    """
    o0053Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0053.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0053VisualObject(MujocoXMLObject):
    """
    o0053ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0053v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0054Object(MujocoXMLObject):
    """
    o0054Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0054.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0054VisualObject(MujocoXMLObject):
    """
    o0054ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0054v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0055Object(MujocoXMLObject):
    """
    o0055Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0055.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0055VisualObject(MujocoXMLObject):
    """
    o0055ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0055v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0056Object(MujocoXMLObject):
    """
    o0056Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0056.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0056VisualObject(MujocoXMLObject):
    """
    o0056ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0056v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0057Object(MujocoXMLObject):
    """
    o0057Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0057.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0057VisualObject(MujocoXMLObject):
    """
    o0057ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0057v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0058Object(MujocoXMLObject):
    """
    o0058Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0058.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0058VisualObject(MujocoXMLObject):
    """
    o0058ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0058v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0059Object(MujocoXMLObject):
    """
    o0059Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0059.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0059VisualObject(MujocoXMLObject):
    """
    o0059ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0059v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0060Object(MujocoXMLObject):
    """
    o0060Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0060.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0060VisualObject(MujocoXMLObject):
    """
    o0060ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0060v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0061Object(MujocoXMLObject):
    """
    o0061Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0061.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0061VisualObject(MujocoXMLObject):
    """
    o0061ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0061v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0062Object(MujocoXMLObject):
    """
    o0062Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0062.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0062VisualObject(MujocoXMLObject):
    """
    o0062ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0062v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0063Object(MujocoXMLObject):
    """
    o0063Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0063.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0063VisualObject(MujocoXMLObject):
    """
    o0063ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0063v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0064Object(MujocoXMLObject):
    """
    o0064Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0064.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0064VisualObject(MujocoXMLObject):
    """
    o0064ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0064v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0065Object(MujocoXMLObject):
    """
    o0065Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0065.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0065VisualObject(MujocoXMLObject):
    """
    o0065ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0065v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0066Object(MujocoXMLObject):
    """
    o0066Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0066.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0066VisualObject(MujocoXMLObject):
    """
    o0066ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0066v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0067Object(MujocoXMLObject):
    """
    o0067Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0067.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0067VisualObject(MujocoXMLObject):
    """
    o0067ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0067v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0068Object(MujocoXMLObject):
    """
    o0068Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0068.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0068VisualObject(MujocoXMLObject):
    """
    o0068ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0068v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)


class o0069Object(MujocoXMLObject):
    """
    o0069Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0069.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0069VisualObject(MujocoXMLObject):
    """
    o0069ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0069v.xml"),
                         name=name, joints=None,
                         obj_type="visual", duplicate_collision_geoms=True)
        
        
class o0070Object(MujocoXMLObject):
    """
    o0070Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0070.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)
        
        
class o0070VisualObject(MujocoXMLObject):
    """
    o0070ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0070v.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0071Object(MujocoXMLObject):
    """
    o0071Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0071.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0071VisualObject(MujocoXMLObject):
    """
    o0071ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0071v.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)
        

class o0072Object(MujocoXMLObject):
    """
    o0072Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0072.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class o0072VisualObject(MujocoXMLObject):
    """
    o0072ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0072v.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class o0192Object(MujocoXMLObject):
    """
    o0192Object
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0192.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class o0192VisualObject(MujocoXMLObject):
    """
    o0192ObjectVisual
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("./objects/o0192v.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

