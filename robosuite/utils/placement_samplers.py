import collections
import numpy as np

from copy import copy

from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply
from robosuite.models.objects import MujocoObject
from robosuite.models.robots import ManipulatorModel, RobotModel


class ObjectPositionSampler:
    """
    Base class of object placement sampler.

    Args:
        - name (str): Name of this sampler.

        - mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        - ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)

        - ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        - reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        - z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
            self,
            name,
            mujoco_objects                  = None,
            ensure_object_boundary_in_range = True,
            ensure_valid_placement          = True,
            reference_pos                   = (0, 0, 0),
            z_offset                        = 0.,
    ):

        # Setup attributes
        self.name = name

        if mujoco_objects is None:
            self.mujoco_objects = []

        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else copy(mujoco_objects)

        self.ensure_object_boundary_in_range    = ensure_object_boundary_in_range
        self.ensure_valid_placement             = ensure_valid_placement
        self.reference_pos                      = reference_pos
        self.z_offset                           = z_offset

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, "Object '{}' already in sampler!".format(obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object.

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.

        z_offset_prob (float): probability with which to apply the offset. Useful to set goals in the air and table. 
    """

    def __init__(
            self,
            name,
            mujoco_objects=None,
            x_range=(0, 0),
            y_range=(0, 0),
            rotation=None,
            rotation_axis='z',

            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0),

            z_offset=0.,
            z_offset_prob=0.,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.z_offset_prob = z_offset_prob

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range

        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius

        return np.random.uniform(low=minimum, high=maximum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range

        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius

        return np.random.uniform(low=minimum, high=maximum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled (r,p,y) euler angle orientation

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == 'x':
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == 'y':
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == 'z':
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis))

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement for the top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        TODO: does this assume objects are always facing up vs tripped over? It's possible the computations here will fall apart if the objects have a different orientaiton

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)

        if reference is None:
            base_offset = self.reference_pos

        elif type(reference) is str:
            assert reference in placed_objects, "Invalid reference received. Current options are: {}, requested: {}" \
                .format(placed_objects.keys(), reference)

            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)

            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))

        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}" \
                .format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        # print("{} objs of self muj objs {}".format(len(self.mujoco_objects), self.mujoco_objects))
        # print("{} placed objs of placed obj {}".format(len(placed_objects), placed_objects))
        for obj in self.mujoco_objects:
            # print(obj.name)
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(100):  # 5000 retries
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]

                # Place the object in the air with z_offset with probability z_offset_prob
                if np.random.uniform() < self.z_offset_prob:
                    object_z = self.z_offset + base_offset[2] + obj.vertical_radius / 2
                else:
                    object_z = base_offset[2] + obj.vertical_radius / 2

                if on_top:
                    object_z -= bottom_offset[-1]  # subtract the negative bottom_offset equal to adding the top offset.

                # objects cannot overlap
                location_valid = True

                # Once an xyz is computed for the object, make sure it does not collide with other objects. 
                # TODO: does this assume objects are always facing up vs tripped over?
                #  It's possible the computations here will fall apart if the objects have a different orientaiton
                location_valid, success, placed_objects = self.recheck_validity_pos(location_valid=location_valid,
                                                                                    success=success,
                                                                                    placed_objects=placed_objects,
                                                                                    object_x=object_x,
                                                                                    object_y=object_y,
                                                                                    object_z=object_z,
                                                                                    obj=obj)
                if location_valid:
                    break
                else:
                    # We cannot find a good location, so raise it by 10cm the air and drop it.
                    object_z += 0.10
                    quat = self._sample_quat()
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    # TODO: recheck validity of this position. Convert code segment starting with if self.ensure_valid_placement into a function and call here.
                    location_valid, success, placed_objects = self.recheck_validity_pos(location_valid=location_valid,
                                                                                        success=success,
                                                                                        placed_objects=placed_objects,
                                                                                        object_x=object_x,
                                                                                        object_y=object_y,
                                                                                        object_z=object_z,
                                                                                        obj=obj)

        if not success:
            print("cannot place all objs")
            # raise RandomizationError("Cannot place all objects ):")
        # print("{} placed objs are {}".format(len(placed_objects.keys()), placed_objects.keys()))
        # print("{} placed vis objs are {}")
        return placed_objects

    def recheck_validity_pos(self, location_valid, success, placed_objects, object_x, object_y, object_z, obj):
        """
        check validity of the position
        requirements:
        location_valid
        placed_objects
        object_x
        object_y
        object_z
        obj.horizontal_radius
        obj.bottom_offset
        """
        if self.ensure_valid_placement:
            for (x, y, z), _, other_obj in placed_objects.values():
                # if (
                #         np.linalg.norm((object_x - x,
                #                         object_y - y))  # Compute the norm between current object and each of the other objects
                #         <= max(other_obj.horizontal_radius, other_obj.vertical_radius/2) + max(horizontal_radius, vertical_radius/2)
                # # If the norm is less than the sum of the horizontal radius of both objects it means collision
                # ) and (
                #         object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]  # ??
                # ):
                if (
                        np.linalg.norm((object_x - x,
                                        object_y - y,
                                        object_z - z))  # Compute the norm between current object and each of the other objects
                        <= max(other_obj.horizontal_radius, other_obj.vertical_radius / 2) + max(
                    obj.horizontal_radius, obj.vertical_radius / 2)
                        # If the norm is less than the sum of the horizontal radius of both objects it means collision
                ):
                    location_valid = False
                    # if location_valid is False:
                    # print(f'Could find a location to place object {obj.name} without collision in the placement_sampler')
                    # print(f'position is ({object_x},{object_y},{object_z}), norm is {np.linalg.norm((object_x - x, object_y - y))}, and the sum of horizontal raidii between the two objects is {other_obj.horizontal_radius + horizontal_radius}')
                    break

            if location_valid:
                # random rotation
                quat = self._sample_quat()

                # multiply this quat by the object's initial rotation if it has the attribute specified
                if hasattr(obj, "init_quat"):
                    quat = quat_multiply(quat, obj.init_quat)

                # location is valid, put the object down
                pos = (object_x, object_y, object_z)
                placed_objects[obj.name] = (pos, quat, obj)
                success = True

        return location_valid, success, placed_objects


class UniformWallSampler(ObjectPositionSampler):
    """
    Let objects hug walls from left to right (clockwise), going to divide each bin to 4 walls.
    Might randomize placement in each walls later.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x[i] (2-array of float): Specify the x pos for ith wall

        y[i] (2-array of float): Specify the y pos for ith wall

        rotation (None or float or Iterable):
            :`None`: Add uniform random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
            self,
            name,
            mujoco_objects=None,
            x_range=(0, 0),
            y_range=(0, 0),
            rotation=None,
            rotation_axis='z',
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0),
            z_offset=0.,
            z_offset_prob=0.,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.z_offset_prob = z_offset_prob

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, count, object_horizontal_radius):
        """
        Samples the x location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range

        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius

        if (count == 0) or (count == 2):  # wall 0 or wall 2
            return np.random.uniform(low=minimum, high=maximum)
        elif (count == 1):  # wall 1
            return minimum
        else:  # wall 3
            return maximum

    def _sample_y(self, count, object_horizontal_radius):
        """
        Samples the y location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range

        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius

        if count == 0:  # wall 0
            return minimum
        elif count == 2:  # wall 3
            return maximum
        else:  # (count == 1) or (count == 2) , wall 1 or wall 3
            return np.random.uniform(low=minimum, high=maximum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled (r,p,y) euler angle orientation

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == 'x':
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == 'y':
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == 'z':
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis))

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement for the top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        TODO: does this assume objects are always facing up vs tripped over? It's possible the computations here will fall apart if the objects have a different orientaiton

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)

        if reference is None:
            base_offset = self.reference_pos

        elif type(reference) is str:
            assert reference in placed_objects, "Invalid reference received. Current options are: {}, requested: {}" \
                .format(placed_objects.keys(), reference)

            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)

            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))

        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}" \
                .format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        count = 0  # create counter
        for obj in self.mujoco_objects:

            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False

            for i in range(100):  # 5000 retries
                object_x = self._sample_x(count, horizontal_radius) + base_offset[0]
                object_y = self._sample_y(count, horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]

                # Place the object in the air with z_offset with probability
                if np.random.uniform() < self.z_offset_prob:
                    object_z = self.z_offset + base_offset[2]

                if on_top:
                    object_z -= bottom_offset[
                        -1]  # subtract the negative bottom_offset equal to adding the top offset.

                # objects cannot overlap
                location_valid = True

                # Once an xyz is computed for the object, make sure it does not collide with other objects.
                # TODO: does this assume objects are always facing up vs tripped over? It's possible the computations here will fall apart if the objects have a different orientaiton
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                                np.linalg.norm((object_x - x,
                                                object_y - y))  # Compute the norm between current object and each of the other objects
                                <= other_obj.horizontal_radius + horizontal_radius
                                # If the norm is less than the sum of the horizontal radius of both objects it means collision
                        ) and (
                                object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]  # ??
                        ):
                            location_valid = False
                            # if location_valid is False:
                            # print(f'Could find a location to place object {obj.name} without collision in the placement_sampler')
                            # print(f'position is ({object_x},{object_y},{object_z}), norm is {np.linalg.norm((object_x - x, object_y - y))}, and the sum of horizontal raidii between the two objects is {other_obj.horizontal_radius + horizontal_radius}')
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                # We cannot find a good location, so raise it by 10cm the air and drop it.
                object_z += 0.10
                quat = self._sample_quat()
                pos = (object_x, object_y, object_z)
                placed_objects[obj.name] = (pos, quat, obj)
                # TODO: recheck validity of this position. Convert code segment starting with if self.ensure_valid_placement into a function and call here.
                # if it failes then raise RandomizationError
                # raise RandomizationError("Cannot place all objects ):")
            count += 1  # add count, count = (0,2) = place in wall 1, wall 3 , otherwise wall 2, wall 4
            count %= 4
        return placed_objects


class robotUniformRandomSampler:  # UniformRandomSampler):
    '''
    Place robot eef within the table in a randomly uniform manner. 

    Args:
        name (str): Name of this sampler.

        mujoco_robots (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        z_range (2-array of float): Specify the (min, max) relative z_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation 

        reference_pos
    '''

    def __init__(
            self,
            name,
            mujoco_robots=None,
            x_range=(0, 0),
            y_range=(0, 0),
            z_range=(0, 0,),
            rotation=None,
            rotation_axis='z',
            reference_pos=(0, 0, 0),
    ):

        self.name = name,
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.reference_pos = reference_pos

        if mujoco_robots is None:
            self.mujoco_robots = []

        else:
            self.mujoco_robots = [robot if isinstance(robot, ManipulatorModel) else copy(mujoco_robots) for robot in
                                  mujoco_robots]

    def _sample_x(self):
        """
        Samples the x position of eef

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        return np.random.uniform(low=minimum, high=maximum)

    def _sample_y(self):
        """
        Samples the y location of eef

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        return np.random.uniform(low=minimum, high=maximum)

    def _sample_z(self):
        """
        Samples the z location of eef

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.z_range
        return np.random.uniform(low=minimum, high=maximum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled (r,p,y) euler angle orientation

        TODO: could make this more advanced and consider more than 1 axis at a time.

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == 'x':
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == 'y':
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == 'z':
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis))

    def sample(self, reference=None):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

        Return:
            dict: dictionary of all eef placements, mapping mujoco_robot eef's to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all eef's]
            AssertionError: [Reference mujoco_robot name does not exist, invalid inputs]
        """
        # Standardize inputs

        placed_robots = {}
        location_valid = False

        if self.reference_pos is None:
            base_offset = np.array([0, 0, 0])

        else:
            base_offset = np.array(self.reference_pos)
            assert base_offset.shape[0] == 3, "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}" \
                .format(base_offset)

        # Sample pos and quat for all robots assigned to this sampler
        for robot in self.mujoco_robots:

            # First make sure the currently sampled object hasn't already been sampled
            assert robot.name not in placed_robots, "Robot '{}' has already been sampled!".format(robot.name)

            for i in range(5000):  # 5000 retries
                robot_x = self._sample_x() + base_offset[0]
                robot_y = self._sample_y() + base_offset[1]
                robot_z = self._sample_z() + base_offset[2]

                # Check that the results are within the bounds of the box
                if (robot_x <= self.x_range[1] + base_offset[0] and robot_x >= self.x_range[0] + base_offset[0] and
                        robot_y <= self.y_range[1] + base_offset[1] and robot_y >= self.y_range[0] + base_offset[1] and
                        robot_z <= self.z_range[1] + base_offset[2] and robot_z >= self.z_range[0] + base_offset[2]):
                    location_valid = True

                # Now compute the orientaiton
                if location_valid:
                    quat = self._sample_quat()

                    # # multiply this quat by the object's initial rotation if it has the attribute specified
                    # if hasattr(obj, "init_quat"):
                    #     quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (robot_x, robot_y, robot_z)
                    placed_robots[robot.name] = (pos, quat, robot)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot set all robot's eef):")

        return placed_robots


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.

    Args:
        name (str): Name of this sampler.
    """

    def __init__(self, name):
        # Samplers / args will be filled in later
        self.samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()

        super().__init__(name=name)

    def append_sampler(self, sampler, sample_args=None):
        """
        Adds a new placement initializer with corresponding @sampler and arguments

        Args:
            sampler (ObjectPositionSampler): sampler to add
            sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
                call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)

        Raises:
            AssertionError: [Object name in samplers]
        """
        # Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
        # if sampler.mujoco_objects is not None:
        for obj in sampler.mujoco_objects:
            assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)

        # # Verify that all added mujoco robots haven't already been added, and add to this sampler's robot dict            
        # if sampler.mujoco_robots is not None:
        #     for robot in sampler.mujoco_robots:
        #         assert robot not in self.mujoco_robots, f"Robot '{robot.name}' already has a sampler associated with it!"
        #         self.mujoco_robots.append(robot)

        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

    def hide(self, mujoco_objects):
        """
        Helper method to remove an object from the workspace.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            mujoco_objects=mujoco_objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis='z',
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.append_sampler(sampler=sampler)

    def add_objects(self, mujoco_objects):
        """
        Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
        """
        raise AttributeError("add_objects() should not be called for SequentialCompsiteSamplers!")

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.

        Args:
            sampler_name (str): Existing sub-sampler name
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
        """
        # First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
        mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # Make sure sampler_name exists
        assert sampler_name in self.samplers.keys(), "Invalid sub-sampler specified, valid options are: {}, " \
                                                     "requested: {}".format(self.samplers.keys(), sampler_name)
        # Add the mujoco objects to the requested sub-sampler
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
                sampler's @reference argument if not already specified. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This will override each
                sampler's @on_top argument if not already specified. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)

        # Iterate through all samplers to sample
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):

            # Pre-process sampler args {'reference', 'on_otp')
            if s_args is None:
                s_args = {}

            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg

            # Run sampler: get (pos,quat,name) for each object
            new_placements = sampler.sample(fixtures=placed_objects, **s_args)

            # Update placements
            placed_objects.update(new_placements)

        return placed_objects
