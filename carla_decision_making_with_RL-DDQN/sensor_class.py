# import sys
#
# tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
#        '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
# for index in range(0, len(tmp)):
#     if tmp[index] in sys.path:
#         sys.path.remove(tmp[index])
#
# if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES')
#
# if not ('/home/zz/Downloads/CARLA_0.9.6/HDMaps' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/HDMaps')
# if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla')

import carla
# from carla import ColorConverter as cc
import numpy as np
import pygame
import weakref
import collections
import math
import cv2
import queue

class CarlaSyncMode(object): #센서들 간에 동기화
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put) #??? q.put은 머냐 ->
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

class RGBSensor(object):
    def __init__(self, parent_actor,hud):
        self.sensor = None
        self.surface = None
        self.image = None
        self.hud= hud
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(z=50),carla.Rotation(pitch=-90)),
            carla.Transform(carla.Location(z=100),carla.Rotation(pitch=-90))]
        self.transform_index = 1
        self.transform = self._camera_transforms[self.transform_index]
        self.parent = parent_actor
        self.world = self.parent.get_world() #parent actor가 속한 actor
        self.bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.bp.set_attribute('image_size_x',str(self.hud.dim[0]))
        self.bp.set_attribute('image_size_y',str(self.hud.dim[1]))
        self.bp.set_attribute('fov',str(90))
        self.setup_camera()

    def setup_camera(self):
        self.sensor = self.world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))

    def render(self,display):
        if self.surface is not None:
            # print(self.surface)
            display.blit(self.surface, (0, 0))

    def toggle_camera(self):
        self.transform_index= (self.transform_index+1) % len(self._camera_transforms)
        # weak_self = weakref.ref(self)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    @staticmethod
    def _parse_image(weak_self,img):
        self=weak_self()
        self.image = img
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1] #역순으로 전개
            # print(array)
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # array.swapaxes(0, 1)=행렬전치
            # cv2.namedWindow('RGB_imge')
            # cv2.imshow("RGB_image", array)
            # cv2.waitkey()

class DepthCamera(object):
    def __init__(self,parent_actor,hud):
        self.sensor = None
        self.image = None
        self.display = cv2.namedWindow('depth_image')
        self.width = 800#hud.dim[0]
        self.height = 600#hud.dim[1]
        self.parent = parent_actor
        self.world = parent_actor.get_world()
        self.bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        self.bp.set_attribute('image_size_x',str(self.width))
        self.bp.set_attribute('image_size_y',str(self.height))
        self.bp.set_attribute('fov',str(105))
        self.transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.set_sensor()

    def set_sensor(self):
        self.sensor = self.world.spawn_actor(self.bp,self.transform,attach_to=self.parent)
        self.sensor.listen(lambda image : self._parse_image(image))

    def _parse_image(self,image):
        self.image = image

    def render(self,display):
        if self.image is not None:
            i = np.array(self.image.raw_data)
            i2 = i.reshape((self.height, self.width, 4))
            i3 = i2[:, :, :3]
            self.image = i3
            cv2.imshow("depth_image", self.image)

class SegmentationCamera(object):
    def __init__(self,parent_actor,hud):
        self.sensor = None
        # self.seg_image = None
        # self.display = cv2.namedWindow('seg_image')
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.transform = self._camera_transforms[self.transform_index]

        self.width = hud.dim[0]
        self.height = hud.dim[1]
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # self.sem_bp.set_attribute("image_size_x", str(self.width))
        # self.sem_bp.set_attribute("image_size_y", str(self.height))
        # self.sem_bp.set_attribute("fov", str(105))
        # self.sem_bp.set_attribute('sensor_tick','1.0')
        self.transform = carla.Transform(carla.Location(x=1.6, z=1.7))

        self.set_sensor()

    def set_sensor(self):
        self.sensor = self.world.spawn_actor(self.sem_bp,self.transform,attach_to=self.parent)
        # self.sensor.listen(lambda img: self._parse_seg_image(img))
        # self.sensor.listen(lambda image: image.save_to_disk('image/%.6d.jpg' % image.frame,
        #                                                carla.ColorConverter.CityScapesPalette))
    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])
    def render(self,display):

        if self.seg_image is not None:

            array = np.frombuffer(self.seg_image.raw_data)
            array = array.reshape((self.height,self.width,4))
            array = array[:,:,:3]
            self.seg_image = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            # self.seg_image=array
            cv2.imshow("seg_image",self.seg_image)
            cv2.waitKey(1)


    def _parse_seg_image(self,img):
        img.convert(carla.ColorConverter.CityScapesPalette)

        array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
        array = array.reshape((self.height, self.width, 4))
        array = array[:, :, :3]
        self.seg_image = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        # self.seg_image = array
        cv2.imshow("camera.semantic_segmentation",self.seg_image)
        cv2.resizeWindow('Resized Window', self.width, self.height)
        cv2.waitKey(1)

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

