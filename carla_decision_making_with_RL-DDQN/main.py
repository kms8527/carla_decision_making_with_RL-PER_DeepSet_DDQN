import time
import glob
import os
import sys

tmp = ['/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload',
       '/home/a/.local/lib/python3.8/site-packages','/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages',
       '/home/a/Downloads/pycharm-2020.3.3/plugins/python/helpers/pycharm_display','/home/a/anaconda3/envs/RL_decision/lib/python37.zip',
       '/home/a/Downloads/pycharm-2020.3.3/plugins/python/helpers/pycharm_matplotlib_backend']
for index in range(0, len(tmp)):
    if tmp[index] in sys.path:
        sys.path.remove(tmp[index])

carla_library_path = 'home/a/Downloads/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64'
if not (carla_library_path in sys.path):
    sys.path.append(carla_library_path)

map_path = 'home/a/Downloads/CARLA_0.9.10/HDMaps'
if not (map_path in sys.path):
    sys.path.append(map_path)

print(sys.path)
print(sys.version)

sys.path.append('home/a/Downloads/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64/carla')

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from sensor_class import *
from KeyboardShortCutSetting import *
import random
from HUD import HUD
# from model import *
# import MPCController
from Controller import *
import logging
# from agents.navigation.roaming_agent import RoamingAgent
# from agents.navigation.basic_agent import BasicAgent

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from pygame import gfxdraw

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue
import cv2

class CarlaEnv():
    pygame.init()
    font = pygame.font.init()

    def __init__(self,world):
        #화면 크기
        self.width = 800
        self.height = 600
        #센서 종류
        self.actor_list = []
        self.extra_list = []
        self.extra_controller_list = []
        self.player = None
        self.camera_rgb = None
        self.camera_semseg = None
        self.lane_invasion_sensor = None
        self.collision_sensor = None
        self.gnss_sensor = None
        self.waypoint = None
        self.path = []

        self.world = world
        self.map = world.get_map()
        self.spectator = self.world.get_spectator()
        self.hud = HUD(self.width, self.height)
        self.waypoints = self.map.generate_waypoints(3.0)
                    ## visualize all waypoints ##
        # for n, p in enumerate(self.waypoints):
            # if n>1000:
            #     break
            # world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
            #                         color=carla.Color(r=255, g=255, b=255), life_time=30)

        self.extra_num = 5
        self.control_mode = None

        self.restart()
        self.main()

    def restart(self):
        blueprint_library = world.get_blueprint_library()
        # start_pose = random.choice(self.map.get_spawn_points())
        start_pose = self.map.get_spawn_points()[102]

        ##spawn points 시뮬레이션 상 출력##
        # print(start_pose)
        # for n, x in enumerate(self.map.get_spawn_points()):
        #     world.debug.draw_string(x.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=0, g=255, b=255), life_time=30)

        # self.load_traj()

        self.waypoint = self.map.get_waypoint(start_pose.location,lane_type=carla.LaneType.Driving)
        # print(self.waypoint.transform)
        ## Ego vehicle의 global Route 출력##
        # world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
        #                         color=carla.Color(r=0, g=255, b=255), life_time=100)


        # print(start_pose)
        # print(self.waypoint.transform)

        # self.controller = MPCController.Controller

        self.spectator.set_transform(carla.Transform(start_pose.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        self.player = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')),
            start_pose)
        self.actor_list.append(self.player)
        # vehicle.set_simulate_physics(False)

        self.camera_rgb =RGBSensor(self.player, self.hud)
        self.actor_list.append(self.camera_rgb.sensor)

        self.camera_semseg = SegmentationCamera(self.player,self.hud)
        self.actor_list.append(self.camera_semseg.sensor)

        self.collision_sensor = CollisionSensor(self.player, self.hud)  # 충돌 여부 판단하는 센서
        self.actor_list.append(self.collision_sensor.sensor)

        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)  # lane 침입 여부 확인하는 센서
        self.actor_list.append(self.lane_invasion_sensor.sensor)

        self.gnss_sensor = GnssSensor(self.player)
        self.actor_list.append(self.gnss_sensor.sensor)

        # --------------
        # Spawn Surrounding vehicles
        # --------------


        # spawn_points = carla.Transform(start_pose.location+carla.Location(0,10,0),start_pose.rotation)  #world.get_map().get_spawn_points()
        # spawn_points = self.map.get_waypoint(spawn_points.location,lane_type=carla.LaneType.Driving)
        spawn_points = world.get_map().get_spawn_points()[85:]
        number_of_spawn_points=len(world.get_map().get_spawn_points())#len(spawn_points)

        if self.extra_num <= number_of_spawn_points:
            # random.shuffle(spawn_points)
            pass
        elif self.extra_num > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            print(msg)
            # logging.warning(msg, self.extra_num, number_of_spawn_points)
            self.extra_num = number_of_spawn_points

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        # blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        # print(number_of_spawn_points)
        for n, transform in enumerate(spawn_points):

            ## surrounding vehicle 차량 spawn 지점 visualize
        #     world.debug.draw_string(transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=30)

            if n >= self.extra_num:
                break
            if transform == start_pose:
                continue
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            # blueprint.set_attribute('role_name', 'autopilot')
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

            extra = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')), transform)
            extra_pose = self.map.get_waypoint(transform.location,lane_type=carla.LaneType.Driving)
            self.extra_controller_list.append(Pure_puresuit_controller(extra, extra_pose , None , 30))  # km/h
            # extra.set_autopilot(True)
            self.extra_list.append(extra)

        # for response in client.apply_batch_sync(batch):
        #     if response.error:
        #         logging.error(response.error)
        #     else:
        #         self.extra_list.append(response.actor_id)

    def draw_image(self, surface, image, blend=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    def save_traj(self,current_location):
        f= open('route.txt','a')
        f.write(str(current_location.transform.location.x)+" ")
        f.write(str(current_location.transform.location.y)+" ")
        f.write(str(current_location.transform.location.z))
        f.write("\n")
        f.close()

    def load_traj(self):
        f= open("route.txt",'r')
        while True:
            line = f.readline()
            if not line: break
            _list = line.split(' ')
            _list = list(map(float,_list))
            point = carla.Transform()
            point.location.x = _list[0]
            point.location.y = _list[1]
            point.location.z = _list[2]
            self.path.append(point)
        #trajectory visualize
        # for n, x in enumerate(self.path):
        #     world.debug.draw_string(x.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=30)
        f.close()
    def get_gray_segimg(self,image_semseg):
        image_size = 96
        array = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8"))
        array = array.reshape((self.height, self.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


        cv2.imshow("camera.semantic_segmentation", array)
        cv2.resizeWindow('Resized Window', 100, 100)
        cv2.waitKey(1)

        array = np.reshape([cv2.resize(array, (image_size, image_size))],
                           (1, image_size, image_size))  # (1, 96, 96)

        return array


    def main(self):

        clock = pygame.time.Clock()
        display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        controller = Pure_puresuit_controller(self.player,self.waypoint,self.extra_list,50) # km/h
        Keyboardcontrol = KeyboardControl(self, False)
        # controller = VehicleControl(self, True)
        # self.player.apply_control(carla.Vehicle
        # Control(throttle=1.0, steer=0.0, brake=0.0))
        vehicles = self.world.get_actors().filter('vehicle.*')
        # for i in vehicles:
        #     print(i.id)
        cnt = 0
        decision = None
        try:
            # Create a synchronous mode context.
            with CarlaSyncMode(world, self.camera_rgb.sensor, self.camera_semseg.sensor, fps=40) as sync_mode:
                while True:
                    if Keyboardcontrol.parse_events(client, self, clock):
                        return
                    clock.tick()


                    cnt+=1
                    if cnt ==1000:
                        print('수해유 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ')
                        decision = 1
                        aa = controller.apply_control(decision)
                    else:
                        aa = controller.apply_control()

                    for i in range(len(self.extra_controller_list)):
                        self.extra_controller_list[i].apply_control()

                    # world.debug.draw_string(aa, 'o', draw_shadow=True,
                    #                         color=carla.Color(r=255, g=255, b=255), life_time=-1)
                    # print(self.waypoint.transform)
                    # print(self.player.get_transform())
                    # print(dist)

                    self.hud.tick(self, clock)
                    # print( clock.get_fps())
                    # Advance the simulation and wait for the data.
                    snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=0.1)

                    # Choose the next waypoint and update the car location.
                    # self.waypoint = random.choice(self.waypoint.next(1.5))
                    # self.player.set_transform(self.waypoint.transform)
                    # self.world.debug.draw_point(self.waypoint.transform.location,size=0.1,color=carla.Color(r=255, g=255, b=255),life_time=1)

                    ## player trajactory history visualize ## -> project_to_road = false 시 차량의 궤적, True 시 차선센터 출력
                    # curent_location=self.map.get_waypoint(self.player.get_transform().location,project_to_road=True)
                    # world.debug.draw_string(curent_location.transform.location, 'o', draw_shadow=True,
                    #                         color=carla.Color(r=0, g=255, b=255), life_time=100)
                    ##이동 궤적 저장
                    # self.save_traj(curent_location)

                    # self.world.debug.draw_point(ww.transform.location,size=0.1,color=carla.Color(r=255, g=255, b=255),life_time=1)

                    self.spectator.set_transform(
                        carla.Transform(self.player.get_transform().location + carla.Location(z=100), carla.Rotation(pitch=-90)))

                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    gray_image = self.get_gray_segimg(image_semseg)
                    # fps = round(1.0 / snapshot.timestamp.delta_seconds)

                    # Draw the display.
                    self.draw_image(display, image_rgb)
                    # self.draw_image(display, image_semseg, blend=False)
                    # pygame.gfxdraw.filled_circle(display,int(self.waypoint.transform.location.x),int(self.waypoint.transform.location.y),5,(255,255,255))
                    self.hud.render(display)

                    pygame.display.flip()

        finally:
            print('\ndestroying %d vehicles' % len(self.extra_list))
            # client.apply_batch([carla.command.DestroyActor(x) for x in self.extra_list])

            print('destroying actors.')
            for actor in self.actor_list:
                actor.destroy()
            for extra in self.extra_list:
                extra.destroy()

            pygame.quit()
            print('done.')

if __name__ == '__main__':

    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        client.load_world('Town04')
        client.set_timeout(4.0)

        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=30.0,
            sun_altitude_angle=90.0)

        world = client.get_world()
        world.set_weather(weather)
        CarlaEnv(world)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
