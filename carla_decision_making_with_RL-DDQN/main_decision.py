import time
import glob
import os
import sys


# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
# import cv2

import carla
import math
from sensor_class import *
from KeyboardShortCutSetting import *
import random
from HUD import HUD
# from model import *
# import MPCController
from Controller import *
from decision_trainer import *
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pyautogui
writer = SummaryWriter('runs/Jul07_16-15-29_a')#'runs/Jul06_11-16-43_a')#runs/May10_08-05-04_a')#runs/May09_13-35-46_a')
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


class CarlaEnv():
    pygame.init()
    font = pygame.font.init()

    def __init__(self,world):
        GPU_NUM = 1  # 원하는 GPU 번호 입력
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)  # change allocation of current GPU

        self.safety_mode = True
        self.safety_mode2 = False
        self.mission_mode = False
        #화면 크기
        self.start_epoch = True
        self.input_size = 4  # dr dv da dl
        self.output_size = 3
        self.width = 800
        self.height = 600
        #센서 종류
        self.actor_list = []
        self.extra_list = []
        self.extra_list = []
        self.trash_extra_list =[]
        self.extra_controller_list = []
        self.extra_dl_list = []
        self.extra_dl_list = []
        self.player = None
        self.camera_rgb = None
        self.camera_semseg = None
        self.lane_invasion_sensor = None
        self.collision_sensor = None
        self.gnss_sensor = None
        self.spawn_waypoint = None
        self.path = []
        self.save_dir = None
        self.world = world
        self.map = world.get_map()
        self.spectator = self.world.get_spectator()
        self.mission_goal_point = self.map.get_waypoint(carla.Location(x=-137.000275, y=16.263311, z=7.994740),
                                                    lane_type=carla.LaneType.Driving).transform.location

        self.hud = HUD(self.width, self.height)
        # self.spawn_waypoints = self.map.generate_waypoints(3.0)
        self.lane_change_time = time.time()-100.0
        self.max_Lane_num = 4
        self.ego_Lane = 2
        self.pre_ego_lane = self.ego_Lane
        self.agent = None
        self.controller = None
        self.is_first_time = True
        self.decision = 0
        self.pre_decision = 0
        self.simul_time = time.time()
        self.accumulated_reward = 0
        self.accumulated_loss = 0

        self.end_point = 0
        self.ROI_length = 100.0 #(meters)
        self.safe_lane_change_distance = 0
        self.search_radius = None
        self.left_search_radius = None
        self.right_search_radius = None
        self.decision_changed = False
        self.check = 0
        self.check2 = 0
        self.append_lane_change_sample=False
        self.current_ego_lane =0

        # about mission mode #
        self.collision_num = 0
        self.left_col_num = 0
        self.right_col_num = 0
        self.exit_lane_col_num = 0
        self.mission_clear_num = 0
        self.iter = 0
        self.eplased_time = 0
        ## visualize all waypoints ##
        # for n, p in enumerate(self.spawn_waypoints):
        #     world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=999)

        settings = self.world.get_settings()
        # settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(settings)
        self.extra_num = 20
        self.scenario =  "random"# "scenario3" # #
        self.pilot_style = "manual" # "auto"

        self.section = 0
        self.lane_distance_between_start =None
        self.lane_distance_between_end = None
        self.lane_change_point = None
        self.episode_start = None
        self.index = 0
        self.distance_memory = None
        self.vehicles_distance_memory = torch.zeros(self.extra_num)
        self.pre_max_Lane_num = self.max_Lane_num
        self.ready_to_store_sample = False
        self.is_lane_out = False
        self.actor_pos = None # carla.location type
        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),
                                 carla.Location(x=172.745468, y=-364.531799, z=0.000000),
                                 carla.Location(x=382.441040, y=-212.488907, z=0.000000),
                                 carla.Location(x=161.251068, y=7.560803, z=8.935559),
                                 carla.Location(x=16.137751, y=143.156509, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),
                                    carla.Location(x=232.962860, y=-364.149139, z=0.000000),
                                    carla.Location(x=376.542816, y=-10.352980, z=0.000000),
                                    carla.Location(x=-136.986206, y=5.763320, z=7.994740),
                                    carla.Location(x=15.144917, y=-74.823540, z=0.000000)]

        self.lane_change_point = []  # [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) ,carla.Location(x=232.962860, y=-364.149139, z=0.000000), carla.Location(x=382.441040, y=-212.488907, z=0.000000),carla.Location(x=376.542816, y=-10.352980, z=0.000000)]
        for i in range(len(self.lane_start_point)):
            self.lane_change_point.append(self.lane_start_point[i])
            self.lane_change_point.append(self.lane_finished_point[i])
        self.lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            self.lane_distance_between_points.append(self.uclidian_distance(self.lane_start_point[i],self.lane_finished_point[i]))

        self.visualize_virtual_invalid_forth_lane()

        self.restart()
        self.main()

    def restart(self):
        self.is_lane_out = False
        self.actor_pos = None
        # self.check = 0
        self.append_lane_change_sample=False
        self.decision = 0
        self.pre_decision = 0
        self.decision_changed = False
        self.simul_time = time.time()
        self.distance_memory = None
        self.vehicles_distance_memory = torch.zeros(self.extra_num)
        self.lane_change_time = time.time()-100.0
        self.can_lane_change = True
        self.pre_can_lane_change = True
        self.ready_to_store_sample = False
        # print("-----start-------")
        # print("pre:", self.pre_can_lane_change)
        # print("pre:", self.can_lane_change)
        # print("------------")
        self.max_Lane_num = 4
        self.ego_Lane = 2
        self.pre_ego_lane = self.ego_Lane
        self.controller = None
        self.accumulated_reward = 0
        self.acummulated_loss = 0
        self.section = 0
        self.episode_start = time.time()
        self.pre_max_Lane_num = self.max_Lane_num
        self.index = 0
        self.iter = 0
        self.arrive_lane_chagne_change_point = False
        # print('start destroying actors.')
        # Traffic_light.freeze(True)
        self.clear_all_actors()

        global actors
        global sensors
        actors = []
        sensors = []
        self.actor_list = []
        self.extra_list = []
        self.trash_extra_list = []
        self.ROI_extra_list = []
        self.extra_controller_list = []
        self.extra_dl_list = []
        self.ROI_extra_dl_list = []
        blueprint_library = self.world.get_blueprint_library()
        # start_pose = random.choice(self.map.get_spawn_points())
        start_pose = carla.Transform(carla.Location(x=8.180587, y=-85.720520, z=0.281942), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000))
        ##spawn points 시뮬레이션 상 출력##
        # print(start_pose)
        # for n, x in enumerate(self.map.get_spawn_points()):
        #     world.debug.draw_string(x.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=0, g=255, b=255), life_time=30)

        # self.load_traj()

        self.spawn_waypoint = self.map.get_waypoint(start_pose.location,lane_type=carla.LaneType.Driving)
        # self.end_point = self.spawn_waypoint.next(400)[0].transform.location
        # print(self.spawn_waypoint.transform)
        ## Ego vehicle의 global Route 출력 ##
        # world.debug.draw_string(self.spawn_waypoint.transform.location, 'o', draw_shadow=True,
        #                         color=carla.Color(r=255, g=0, b=255), life_time=100)

        # print(start_pose)
        # print(self.spawn_waypoint.transform)

        # self.controller = MPCController.Controller
        player_blueprint=random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))
        color = '0,255,0'
        player_blueprint.set_attribute('color', color)
        self.player = world.spawn_actor(player_blueprint, start_pose)
        # print(self.player.bounding_box) # ego vehicle length

        self.actor_list.append(self.player)
        actors.append(self.player)
        # self.camera_rgb =RGBSensor(self.player, self.hud)
        # self.actor_list.append(self.camera_rgb.sensor)

        # self.camera_depth =DepthCamera(self.player, self.hud)
        # self.actor_list.append(self.camera_depth.sensor)

        # self.camera_semseg = SegmentationCamera(self.player,self.hud)
        # self.actor_list.append(self.camera_semseg.sensor)

        self.collision_sensor = CollisionSensor(self.player, self.hud)  # 충돌 여부 판단하는 센서
        self.actor_list.append(self.collision_sensor.sensor)
        sensors.append(self.collision_sensor.sensor)

        # self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)  # lane 침입 여부 확인하는 센서
        # self.actor_list.append(self.lane_invasion_sensor.sensor)

        # self.gnss_sensor = GnssSensor(self.player)
        # self.actor_list.append(self.gnss_sensor.sensor)

        self.spectator.set_transform(
            carla.Transform(self.player.get_transform().location + carla.Location(z=130),
                            carla.Rotation(pitch=-90)))

        # --------------
        # Spawn Surrounding vehicles
        # --------------

        print("Generate Extra")
        spawn_points=[]
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # print(*blueprints)
        spawn_point = None

        if self.scenario == "random":
            extra_spawn_point = 20
            distance_step = 20
            end = distance_step * self.extra_num# (self.extra_num + len(self.lane_start_point))
            # for i in range(distance_step, end, distance_step):
            while extra_spawn_point <= end:
                dl = random.choice([-1, 0, 1]) ## for 4 lane spawn: , 2])
                self.extra_dl_list.append(dl)
                if dl == -1:
                    spawn_point = self.spawn_waypoint.next(extra_spawn_point)[0].get_left_lane().transform
                elif dl == 0:
                    spawn_point = self.spawn_waypoint.next(extra_spawn_point)[0].transform
                elif dl == 1:
                    spawn_point = self.spawn_waypoint.next(extra_spawn_point)[0].get_right_lane().transform
                # elif dl == 2:
                #     if self.check_max_lane(self.spawn_waypoint.next(extra_spawn_point)[0].transform.location,
                #                            self.lane_start_point, self.lane_finished_point) == 3:
                #         continue
                #     else:
                #         spawn_point = self.spawn_waypoint.next(extra_spawn_point)[
                #             0].get_right_lane().get_right_lane().transform
                else:
                    print("Except ")


                # print(blueprint_library.filter('vehicle.bmw.grandtourer'))
                # blueprint = random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))

                blueprint = random.choice(blueprints)

                # print(blueprint.has_attribute('color'))

                if blueprint.has_attribute('color'):
                        # color = random.choice(blueprint.get_attribute('color').recommended_values)
                        # print(blueprint.get_attribute('color').recommended_values)
                        color = '255,255,255'
                        blueprint.set_attribute('color', color)
                if extra_spawn_point <= distance_step * (self.extra_num):
                    spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                    spawn_points.append(spawn_point)
                    extra = self.world.spawn_actor(blueprint,spawn_point)
                    self.extra_list.append(extra)
                    actors.append(extra)

                extra_spawn_point += 20

##--lane_start_point extra spawn
            # for point in self.lane_start_point:
            #     spawn_point = self.map.get_waypoint(point, lane_type=carla.LaneType.Driving).next(1)[0]
            #     spawn_point = carla.Transform((spawn_point.transform.location + carla.Location(z=1)),
            #                                   spawn_point.transform.rotation)
            #     spawn_points.append(spawn_point)
            #     # print(extra_spawn_point.transform.location)
            #     trash_extra = world.spawn_actor(blueprint, spawn_point)
            #     self.trash_extra_list.append(trash_extra)
            #     actors.append(trash_extra)
##--lane_start_point extra spawn

            # spawn_point = carla.Transform(carla.Location(x=14.797643, y=-163.310318, z=2.000000),
            #                               carla.Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000))
            # blueprint = random.choice(blueprints)
            # print("spawn_points:",*spawn_points)
            # print(self.extra_dl_list)

            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    tm.auto_lane_change(extra, False)

                    # tm.vehicle_percentage_speed_difference(extra,0)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    # target_velocity = 30  # random.randrange(10, 40) # km/h
                    # extra.set_target_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)

                    # extra.enable_constant_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)
                    extra.set_autopilot(True,tm_port)

                    # self.world.constant_velocity_enabled = True

        elif self.scenario == "scenario1":
            self.extra_num = 3
            d = 15
            self.extra_dl_list = [-1, 0, 1]

            for i in range(3):
                if i==0:
                    spawn_point = self.spawn_waypoint.next(d)[0].get_left_lane().transform
                elif i==1:
                    spawn_point = self.spawn_waypoint.next(d)[0].transform
                elif i==2:
                    spawn_point = self.spawn_waypoint.next(d)[0].get_right_lane().transform
                else:
                    print("Except ")

                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 30  # random.randrange(10, 40) # km/h
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        elif self.scenario == "scenario2":
            self.extra_num = 4
            d = 15
            self.extra_dl_list = [-1, 0, 1, 2]

            for i in range(4):
                if i==0:
                    spawn_point = self.spawn_waypoint.next(d)[0].get_left_lane().transform
                elif i==1:
                    spawn_point = self.spawn_waypoint.next(d)[0].transform
                elif i==2:
                    spawn_point = self.spawn_waypoint.next(d)[0].get_right_lane().transform
                elif i ==3: # 200 originally
                    spawn_point = self.spawn_waypoint.next(1000)[0].get_right_lane().get_right_lane().transform
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    if extra == self.extra_list[-1]:
                        target_velocity = 0.0
                    else:
                        target_velocity = 30.0  # random.randrange(10, 40) # km/h
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        elif self.scenario == "scenario3":
            self.extra_num = 4
            self.extra_dl_list = [-1, 0, 1, 2]

            for i in range(5):
                if i == 0:
                    spawn_point = self.spawn_waypoint.next(100)[0].get_left_lane().transform
                elif i == 1:
                    spawn_point = self.spawn_waypoint.next(150)[0].transform
                elif i == 2:
                    spawn_point = self.spawn_waypoint.next(100)[0].get_right_lane().transform
                elif i == 3:
                    spawn_point = self.spawn_waypoint.next(150)[0].get_right_lane().get_right_lane().transform
                elif i == 4:
                    spawn_point = carla.Transform(carla.Location(x=14.797643, y=-163.310318, z=2.000000), carla.Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000))
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                # if blueprint.has_attribute('color'):
                #     # color = random.choice(blueprint.get_attribute('color').recommended_values)
                #     # print(blueprint.get_attribute('color').recommended_values)
                #     color = '255,255,255'
                #     blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)

                self.extra_list.append(extra)
                # if i < 4:
                #     self.extra_list.append(extra)
                # else:
                #     trash_extra = extra

            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 20.0

                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)

            # trash_extra.enable_constant_velocity(extra.get_transform().get_right_vector() * 0.0 / 3.6)
            # trash_extra.destroy()

        elif self.scenario == "validation":
            self.extra_dl_list=[0, 0, 1, -1, 1, 1, -1, 0, 1, 0, 1, -1, -1, 1, 0, -1, -1, 0, 0, -1]
            spawn_points = [carla.Transform(carla.Location(x=8.023607, y=-105.720062, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000)), carla.Transform(carla.Location(x=7.945116, y=-125.719902, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000)), carla.Transform(carla.Location(x=11.366597, y=-145.733475, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000)), carla.Transform(carla.Location(x=4.288160, y=-165.705856, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000)), carla.Transform(carla.Location(x=11.209615, y=-185.733170, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000)), carla.Transform(carla.Location(x=11.131124, y=-205.733017, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000)), carla.Transform(carla.Location(x=4.167244, y=-225.503204, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-88.029251, roll=0.000000)), carla.Transform(carla.Location(x=9.491584, y=-244.189850, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-80.877213, roll=0.000000)), carla.Transform(carla.Location(x=17.005051, y=-261.642273, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-73.725182, roll=0.000000)), carla.Transform(carla.Location(x=20.061684, y=-280.395905, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-66.573143, roll=0.000000)), carla.Transform(carla.Location(x=31.654175, y=-295.450928, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-59.421112, roll=0.000000)), carla.Transform(carla.Location(x=36.481331, y=-315.009735, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-52.269077, roll=0.000000)), carla.Transform(carla.Location(x=49.242310, y=-329.531738, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-45.117043, roll=0.000000)), carla.Transform(carla.Location(x=68.018288, y=-336.833252, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-37.965008, roll=0.000000)), carla.Transform(carla.Location(x=81.458138, y=-350.264893, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-30.812973, roll=0.000000)), carla.Transform(carla.Location(x=96.853889, y=-362.118622, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-23.660938, roll=0.000000)), carla.Transform(carla.Location(x=115.010300, y=-368.757507, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-16.508904, roll=0.000000)), carla.Transform(carla.Location(x=134.421051, y=-369.630768, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-9.356870, roll=0.000000)), carla.Transform(carla.Location(x=153.220459, y=-371.533997, z=1.000000), carla.Rotation(pitch=0.000000, yaw=-2.204835, roll=0.000000)), carla.Transform(carla.Location(x=172.837189, y=-375.031403, z=1.000000), carla.Rotation(pitch=0.000000, yaw=0.501399, roll=0.000000))]
            for spawn_point in spawn_points:
                blueprint = random.choice(blueprints)

                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)

                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
                actors.append(extra)

            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    tm.auto_lane_change(extra, False)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 30  # random.randrange(10, 40) # km/h
                    extra.set_target_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)

                    # extra.enable_constant_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)
                    extra.set_autopilot(True, tm_port)

                    self.world.constant_velocity_enabled = True
                    # traffic_manager.auto_lane_change(extra,False)
        # print('Extra Genration Finished')

        # tmp = self.map.get_waypoint(self.player.get_location(),lane_type=carla.LaneType.Driving)
        # tmp_rotation = tmp.transform.rotation

        # extra_target_velocity = 10

        # traffic_manager = client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(100.0)
        # traffic_manager.set_synchronous_mode(True) # for std::out_of_range eeror
        # tm_port = traffic_manager.get_port()

        # print("tm setting finished")
        # self.player.set_autopilot(True,tm_port)
        # traffic_manager.auto_lane_change(self.player, False)
        self.controller = Pure_puresuit_controller(self.player, self.spawn_waypoint, self.extra_list, 30)  # km/h
        # print("controller setting finished")
        # target_velocity = 60 / 3.6
        # forward_vec = self.player.get_transform().get_forward_vector()
        # print(forward_vec)
        # velocity_vec =  target_velocity*forward_vec
        # self.player.set_target_velocity(velocity_vec)
        # print(velocity_vec)

        # print(velocity_vec)
        # client.get_trafficmanager.auto_lane_change(extra, False)
        ###Test####
        # clock = pygame.time.Clock()
        # Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        # while True:
        #     if Keyboardcontrol.parse_events(client, self, clock):
        #         return
        #     self.spectator.set_transform(
        #         carla.Transform(self.player.get_transform().location + carla.Location(z=50),
        #                         carla.Rotation(pitch=-90)))
        #     self.camera_rgb.render(display)
        #     self.hud.render(display)
        #     pygame.display.flip()
        #
        #     self.controller.apply_control()
        #     # self.world.wait_for_tick(10.0)
        #     clock.tick(30)
        #
        #     self.hud.tick(self, clock)

        #### Test Finished #####
        #### Test2 #####
        # cnt=0
        # clock = pygame.time.Clock()
        # while True:
        #     # print(self.spawn_waypoint.lane_id)
        #     self.spectator.set_transform(
        #         carla.Transform(self.player.get_transform().location + carla.Location(z=100),
        #                         carla.Rotation(pitch=-90)))
        #     cnt += 1
        #     if cnt == 100:
        #         print('수해유 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ')
        #         decision = 1
        #         self.controller.apply_control(decision)
        #     else:
        #        self.controller.apply_control()
        #     clock.tick(30)
        #### Test2 Finished #####


        # self.input_size = (self.extra_num)*4 + 1

        # for response in client.apply_batch_sync(batch):
        #     if response.error:
        #         logging.error(response.error)
        #     else:
        #         self.extra_list.append(response.actor_id)

    # def restart(self):
    #     self.controller = Pure_puresuit_controller(self.player, self.spawn_waypoint, self.extra_list, 70)  # km/h

    def check_max_lane(self,actor_pos, lane_start_point, lane_finished_point):
        self.section = self.search_section(actor_pos, lane_start_point, lane_finished_point)
        if self.section % 2 == 0:
            return 4
        else:
            return 3

    def search_section(self,actor_pos, lane_start_point, lane_finished_point):
        """
        :param actor_pos: carla.location
        :param lane_start_point: carla.location list
        :param lane_finished_point: carla.location
        :return: int
        """


        if (actor_pos.x >= lane_finished_point[4].x-20.0 and actor_pos.x < lane_start_point[0].x+5) and (actor_pos.y >=lane_start_point[0].y and actor_pos.y < lane_finished_point[4].y):
            return 0
        elif (actor_pos.x >= lane_finished_point[0].x-20.0 and actor_pos.x < lane_start_point[0].x+5) and (actor_pos.y >= lane_finished_point[0].y and actor_pos.y < lane_start_point[0].y):
            return 1
        elif (actor_pos.x >= lane_finished_point[0].x-20.0 and actor_pos.x < lane_start_point[1].x) and (actor_pos.y >=lane_start_point[1].y-20 and actor_pos.y < lane_finished_point[0].y):
            return 2
        elif (actor_pos.x >= lane_start_point[1].x and actor_pos.x < lane_finished_point[1].x) and (actor_pos.y >= lane_start_point[1].y-20 and actor_pos.y < lane_finished_point[1].y):
            return 3
        elif (actor_pos.x >= lane_finished_point[1].x and actor_pos.x < lane_start_point[2].x+20) and (actor_pos.y >= lane_finished_point[1].y-20 and actor_pos.y < lane_start_point[2].y):
            return 4
        elif (actor_pos.x >= lane_finished_point[2].x and actor_pos.x < lane_start_point[2].x + 20.0) and (actor_pos.y >= lane_start_point[2].y and actor_pos.y < lane_finished_point[2].y):
            return 5
        elif (actor_pos.x >= lane_start_point[3].x and actor_pos.x < lane_finished_point[2].x + 20.0) and (actor_pos.y >= lane_finished_point[2].y and actor_pos.y < lane_start_point[3].y+10):
            return 6
        elif (actor_pos.x >= lane_finished_point[3].x and actor_pos.x < lane_start_point[3].x) and (actor_pos.y >= lane_finished_point[3].y-10 and actor_pos.y < lane_start_point[3].y+10) and (self.index == 6 or self.index == 7):
            return 7
        elif (actor_pos.x >= lane_finished_point[4].x-20.0 and actor_pos.x < lane_start_point[4].x+5) and (actor_pos.y >= lane_finished_point[4].y and actor_pos.y < lane_start_point[4].y) and (self.index == 8 or self.index == 9):
            return 9
        else:
            return 8

    def clear_all_actors(self):
        if len(self.actor_list) != 0:
            # print('destroying actors.')
            # print("actor 제거 :", self.actor_list)

            if self.collision_sensor.sensor.is_listening:
                self.collision_sensor.sensor.stop()
            for actor in self.actor_list:
                    actor.destroy()
            # print("finshed actor destroy")

            # for x in self.actor_list:
            #     try:
            #         client.apply_batch([carla.command.DestroyActors(x.id)])
            #     except:
            #         continue
        if len(self.trash_extra_list) != 0:
            for x in self.trash_extra_list:
                try:
                    client.apply_batch([carla.command.DestroyActor(x.id)])
                except:
                    continue

        if len(self.extra_list) != 0:
            # client.apply_batch([carla.command.DestoryActors(x.id) for x in self.extra_list])
            # print("extra 제거 :", self.extra_list)
            for x in self.extra_list:
                try:
                    client.apply_batch([carla.command.DestroyActor(x.id)])
                except:
                    continue
            # for extra in self.extra_list:
            #     # print("finally 에서 actor 제거 :", self.extra_list)
            #     print(extra.is_alive)
            #     if extra.is_alive:
            #         extra.destroy()
            # print("finshed extra destroy")

            # for x in self.extra_list:
            #     try:
            #         client.apply_batch([carla.command.DestroyActor(x.id)])
            #     except:
            #         continue
        # print('finished destroying actors.')
    def is_extra_front_than_this_point(self,extra_pos,point):
        """

        :param extra_pos:  carla.location type, extra first lane pos
        :param point: carla.Transform type,
        :return: bool type
        """
        extra_pos_tensor = torch.tensor([[extra_pos.x, extra_pos.y, extra_pos.z, 1.0]])

        theta_z =  math.radians(point.rotation.yaw)
        trans = point.location

        n = torch.tensor([[math.cos(theta_z), math.sin(theta_z), 0]])
        o = torch.tensor([[-math.sin(theta_z),  math.cos(theta_z), 0]])
        a = torch.tensor([[0.0, 0.0, 1.0]])
        p = torch.tensor([[trans.x, trans.y, trans.z]])

        #get inverse matrix of homogenous matrix
        m41 = -torch.dot(torch.squeeze(n),torch.squeeze(p)).unsqueeze(0).unsqueeze(1) # row4 col1 componets
        m42 = -torch.dot(torch.squeeze(o),torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
        m43 = -torch.dot(torch.squeeze(a), torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
        m = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


        Global_T_relative =\
            torch.cat((torch.cat((n,m41),1),torch.cat((o,m42),1),torch.cat((a,m43),1),m),0)

        output = torch.matmul(Global_T_relative, extra_pos_tensor.T)
        if output[0]>0:
            return True
        else:
            return False
        # print(Global_T_relative)
        # print("output = ", Global_T_relative*[[trans.x], [trans.y], [trans.z], [1]])


    def step(self, decision):

        plc = 0.1
        # decision = None
        '''
        # Simple Action (action number: 3)
        action_test = action -1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
        '''

        # Complex Action
        if decision == 0:  # LK
            plc = 0
        # elif action == -1:  # left
        #     plc += 10
        #     decision = -1
        # elif action == 1:  # right
        #     plc += 10
        #     decision = 1
        pre_ego_lane = self.ego_Lane

        tmp = self.get_next_state(decision)  # get now state

        if tmp is None:
            return None
        else:
            next_state, next_x_static = tmp

        # end_length=math.sqrt((self.end_point.x - self.player.get_location().x)**2+(self.end_point.y - self.player.get_location().y)**2)

        done = False
        if self.mission_mode == False:
            if time.time() - self.simul_time > 25:
                print("simultime done")
                done = True
                reward = 0
            elif len(self.collision_sensor.history) != 0:
                done = True
                print("collision")
                self.collision_num +=1
                reward = -10
            elif decision == 1 and pre_ego_lane >= self.max_Lane_num:  # dont leave max lane
                done = True
                self.is_lane_out = True
                self.right_col_num +=1
                print("lane right collision")
                reward = -10
            elif decision == -1 and pre_ego_lane <= 1:  # dont leave min lane
                done = True
                self.left_col_num +=1
                self.is_lane_out = True
                print("lane left collision")
                reward = -10
            elif next_x_static[2] < 0 and self.ego_Lane > self.max_Lane_num :
                done = True
                self.exit_lane_col_num +=1
                self.is_lane_out = True
                # print("ego_Lane",self.ego_Lane,"mam lane",self.max_Lane_num)
                print("Agent get into exit, Done")
                reward = -10
            else:
                reward = (1-1*abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel))**3-plc
                # print(abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel))
        else:
            if len(self.collision_sensor.history) != 0:
                done = True
                print("collision")
                self.collision_num +=1
                reward = -10
            elif decision == 1 and pre_ego_lane >= self.max_Lane_num:  # dont leave max lane
                done = True
                self.is_lane_out = True
                self.right_col_num +=1
                print("lane right collision")
                reward = -10
            elif decision == -1 and pre_ego_lane <= 1:  # dont leave min lane
                done = True
                self.left_col_num +=1
                self.is_lane_out = True
                print("lane left collision")
                reward = -10
            elif next_x_static[2] < 0 and self.ego_Lane > self.max_Lane_num :
                done = True
                self.exit_lane_col_num +=1
                self.is_lane_out = True
                # print("ego_Lane",self.ego_Lane,"mam lane",self.max_Lane_num)
                print("Agent get into exit, Done")
                reward = -10
            else:
                reward = (1-1*abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel))**3-plc
                # print(abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel))

        # if self.mission_mode == False:
            # if time.time()-self.simul_time > 25:
            #     print("simultime done")
            #     done = True
            #     reward = 0
        if self.mission_mode == True:
            if self.uclidian_distance(self.get_waypoint_of_first_lane(self.player).transform.location,self.mission_goal_point) < 50:
                self.mission_clear_num +=1
                done = True
                reward = 0

        # print("ego:", self.ego_Lane, "max_lane", self.max_Lane_num,"decision:", decision)

        # print(reward)
        # if self.decision_changed == True:
        #     reward -= -1
        self.accumulated_reward += reward

        #state length = 4 * num of extra vehicles + 1

        return None, None, decision ,reward, next_state, next_x_static , done
                                        # Next State 표현 필요

    def get_dr(self,extra,num):
        ego_first_lane = self.get_waypoint_of_first_lane(self.player)
        extra_first_pos = self.get_waypoint_of_first_lane(extra)
        if isinstance(extra_first_pos,bool) is False:
            extra_first_pos= extra_first_pos.transform.location
        else:
            return False

        distance1 = 0
        step = 6
        waypoint = ego_first_lane

        # self.world.debug.draw_string(self.player.get_location(),
        #                              'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=-1)
        # self.world.debug.draw_string(extra.get_location(),
        #                              'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=255, b=255), life_time=-1)

        # self.world.debug.draw_string(ego_first_lane.transform.location,
        #                              'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=0.1)
        # self.world.debug.draw_string(extra_first_pos,
        #                              'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=255, b=255), life_time=0.1)

        if self.vehicles_distance_memory[num] != 0: # abpit not initial state
            if self.vehicles_distance_memory[num] > 0: # about front extra vehicle
                if isinstance(waypoint,bool) or len(waypoint.next(self.vehicles_distance_memory[num].item()))==0:
                    return False
                tmp = waypoint.next(self.vehicles_distance_memory[num].item())[0].transform

            elif self.vehicles_distance_memory[num] < 0:
                if isinstance(waypoint,bool) or len( waypoint.previous(abs(self.vehicles_distance_memory[num].item())))==0:
                    return False
                tmp = waypoint.previous(abs(self.vehicles_distance_memory[num].item()))[0].transform

            if self.is_extra_front_than_this_point(extra_first_pos,tmp):
                self.vehicles_distance_memory[num] += self.uclidian_distance(tmp.location, extra_first_pos)
            else:
                self.vehicles_distance_memory[num] -= self.uclidian_distance(tmp.location, extra_first_pos)

            return self.vehicles_distance_memory[num]

        else:
            while self.uclidian_distance(waypoint.transform.location, extra_first_pos) >= 2 * step:
                waypoint = waypoint.next(step)[0]

                # self.world.debug.draw_string(waypoint.transform.location,
                #                              'o', draw_shadow=True,
                #                              color=carla.Color(r=0, g=255, b=255), life_time=1)

                distance1 += step
            self.vehicles_distance_memory[num] = distance1 + self.uclidian_distance(waypoint.transform.location, extra_first_pos)

            return self.vehicles_distance_memory[num]

    def get_next_state(self,decision=None):
        """
        dl : relative lane num after ching the lane
        dr, dv, da : now state
        """
        state = []
        self.actor_pos = self.player.get_transform().location
        # self.get_max_lane(self.lane_distance_between_points)
        self.max_Lane_num = self.check_max_lane(self.actor_pos, self.lane_start_point,self.lane_finished_point)

        lane_valid_distance = self.search_distance_valid(self.max_Lane_num)
        if lane_valid_distance is None:
            return None
        # else: ## for debug
        #     print("lane_valid_distance:", lane_val"id_distance)

        for x, extra in enumerate(self.extra_list):

            if decision == 1:
                self.extra_dl_list[x] =self.extra_dl_list[x]-1
                # self.ROI_extra_dl_list[x] =self.ROI_extra_dl_list[x]+1

            elif decision == -1:
                self.extra_dl_list[x] = self.extra_dl_list[x]+1
                # self.ROI_extra_dl_list[x] = self.ROI_extra_dl_list[x]-1

            else:
                pass
            if extra.is_alive == False:
                return None
            extra_vel = extra.get_velocity()

            # extra_acel = extra.get_acceleration()
            # sign = 0

            # extra_pos = extra.get_transform().location
            # cur_agent_pos = self.player.get_transform().location

            # if self.is_extra_front_than_this_point(extra_pos) == True:
            #     sign = 1
            # else:
            #     sign = -1
            # dr = sign * self.uclidian_distance(cur_agent_pos,extra_pos) - abs(self.spawn_waypoint.lane_width*(self.extra_dl_list[x]))

            dr = self.get_dr(extra,x)

            # print(dr)
            if dr == False:
                return None
            player_vel = (self.player.get_velocity().x** 2 + self.player.get_velocity().y** 2 + self.player.get_velocity().z** 2) ** 0.5
            dv =player_vel - (extra_vel.x ** 2 + extra_vel.y ** 2 + extra_vel.z ** 2) ** 0.5
            length = 2 * extra.bounding_box.extent.x # length of extra vehicle


            # da = ((extra_acel.x - self.player.get_acceleration().x) ** 2 + (
            #         extra_acel.y - self.player.get_acceleration().y) ** 2 +
            #       (extra_acel.z - self.player.get_acceleration().z) ** 2) ** 0.5
            state_dyn = torch.tensor([dr/self.ROI_length, dv/10.0 , length/10.0, self.extra_dl_list[x]])
            # print(state_dyn[0])
            state.append(state_dyn)

            # state.append(dr)
            # state.append(dv)
            # state.append(da)
            # state.append(self.extra_dl_list[x])


        # if self.can_lane_change == True:
        #     # print("finished lane change", self.pre_decision)
        #     if self.pre_decision == 1.0:
        #         # print("finish right")
        #         self.ego_Lane = math.ceil(self.ego_Lane) / 1.0
        #     elif self.pre_decision == -1.0:
        #         # print("finish left")
        #         self.ego_Lane = math.floor(self.ego_Lane) / 1.0
        #
        # self.current_ego_lane = self.ego_Lane

        if decision == 1 :
            self.ego_Lane += 1
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = False
            # print("-----at get next state-------")
            # print("pre:", self.pre_can_lane_change)
            # print("now:", self.can_lane_change)
            # print("------------")

        elif decision == -1:
            self.ego_Lane += -1
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = False
            # print("-----at get next state-------")
            # print("pre:", self.pre_can_lane_change)
            # print("now:", self.can_lane_change)
            # print("------------")
        else:
            pass

        # if self.ego_Lane % 1 == 0 and self.decision !=0:
        #     print("here")

        x_static= []

        x_static.append(torch.tensor([self.ego_Lane, self.controller.velocity / 100.0, lane_valid_distance / self.ROI_length]))

        x_static = torch.cat(x_static)
        state = torch.cat(state)

        # state.append(x_static)

        # state.append(self.ego_Lane)
        out = [state, x_static]
        return out

    def uclidian_distance(self,a,b):
        return ((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)**0.5

    def get_distance_from_waypoint_to_goal_point(self, starting_waypoint, goal_point):
        """
        starting_waypoint : actor's last lane waypoint position
        goal_point  :  goal position
        """

        distance1 = 0
        step = 6
        waypoint = starting_waypoint
        # print("distance memroy:", self.distance_memory)
        if self.distance_memory is not None:
            # tmp = self.uclidian_distance(waypoint.transform.location, goal_point)
            # if tmp >= 10:
            #     print("asdf")
            estimated_waypoint = waypoint.next(self.distance_memory)[0]

            # if self.distance_memory > 0:
            #     self.world.debug.draw_string(waypoint.next(self.distance_memory)[0].transform.location,
            #                                  'o', draw_shadow=True,
            #                                  color=carla.Color(r=255, g=0, b=0), life_time=9991)
            #     self.world.debug.draw_string(goal_point,
            #                                  'o', draw_shadow=True,
            #                                  color=carla.Color(r=0, g=0, b=255), life_time=9991)


            if self.is_extra_front_than_this_point(goal_point,estimated_waypoint.transform):
                self.distance_memory += self.uclidian_distance(estimated_waypoint.transform.location, goal_point)
            else:
                self.distance_memory -= self.uclidian_distance(estimated_waypoint.transform.location, goal_point)

            if self.distance_memory < 0:
                self.world.debug.draw_string(estimated_waypoint.transform.location,
                                                 'o', draw_shadow=True,
                                                 color=carla.Color(r=255, g=0, b=0), life_time=9999)
                # self.world.debug.draw_string(goal_point,
                #                              'o', draw_shadow=True,
                #                              color=carla.Color(r=255, g=0, b=255), life_time=9999)

            return self.distance_memory

        else:
            while self.uclidian_distance(waypoint.transform.location, goal_point) >= 2 * step:
                if len(waypoint.next(step)) == 0:
                    return False
                waypoint = waypoint.next(step)[0]
                distance1 += step

                # print(waypoint.transform.location)
                # print(goal_point)
                # print(self.uclidian_distance(waypoint.transform.location, goal_point))
                self.distance_memory = distance1 + self.uclidian_distance(waypoint.transform.location, goal_point)


                # self.world.debug.draw_string(carla.Location(x=172.776093, y=-368.031677, z=0.000000),'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255), life_time=999)
                # self.world.debug.draw_string(carla.Location(x=182.776093, y=-368.031677, z=0.000000),'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255), life_time=999)
            # if self.index == 8:
            #     print()
            #     print("goal_point:", goal_point)
            #     print("wayhpoint:", waypoint.transform.location)
            #     self.world.debug.draw_string(waypoint.transform.location, 'o', draw_shadow=True,
            #                                  color=carla.Color(r=0, g=255, b=255), life_time=999)
            #     self.world.debug.draw_string(goal_point, 'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255),
            #                                  life_time=999)
            #     print("asdfasdf:", self.distance_memory)
            #     self.world.debug.draw_string(starting_waypoint.next(self.distance_memory)[0].transform.location, 'o',
            #                                  draw_shadow=True, color=carla.Color(r=0, g=0, b=255), life_time=999)
            return self.distance_memory


    # def asdf(self,first_lane_waypoint):
    #     if first_lane_waypoint is not None:
    #         while first_lane_waypoint.lane_id != -1:  # get waypoint of forth's lane
    #             #### for debug
    #             # pre_last_lane_waypoint = last_lane_waypoint
    #             # self.check = pre_last_lane_waypoint
    #             #### for debug
    #             first_lane_waypoint = first_lane_waypoint.get_left_lane()
    #             if first_lane_waypoint is None:
    #                 return False
    #
    #         # print("lane id : ", last_lane_waypoint.lane_id)
    #     else:
    #         print("getting first lane false2")
    #         return False
    #     print(first_lane_waypoint.transform.location)
    #
    #     return first_lane_waypoint


    def get_waypoint_of_first_lane(self,actor):
        first_lane_waypoint = self.map.get_waypoint(actor.get_location(),
                                                   lane_type=carla.LaneType.Driving)  # self.controller.waypoint

        if first_lane_waypoint is not None:
            while first_lane_waypoint.lane_id != -1:  # get waypoint of forth's lane
                #### for debug
                # pre_last_lane_waypoint = last_lane_waypoint
                # self.check = pre_last_lane_waypoint
                #### for debug
                first_lane_waypoint = first_lane_waypoint.get_left_lane()
                if first_lane_waypoint is None:
                    return False

            # print("lane id : ", last_lane_waypoint.lane_id)
        else:
            print("getting first lane false2")
            return False

        return first_lane_waypoint

    def get_waypoint_of_n_th_lane(self,actor,n):
        n_lane_waypoint = self.map.get_waypoint(actor.get_location(),
                                                   lane_type=carla.LaneType.Driving)  # self.controller.waypoint
        if n_lane_waypoint is not None:
            while abs(n_lane_waypoint.lane_id) != n:
                if abs(n_lane_waypoint.lane_id) > n:
                    n_lane_waypoint = n_lane_waypoint.get_left_lane()
                elif abs(n_lane_waypoint.lane_id) < n:
                    n_lane_waypoint = n_lane_waypoint.get_right_lane()

                if n_lane_waypoint is None:
                    return False

        else:
            print("getting first lane false2")
            return False

        return n_lane_waypoint

    def get_waypoint_of_last_lane(self,actor):
        last_lane_waypoint = self.map.get_waypoint(actor.get_transform().location, lane_type=carla.LaneType.Driving)  # self.controller.waypoint
        my_waypoint = last_lane_waypoint ## for debug

        if last_lane_waypoint is not None:
            while last_lane_waypoint.lane_id > -4:  # get waypoint of forth's lane
                #### for debug
                # pre_last_lane_waypoint = last_lane_waypoint
                # self.check = pre_last_lane_waypoint
                #### for debug
                last_lane_waypoint = last_lane_waypoint.get_right_lane()

                if last_lane_waypoint is None:
                    if abs(my_waypoint.lane_id) == 1 and self.ego_Lane >3:
                        return my_waypoint
                    else: ## for debug
                        print("getting last lane false")
                        # print("pre_last_lane_waypoint1 : ", pre_last_lane_waypoint.lane_id, "my_waypoint_id1:",
                        #       my_waypoint.lane_id,"ego_lane:",self.ego_Lane,"max_lane:",self.max_Lane_num)  ## for debug
                        return False

            # print("lane id : ", last_lane_waypoint.lane_id)
        else:
            print("getting last lane false2")
            return False
        # self.world.debug.draw_string(last_lane_waypoint.transform.location,
        #                                      'o', draw_shadow=True,
        #                                      color=carla.Color(r=255, g=255, b=0), life_time=9999)
        return last_lane_waypoint

    # def save_input_info(self,info):
    #     """
    #     :param info:[state, x_static]
    #         state.append(torch.tensor([dr/self.ROI_length, dv/10.0 , length/10.0, self.extra_dl_list[x]]))
    #         x_static = (torch.tensor([self.ego_Lane, self.controller.velocity / 100.0, lane_valid_distance / self.ROI_length]))
    #     :return:
    #     """
    #     info[0][0]

    def search_leading(self, info):
        extra_infos = info[0].reshape(-1,4)
        leading_extra_info = None
        for extra_info in extra_infos:
            dr = extra_info[0]*self.ROI_length
            dv = info[0][1]*10
            if extra_info[3] == 0:
                if dr <= self.controller.safe_distance:
                    if leading_extra_info is None:
                        leading_Extra_info = extra_info
                    elif leading_extra_info[0]*self.ROI_length > dr:
                        leading_extra_info = info

        return leading_extra_info


    def update_dv_v(self,new_input,leading_info,step):
        """

        :param new_input:
        :param leading_info:1x4  -> dynamic_state
        :param step:
        :return:
        """
        if leading_info is None:
            print("no leading info")
            # 70 = desired velocity of controller
            delta_dv = (70.0-new_input[1][1]*100)/2
            if delta_dv <=0:
                new_input[0][1::4] = new_input[0][1::4] + delta_dv / 10 #dv
                new_input[1][1] = new_input[1][1] + delta_dv / 100 #v
        else:
            print("leading info exist")
            dr = leading_info[0][0]*self.ROI_length
            leading_dv = leading_info[0][1]*10
            if dr >=30:
                delta_dv =  leading_dv/int(dr-30+step)
                new_input[0][1::4] = new_input[0][1::4]-delta_dv/10
                new_input[1][1] = new_input[1][1]-delta_dv/100

            else:
                new_input[1][1] = 0

        return new_input

    def plot_state_action(self,info):
        """
        :param info:[state, x_static]
            state.append(torch.tensor([dr/self.ROI_length, dv/10.0 , length/10.0, self.extra_dl_list[x]]))
            x_static = torch.tensor([self.ego_Lane, self.controller.velocity / 100.0, lane_valid_distance / self.ROI_length])
        :return:
        """
        data_list = []
        step = 1
        new_input = info
        for lane_num in range(1,5):
            lane_gap = info[1][0] - lane_num
            new_input[1][0] = lane_num # static - l
            new_input[0][3::4] = info[0][3::4] + lane_gap # dl
            for x in range(0, 200,step):
                new_input[0][0::4] = info[0][0::4] - x / self.ROI_length # dr
                new_input[1][2] = info[1][2] - x / self.ROI_length       #static - valid
                leading_info = self.search_leading(new_input)
                new_input =  self.update_dv_v(new_input,leading_info,step)

                state = new_input[0].type(torch.FloatTensor).cuda()
                x_static = new_input[1].type(torch.FloatTensor).cuda()
                max_q= self.agent.model(state, x_static).detach().max()
                data_list.append(max_q.item())
                # data_list.append(self.agent.model(state,x_static).detach()[0])
        data_list = tuple(data_list)
        data_list = torch.cat(data_list).tolist()

        data = np.reshape(data_list, (4, -1))
        ay = [1,2,3,4]
        # ax = list(np.linspace(-200,200,10))

        plot = sns.heatmap(data, xticklabels=ay)
        plt.show()

    def search_distance_valid(self,max_Lane_num):
        """
        distance_memory 's value where
        """
        # index= 4-self.ego_Lane
        # print("index :", index)

        third_lane_waypoint = self.get_waypoint_of_n_th_lane(self.player,4)
        if third_lane_waypoint == False:
            return None

        # print(self.index, self.max_Lane_num, self.pre_max_Lane_num, self.section)
        # print(self.ego_Lane)
        if self.max_Lane_num != self.pre_max_Lane_num:
            if self.index >= len(self.lane_change_point):
                self.index = 0
                # self.section = 0
            else:
                self.index += 1
                # self.section = int(self.index / 2)
                # if self.index == 8 :
                #     print("aa")
                # if self.index == 8 and self.distance_memory is None:  # it needs because of bug
                #     pass
                # else:

                # if self.distance_memory is None or abs(self.distance_memory) <=20:

            self.pre_max_Lane_num = self.max_Lane_num
        if self.index >= len(self.lane_change_point):
            self.index = 0
        # if self.index ==9: #bug not fixed completely. this is just a quick fix.
        #     self.index = 8

        # print(self.index)
        third_goal_waypoint = self.map.get_waypoint(self.lane_change_point[self.index], lane_type=carla.LaneType.Driving)
        # print("start")
        if self.index % 2 == 0: # even index = start point
            distance = self.get_distance_from_waypoint_to_goal_point(third_lane_waypoint, third_goal_waypoint.transform.location)
            if distance == False or distance == None:
                return None
            if distance <0:
                self.distance_memory = None
                distance = self.uclidian_distance(third_lane_waypoint.transform.location, self.lane_change_point[self.index+1])
                return -distance
            return distance
        else:
            self.distance_memory = None
            distance = self.uclidian_distance(third_lane_waypoint.transform.location, self.lane_change_point[self.index])
            return -distance

    def loose_safety_check(self,decision,safe_lane_change_again_time = 2.5):
        # min time interval for lane change
        condition1 = (time.time()-self.lane_change_time) >= safe_lane_change_again_time
        # min distance from ego to lane change waypoint
        # condition2 = (self.controller.is_lane_changing == True and self.controller.ld < self.controller.player_length+self.controller.waypoint.lane_width)

        if condition1:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = True


        else:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = False

        if decision != 0:
            if self.can_lane_change == False:
                return 0
            else:
                self.lane_change_time = time.time()
                return decision
        else:
            return 0

    def safety_check(self,decision, safe_lane_change_again_time=2.5):
        action = decision ############### if this line deleted, not alert error properly
        # a=(time.time() - self.lane_change_time)

        # min time interval for lane change
        condition1 = (time.time() - self.lane_change_time) >= safe_lane_change_again_time
        # min distance from ego to lane change waypoint

        # if self.controller.is_fin_to_lane_change:
        #     self.arrive_lane_chagne_change_point = True
        # if self.controller.is_start_to_lane_change:
        #     self.arrive_lane_chagne_change_point = False
        # print(condition1, self.arrive_lane_chagne_change_point)

        if condition1:# or self.arrive_lane_chagne_change_point:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = True

            # print("------------")
            # print("pre:", self.pre_can_lane_change)
            # print("now:", self.can_lane_change)
            # print("------------")


        else:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = False

            # print("------------")
            # print(decision)
            # print("pre:", self.pre_can_lane_change)
            # print("now:", self.can_lane_change)
            # print("------------")

            # print("finished lane change", self.pre_decision)
        # if action != -1 and x_static[0] == 4 and x_static[2] * self.ROI_length <= 5.0:
        #     self.decision_changed = True
        #
        #     if self.can_lane_change:
        #         print("hi1")
        #         return  -1
        #     else:
        #         print("hi2")
        #         return 0

        if decision !=0:
            if self.can_lane_change == False: # dont change frequently
                return 0 #즉 직진
            elif self.agent.selection_method == 'random' and decision == 1 and self.ego_Lane >= self.max_Lane_num-0.5: #dont leave max lane
                # print("ego_lane:",self.ego_Lane, "max_lane:",self.max_Lane_num, "decision:", self.decision)
                self.decision_changed = True
                action = random.randrange(-1,1)

            elif self.agent.selection_method == 'random' and decision == -1 and self.ego_Lane <=1.5: #dont leave min lane
                # print("ego_lane:",self.ego_Lane, "max_lane:",self.max_Lane_num, "decision:", self.decision)
                self.decision_changed = True
                action = random.randrange(0, 2)

            elif  self.agent.selection_method == 'max' and self.ego_Lane >= self.max_Lane_num-0.5 and decision == 1:
                # print("ego_lane:",self.ego_Lane, "max_lane:",self.max_Lane_num, "decision:", self.decision)
                self.decision_changed = True
                remained_action_list = self.agent.q_value[0][0:2]
                # print("remained_action_list:",remained_action_list)
                action = int(remained_action_list.argmax().item())-1

            elif self.agent.selection_method == 'max' and self.ego_Lane <= 1.5 and decision == -1:
                # print("ego_lane:",self.ego_Lane, "max_lane:",self.max_Lane_num, "decision:", self.decision)
                self.decision_changed = True
                remained_action_list =  self.agent.q_value[0][1:3]
                # print("remained_action_list:",remained_action_list)
                action =  int(self.agent.q_value[0][1:3].argmax().item())



            if action != 0:# and self.can_lane_change(action,state):
                self.lane_change_time = time.time()
                # if action != -1 and x_static[0] == 4 and x_static[2] * self.ROI_length <= 5.0:
                #     self.decision_changed = True
                #     print("hi1")
                #     action = -1
                return action
            else:
                # if action != -1 and x_static[0] == 4 and x_static[2] * self.ROI_length <= 5.0:
                #     self.decision_changed = True
                #     print("hi2")
                #
                #     self.lane_change_time = time.time()
                #     return -1

                return 0
        else:
            # if action != -1 and x_static[0] == 4 and x_static[2] * self.ROI_length <= 5.0:
            #     self.decision_changed = True
            #     print("hi3")
            #
            #     self.lane_change_time = time.time()
            #     return -1

            return 0

    def safety_check2(self,decision, safe_lane_change_again_time=2.5):
        self.side_leading_dr = None
        action = decision ############### if this line deleted, not alert error properly
        # a=(time.time() - self.lane_change_time)

        # min time interval for lane change
        condition1 = (time.time() - self.lane_change_time) >= safe_lane_change_again_time
        # min distance from ego to lane change waypoint

        # if self.controller.is_fin_to_lane_change:
        #     self.arrive_lane_chagne_change_point = True
        # if self.controller.is_start_to_lane_change:
        #     self.arrive_lane_chagne_change_point = False
        # print(condition1, self.arrive_lane_chagne_change_point)

        if condition1:# or self.arrive_lane_chagne_change_point:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = True

        else:
            self.pre_can_lane_change = self.can_lane_change
            self.can_lane_change = False

        if decision !=0:
            if self.agent.selection_method == 'max' and self.is_safe_action(decision)==False:
                action = self.agent.q_value[0].sort()[1][1].item()-1
                self.decision_changed = True
                if self.is_safe_action(action)==True:
                    pass
                else:
                    action = self.agent.q_value[0].sort()[1][0].item()-1
                    # assert action == 0, print(self.agent.q_value[0].sort(),"val:",self.agent.q_value[0].sort()[1][0].item())

            elif self.agent.selection_method == 'random' and self.is_safe_action(decision)==False:
                self.decision_changed = True
                while self.is_safe_action(action) == False:
                    action = random.randrange(-1,2)

            if action != 0:# and self.can_lane_change(action,state):
                self.lane_change_time = time.time()
                return action
            else:
                return 0
        else:
            return 0

    def is_safe_action(self,decision):
        if decision !=0:
            if self.can_lane_change == False: # dont change frequently
                # print("1")
                return True
            elif decision == 1 and self.ego_Lane >= self.max_Lane_num-0.5: #dont leave max lane
                # print("ego_lane:",self.ego_Lane,"max_lane:",self.max_Lane_num)
                # print("2")
                return False
            elif decision == -1 and self.ego_Lane <=1.5: #dont leave min lane
                # print("3")
                return False
            elif self.is_side_safe(decision) == False:
                # print("4")
                return False
            else:
                return True
        else:
            # print("5")
            return True

    def is_side_safe(self,decision):
        for num in range(self.extra_num):

            dv = 0
            if self.extra_dl_list[num] == decision :
                # return False
                player_vel = (self.player.get_velocity().x ** 2 + self.player.get_velocity().y ** 2 + self.player.get_velocity().z ** 2) ** 0.5
                extra_vel = self.extra_list[num].get_velocity()
                dv = player_vel - (extra_vel.x ** 2 + extra_vel.y ** 2 + extra_vel.z ** 2) ** 0.5
                # print("앞 안전거리:", self.controller.safe_distance/3 - dv ,"뒤 안전거리 :", self.controller.safe_distance / 4 - dv)
                # print("dr:",self.vehicles_distance_memory[num],"dv:",dv, "save_distance:",self.controller.safe_distance)
                self.side_lead_safe_distance = self.controller.safe_distance / 3 - dv
                self.side_back_safe_distance = self.controller.safe_distance / 6 - dv

                if self.vehicles_distance_memory[num] > 0 and abs(self.vehicles_distance_memory[num]) <= self.side_lead_safe_distance:
                    # self.world.debug.draw_string(self.extra_list[num].get_transform().location,
                    #                              'o', draw_shadow=True,
                    #                              color=carla.Color(r=255, g=0, b=0), life_time=0.1)
                    self.side_leading_dr = self.vehicles_distance_memory[num]
                    # print("1")
                    return False
                elif self.vehicles_distance_memory[num] < 0 and abs(self.vehicles_distance_memory[num]) <= self.side_back_safe_distance:
                    # self.world.debug.draw_string(self.extra_list[num].get_transform().location,
                    #                              'o', draw_shadow=True,
                    #                              color=carla.Color(r=255, g=0, b=0), life_time=0.1)
                    self.side_leading_dr = self.vehicles_distance_memory[num]
                    # print("2")
                    return False

            # self.world.debug.draw_string(self.controller.my_location_waypoint.next(int(20-dv))[0].transform.location,
            #                              'o', draw_shadow=True,
            #                              color=carla.Color(r=255, g=255, b=255), life_time=-1)
            # self.world.debug.draw_string(self.controller.my_location_waypoint.previous( self.controller.safe_distance / 6 - dv)[0].transform.location,
            #                              'o', draw_shadow=True,
            #                              color=carla.Color(r=255, g=255, b=255), life_time=0.1)
        # print("3")
        return True


    def get_follower_accel(self,actor,direction):
        waypoint = self.map.get_waypoint(actor.get_location(), lane_type=carla.LaneType.Driving)
        v_vector = actor.get_velocity()
        velocity = (v_vector.x ** 2 + v_vector.y ** 2 + v_vector.z ** 2) ** 0.5 * 3.6
        safe_distance = int(1.3 * velocity + 20)

        if direction ==0:
            search_point = 0.1
            ao = 0
            ao_bar = 0
            while search_point <= safe_distance:
                if waypoint is not None:
                    x = waypoint.previous(search_point)[0]
                else:
                    return False, False
                for extra in self.extra_list:
                    extra_pos = extra.get_location()
                    if self.uclidian_distance(x.transform.location, extra_pos)<=2.5:
                       ao = extra.get_acceleration()
                       if safe_distance >= x:
                            ao_bar = ao -0.01

                search_point += 1
            return ao, ao_bar


        elif direction == -1:
            an = 0
            an_bar = 0
            search_point = 0.1
            waypoint = waypoint.get_left_lane()
            while search_point <= safe_distance:
                if waypoint is not None:
                    x = waypoint.previous(search_point)[0]
                else:
                    return False, False

                for extra in self.extra_list:
                    extra_pos = extra.get_location()
                    if self.uclidian_distance(x.transform.location, extra_pos) <= 2.5:
                        an = extra.get_acceleration()
                        an = math.sqrt(an.x**2+an.y**2+an.z**2)
                        an_bar = an + 0.01
                search_point += 1
            return an, an_bar

        else:
            an = 0
            an_bar = 0
            search_point = 0.1
            waypoint = waypoint.get_right_lane()
            while search_point <= safe_distance:
                if waypoint is not None:
                    x = waypoint.previous(search_point)[0]
                else:
                    return False, False

                for extra in self.extra_list:
                    extra_pos = extra.get_location()
                    if self.uclidian_distance(x.transform.location, extra_pos) <= 2.5:
                        an = extra.get_acceleration()
                        an = math.sqrt(an.x ** 2 + an.y ** 2 + an.z ** 2)
                        an_bar = an + 0.01
                search_point += 1
            return an, an_bar
    #
    # def get_leading_vehicle_vel(self,actor):
    #     waypoint = self.map.get_waypoint(actor.get_location(), lane_type=carla.LaneType.Driving)
    #     v_vector = actor.get_velocity()
    #     velocity = (v_vector.x ** 2 + v_vector.y ** 2 + v_vector.z ** 2) ** 0.5 * 3.6
    #     safe_distance = int(1.3 * velocity + 20)
    #     # self.world.debug.draw_string(waypoint.next(safe_distance)[0].transform.location,
    #     #                                                           'o', draw_shadow=True,
    #     #                                                           color=carla.Color(r=255, g=255, b=0), life_time=0.1)
    #     search_point = 0.1
    #     for extra in self.extra_list:
    #         extra_pos = extra.get_location()
    #         v_vector = extra.get_velocity()
    #         velocity = (v_vector.x ** 2 + v_vector.y ** 2 + v_vector.z ** 2) ** 0.5 * 3.6
    #         while search_point <= safe_distance:
    #             x = waypoint.next(search_point)[0]
    #             # print(self.uclidian_distance(x.transform.location, extra_pos))
    #             if self.uclidian_distance(x.transform.location, extra_pos) <= waypoint.lane_width:
    #                 print(velocity)
    #                 self.world.debug.draw_string(extra.get_transform().location,'o', draw_shadow=True,color=carla.Color(r=255, g=255, b=0), life_time=-1)
    #
    #                 return velocity
    #             search_point += 1
    #
    #     # print("9999")
    #     return 9999

    # def mobil_lane_change(self, actor, direction):
    #     ae = actor.get_acceleration()
    #     ae= math.sqrt(ae.x**2+ae.y**2+ae.z**2)
    #     v_vector = actor.get_velocity()
    #     velocity = (v_vector.x ** 2 + v_vector.y ** 2 + v_vector.z ** 2) ** 0.5 * 3.6
    #     if self.get_leading_vehicle_vel(actor) > velocity:
    #         ae_bar = ae + 0.1
    #     elif self.get_leading_vehicle_vel(actor) < velocity:
    #         ae_bar = ae - 0.1
    #     else:
    #         ae_bar = ae
    #
    #     an, an_bar = self.get_follower_accel(actor, direction)
    #     ao, ao_bar = self.get_follower_accel(actor, 0)
    #     if an == False:
    #         return False
    #     elif ao == False:
    #         return False
    #     else:
    #         return self.mobil(ae,ae_bar,an, an_bar,ao,ao_bar)

    def rule_lane_change(self):
        if self.controller.leading_vehicle is not None:
            leading_actor = self.controller.leading_vehicle
            leading_vel = leading_actor.get_velocity()
            leading_vel = (leading_vel.x ** 2 + leading_vel.y ** 2 + leading_vel.z ** 2) ** 0.5 * 3.6
            target_vel = 70
            # print("leading_vel:",leading_vel,"agent_vel:",target_vel)
            # print(self.ego_Lane)

            if leading_vel <= target_vel:
                if self.is_safe_action(1):
                    self.deicison = self.loose_safety_check(1)
                elif self.is_safe_action(-1):
                    self.deicison = self.loose_safety_check(-1)
                else:
                    self.decision = 0

    # def mobil(self, ae, ae_bar, an, an_bar, ao, ao_bar, p=0, a_th=0.1, b_safe=4.0):
    #     """
    #     0 보다 크면 차선변경
    #     ae : double(m/s^2)
    #     acceleration ego vehicle+
    #     ae_bar : double(m/s^2)
    #     acceleraion ego vehicle if new lane
    #
    #     an : double(m/s^2)
    #     acceleration follower of target lane
    #
    #     an_bar : double(m/s^2)
    #     post acceleration follower of target lane
    #
    #     ao : double(m/s^2)
    #     acceleration follower of current lane
    #     ao_bar : double(m/s^2)
    #     post acceleration follower of current lane
    #
    #     p : double
    #     Politeness factor
    #
    #     a_th : double(m/s^2) # 0.1
    #     changing threshld
    #
    #     b_safe : double(m/s^2) # 4
    #     Maximum safe deceleraion
    #     """
    #
    #     a = ae_bar - ae + p * ((an_bar - an) + (ao_bar - ao))
    #     if a > a_th:
    #         return True
    #     else:
    #         return False
    #
    def get_max_lane(self,invalid_distances):
        """
        :param invalid_distances: int type list
        :return: thsi function change the variable, self.section and self.max_Lane_num
        """


        # print(aa," ", bb)

        if (self.distance_memory is None or self.distance_memory <0) and self.iter > 1:
            self.max_Lane_num = 3
        else:
            self.max_Lane_num = 4

        # if self.section >= len(invalid_distances): #len((invalid_distances) = 5
        #     self.section = 0

        # else:
            # distance = math.hypot(virtual_point.x-agent_cur_pos.x,virtual_point.y-agent_cur_pos.y)

    def visualize_virtual_invalid_forth_lane(self):
        for i in range(5):
            self.world.debug.draw_string(self.lane_start_point[i],
                                         'o', draw_shadow=True,
                                         color=carla.Color(r=255, g=255, b=0), life_time=9999)
            self.world.debug.draw_string(self.lane_finished_point[i],
                                         'o', draw_shadow=True,
                                         color=carla.Color(r=255, g=255, b=0), life_time=9999)

        # tmp = self.map.get_waypoint(self.lane_finished_point[3], lane_type=carla.LaneType.Driving).next(1340)[0].transform.location
        # self.world.debug.draw_string(tmp,'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255), life_time=9999)


    def main_test(self):
        simul_start_time = time.time()
        print(torch.cuda.get_device_name())
        # clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprint = random.choice(blueprints)

        # for i in range(1,4000):
        #     tmp=self.map.get_waypoint(self.player.get_location(), lane_type=carla.LaneType.Driving).next(i)[0].get_right_lane().transform.location
        #     self.world.debug.draw_string(tmp, 'o', draw_shadow=True,color=carla.Color(r=255, g=0, b=0), life_time=999)

            # ((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
            #         self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
            #          self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2) ** 0.5


        # for i in range(1,9000):
        #     tmp = self.spawn_waypoint.next(0.1)[0].get_right_lane().next(i)
            # if len(tmp)>1:
            #     waypoint = tmp[1]
            # else:
            #     waypoint = tmp[0]
            # waypoint = tmp[0]

        pre_decision = None
        # d = self.search_distance_valid()
        self.get_next_state(0)

        while True:

            self.iter+=1
            # print("right:",self.mobil_lane_change(self.player,1))
            # print("left:",self.mobil_lane_change(self.player,-1))
            # if self.mobil_lane_change(self.player,1):
            #     self.decision = 1
            # elif self.mobil_lane_change(self.player,-1):
            #     self.decision = -1
            # else:
            #     self.decision = 0

            # self.world.debug.draw_string(self.extra_list[0].get_transform().location,
            #                              'o', draw_shadow=True,
            #                              color=carla.Color(r=255, g=255, b=0), life_time=0.1)
            # a=self.extra_list[0].get_acceleration()
            # th = self.extra_list[0].get_transform().rotation.yaw
            # print(a.x*math.cos(math.radians(th))+a.y*math.sin(math.radians(th)))
            # print(self.extra_list[0].get_speed_limit())
            # a= self.extra_list[0].get_velocity()

            # print("vel :", math.sqrt(a.x**2+a.y**2+a.z**2)*3.6, "  limit :", self.extra_list[0].get_speed_limit())
            # print(self.search_section(self.player.get_location(),self.lane_start_point,self.lane_finished_point))

            # if time.time()-simul_start_time > 1:
            #     for extra in self.extra_list:
            #         tm.force_lane_change(extra,False)
            #     print("restart")
            #     self.restart()

            #     print(self.search_distance_valid())
            # print(self.controller.waypoint)

            # for num, extra in enumerate(self.extra_list):
            #     self.get_dr(extra,num)

            # print(self.vehicles_distance_memory)
            # if Keyboardcontrol.parse_events(client, self, clock):
            #     return

            self.spectator.set_transform(
                carla.Transform(self.player.get_transform().location + carla.Location(z=200),
                                carla.Rotation(pitch=-90)))

            # self.camera_rgb.render(display)
            # self.hud.render(display)
            # pygame.display.flip()

            # extra_pos = self.extra_list[0].get_transform().location
            # print(self.is_extra_front_than_this_point(extra_pos))

            ## Get max lane ##
            # print("start get lane")

            # d = self.search_distance_valid()


            # print("d:", d, "section:", self.section, "index", self.index, "max_lane", self.max_Lane_num)

            ## finished get max lane ##
            # print("finished get lane")

            # self.search_distance_valid()
            # print("distance :" ,,"max_lane_num : ", self.max_Lane_num)

            # if self.max_Lane_num==3:
                # self.world.debug.draw_string(self.player.get_transform().location,
                #                              'o', draw_shadow=True,
                #                              color=carla.Color(r=255, g=255, b=0), life_time=9999)\
            # stop after specific time
            # if time.time()-self.simul_time > 20:
            #     for extra in self.extra_list:
            #         extra.enable_constant_velocity(extra.get_transform().get_right_vector() * 0.0)

            # if time.time()-self.lane_change_time > 10:
            #     self.lane_change_time = time.time()
            #     if pre_decision is None:
            #         self.decision = -1
            #         # self.decision = self.loose_safety_check(self.decision)
            #         # self.ego_Lane -= 1
            #         print("after decision: ", self.decision, "after lane", self.ego_Lane)
            #         pre_decision = -1
            #
            #     elif pre_decision == 1:
            #         # self.decision = self.loose_safety_check(self.decision)
            #         pre_decision = -1
            #         # self.ego_Lane += -1
            #         self.decision = -1
            #
            #     elif pre_decision == -1:
            #         # self.decision = self.loose_safety_check(self.decision)
            #         self.decision = 1
            #         # self.ego_Lane += 1
            #         pre_decision = 1
            #
            #
            # else:
            #     self.decision = 0
            # self.get_mobil_vel :input()
            # if self.mobil() == True:
            print("distance memory:", self.distance_memory)

            self.controller.apply_control(self.decision)

            tmp = self.step(0)

            # self.step(self.decision)
            # if self.is_side_safe(1) == True:
            #     print("1")
            #     self.decision = 1
            # elif self.is_side_safe(-1) == True:
            #     print("-1")
            #     self.decision = -1
            # clock.tick()
            for extra_actor in self.extra_list:
                if extra_actor.is_at_traffic_light():
                    traffic_light = extra_actor.get_traffic_light()
                    if traffic_light is not None and traffic_light.get_state() == carla.TrafficLightState.Red:
                        # world.hud.notification("Traffic light changed! Good to go!")
                        traffic_light.set_state(carla.TrafficLightState.Green)
            self.world.tick()


            # self.hud.tick(self, clock)

    def main(self):

        PATH = "/home/a/version_2_per_deepset/"
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.agent = decision_driving_Agent(self.input_size,self.output_size,True,1000,self.extra_num,self.controller)

        # # 모델의 state_dict 출력
        # print("Model's state_dict:")
        # for param_tensor in self.agent.model.state_dict():
        #     print(param_tensor, "\t", self.agent.model.state_dict()[param_tensor].size())
        #
        # # 옵티마이저의 state_dict 출력
        # print("Optimizer's state_dict:")
        # for var_name in self.agent.optimizer.state_dict():
        #     print(var_name, "\t", self.agent.optimizer.state_dict()[var_name])


        self.lane_change_point = []

        previous_reward = None

        for i in range(len(self.lane_start_point)):
            self.lane_change_point.append(self.lane_start_point[i])
            self.lane_change_point.append(self.lane_finished_point[i])
        # for x in self.lane_change_point:
        #     self.world.debug.draw_string(x, 'o', draw_shadow=True, color=carla.Color(r=0, g=255, b=0), life_time=9999)

        epoch = 0
        device = torch.device('cuda')

        load_dir = PATH+'trained_info4123084.pt'
        if(os.path.exists(load_dir)):

            print("저장된 가중치 불러옴")
            checkpoint = torch.load(load_dir)
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.model.to(device)
            self.agent.target_model.load_state_dict((checkpoint['target_model_dict']))
            self.agent.target_model.to(device)
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.agent.buffer.buffer = checkpoint['data']
            # self.agent.buffer.buffer = checkpoint['memorybuffer']
            epoch = checkpoint['epoch']
            self.agent.epsilon = checkpoint['epsilon']

        # print("h")
        if self.mission_mode == True:
            self.agent.is_training = False

        print("epoch : ",epoch)
        # print(self.agent.buffer.size)

            # self.agent.buffer.load_state_dict(self.save_dir['memorybuffer'])

            # self.is_first_time = False
        # controller = VehicleControl(self, True)
        # self.player.apply_control(carla.Vehicle
        # Control(throttle=1.0, steer=0.0, brake=0.0))
        # vehicles = self.world.get_actors().filter('vehicle.*')
        # for i in vehicles:
        #     print(i.id)
        # try:


        while True:
            simulation_step = 0
            # print("start epoch!")
            self.start_epoch = True
            self.iter +=1

            [state, x_static] = self.get_next_state()  # 초기 상태 s0 초기화
            if state is None:
                print("state initialize error")
                self.restart()
                self.start_epoch = False

            while self.start_epoch:
                # self.hud.tick(self, clock)

                if Keyboardcontrol.parse_events(client, self, clock):
                    return

                # self.camera_rgb.render(display)
                # self.hud.render(display)
                # pygame.display.flip()
                # tmp = self.map.get_waypoint(self.player.get_location(), lane_type=carla.LaneType.Driving)
                # tmp_rotation = tmp.transform.rotation
                # tmp2=self.get_waypoint_of_last_lane(self.player)
                self.spectator.set_transform(carla.Transform(self.player.get_transform().location + carla.Location(z=200),
                                    carla.Rotation(pitch = -90)))

                ## finished get max lane ##
                ##visualize when, max lane ==3 ##

                # [self.extra_list, self.1] = self.agent.search_extra_in_ROI(self.extra_list,self.player,self.extra_dl_list)

                ## finished to visualize ##
                # self.agent.is_training = False
                if self.agent.is_training :
                    ##dqn 과정##
                    # 가중치 초기화 (pytroch 내부)
                    # 입실론-그리디 행동 탐색 (act function)
                    # 메모리 버퍼에 MDP 튜플 얻기   ㅡ (step function)
                    # 메모리 버퍼에 MDP 튜플 저장   ㅡ
                    # optimal Q 추정             ㅡ   (learning function)
                    # Loss 계산                  ㅡ
                    # 가중치 업데이트              ㅡ



                    # print(clock.get_fps())

                    # if time.time() - self.simul_time > 7 and time.time() - self.simul_time < 8 and clock.get_fps() < 15:
                    #     self.restart()
                    #     self.start_epoch = False
                    #     break
                    # self.decision_changed = False

                        #online learning
                        # if len(self.agent.buffer.size()) > self.agent.batch_size:
                        #     simulation_step += 1
                        #     self.agent.learning()
                        #     self.acummulated_loss += self.agent.loss

                    self.decision_changed = False

                    if self.can_lane_change == False: #self.controller.is_lane_changing == True and self.controller.is_start_to_lane_change == False
                        self.decision = 0
                        # print(0)
                    else:
                        self.decision= self.agent.act(state, x_static)

                    before_safety_decision = self.decision

                    if self.safety_mode == True:
                        self.decision = self.safety_check(self.decision)
                    else:
                        self.decision = self.loose_safety_check(self.decision)

                    if self.decision != 0:
                        self.check += 1
                    if self.decision_changed:
                        self.check2 += 1
                    if self.decision != 0:
                        self.check2 += 1


                    # print("before:", before_safety_decision)
                    # print("after:", self.decision)

                    is_error = self.controller.apply_control(self.decision)
                    if is_error:
                        print("controller error")
                        self.restart()
                        self.start_epoch = False
                    # for i in range(len(self.extra_controller_list)):
                    #     self.extra_controller_list[i].apply_control()

                    # if self.decision == -1 and self.ego_Lane == 1:

                    clock.tick(40)

                    tmp = self.step(self.decision)

                    self.pre_ego_lane = self.ego_Lane


                    if tmp is None:
                        print("get_state_error in step process")
                        self.restart()
                        self.start_epoch = False
                        break

                    else:
                        [__, _, decision, reward, next_state, next_x_static, done] =  tmp
                        # if self.check == 1:
                        #     sample = [state, x_static, self.decision, reward, next_state, next_x_static, done]
                        #     # self.plot_state_action(sample)
                        #     self.restart()
                        #     self.start_epoch = False

                    if done:
                        # x_static[0] = self.current_ego_lane
                        # info = [state, x_static]
                        # if no collision no data stored in buffer

                        f = open("/home/a/version_2_per_deepset/data/lane_change_num.txt", 'a')
                        # 시나리오 반복 횟수, 미션 성공수    , 소요 시간,      평균 속도, 	차선 변경 횟수,	   left, 	    right,    exit colision   vehicle collision  퍙군속도 구하는데 들어간 iteration 수
                        data_list = [epoch, self.check]
                        for data in data_list:
                            input = "%f \t" % data
                            f.write(input)
                        f.write("\n")
                        f.close()

                        if len(self.collision_sensor.history) != 0:
                            assert reward is not None, "reward= none"
                            sample = [decision_state, decision_x_static, self.pre_decision, reward, None, None,
                                      done]
                            # print("collision:", decision_x_static[0], "decision:", self.pre_decision, "reward",
                            #       reward)
                            # print(sample)

                            self.agent.buffer.append(sample)
                            self.agent.memorize_td_error(0)

                        elif self.is_lane_out:
                            sample = [state, x_static, self.decision, reward, None, None,
                                      done]
                            self.agent.buffer.append(sample)
                            self.agent.memorize_td_error(0)
                        ##-------------
                        # if self.side_leading_dr is not None:
                        #
                        #     if self.side_leading_dr >= 0:
                        #         d = self.side_lead_safe_distance
                        #         x = self.side_leading_dr.item()
                        #         reward = - 20 / (d ** 3) * x ** 3 + 30 / (d ** 2) * (x ** 2) - 2
                        #
                        #     else:
                        #         d = self.side_back_safe_distance
                        #         x = abs(self.side_leading_dr.item())
                        #         reward = -20 / (d ** 3) * x ** 3 + 30 / (d ** 2) * (x ** 2) - 2
                        #
                        #     print("dr:",x, "d: ",d, "reward: ",reward)
                        #
                        #     sample = [state, x_static, before_safety_decision, reward, None, None, done]
                        #     self.agent.buffer.append(sample)
                        #     self.agent.memorize_td_error(0)
                        ##-------------


                        elif self.decision_changed:
                            # if x_static[0] <=1.5:+
                            #     print("lane 1 :", before_safety_decision)
                            # elif x_static[lane_change_fin, straight:0]>3.5:
                            #     print("lane 4 :", before_safety_decision)

                            sample = [state, x_static, before_safety_decision, -10, None, None, done]
                            # print(sample)

                            self.agent.buffer.append(sample)
                            self.agent.memorize_td_error(0)
                        # else:
                        #
                        #     self.plot_state_action(sample)



                        print("buffer size : ", len(self.agent.buffer.size()))
                        n=150.0


                        self.agent.buffer.plot_buffer(epoch)

                        print("epsilon :", self.agent.epsilon)
                        print("learning_rate:",self.agent.learning_rate)
                        # print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)

                        # if epoch == 50:
                        #     self.agent.learning_rate = 0.0005
                        # elif epoch == 100:
                        #     self.agent.learning_rate/= 0.0001
                        # elif epoch == 500:
                        #     self.agent.learning_rate/= 0.00005
                        self.agent.update_td_error_memory(epoch)


                        #offline learning
                        if len(self.agent.buffer.size()) > 2000:#self.agent.batch_size * n:

                            print("start learning")
                            for i in range(int(n)):
                                self.agent.ddqn_learning()
                                self.acummulated_loss += self.agent.loss
                            # self.scenario = "validation"
                            # self.agent.is_training = False
                            writer.add_scalar('Loss', self.acummulated_loss / n, epoch)
                            writer.add_scalar('랜덤 시나리오 누적보상', self.accumulated_reward , epoch)
                            self.agent.update_epsilon()


                        if epoch % 10 == 0:
                            self.agent.target_model.load_state_dict(self.agent.model.state_dict())

                        client.set_timeout(10)
                        if self.mission_mode == True:
                            if epoch % 1 == 0:
                                # [w, b] = self.agent.model.parameters()  # unpack parameters
                                self.save_dir = torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.agent.model.state_dict(),
                                    'target_model_dict': self.agent.target_model.state_dict(),
                                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                                    'data': self.agent.buffer.buffer,
                                    'left_col': self.left_col_num,
                                    'right_col':self.right_col_num,
                                    'exit_col':self.exit_lane_col_num,
                                    'vehicle_col': self.collision_num,
                                    'clear_num': self.mission_clear_num,
                                    'iter': self.iter,
                                    # 'memorybuffer': self.agent.buffer.buffer,
                                    'epsilon': self.agent.epsilon},
                                    PATH + "mission_info" + str(epoch) + ".pt")  # +str(epoch)+
                        if self.safety_mode == True:
                            if epoch % 1 == 0:
                                # [w, b] = self.agent.model.parameters()  # unpack parameters
                                self.save_dir = torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.agent.model.state_dict(),
                                    'target_model_dict': self.agent.target_model.state_dict(),
                                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                                    'data': self.agent.buffer.buffer,
                                    # 'memorybuffer': self.agent.buffer.buffer,
                                    'epsilon': self.agent.epsilon},
                                    PATH + "safe_train_info" + str(epoch) + ".pt")  # +str(epoch)+


                        self.restart()
                        self.start_epoch = False
                        # self.camera_rgb.toggle_camera()
                        # print("toggle_camera finished")

                    else:

                        if self.controller.is_lane_changing == False:
                            if self.controller.is_fin_to_lane_change:
                                self.append_lane_change_sample = True

                                # print("condition satisfied decision :", self.decision) #if not zero -> bug

                                # 예외처리 : 시뮬레이션 시작하자 말자 차선변경 하는 경우

                                # lane change finished , append sample
                                # print("lane change complete")
                                # self.check += 1
                                # print(self.can_lane_change) # is_fin_to_lane_change ->next step -> can_lane_change == Ture



                            if self.can_lane_change and self.append_lane_change_sample:
                                #finished to lane change##
                                # self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
                                #                              color=carla.Color(r=0, g=255, b=255), life_time=99)
                                self.append_lane_change_sample = False
                                sample = [decision_state, decision_x_static, self.pre_decision, decision_reward,
                                          next_state, next_x_static, done]
                                # print(sample)
                                # print("1: lane_change_fin,save when decision time:  ",self.pre_decision,decision_x_static[0].item(), next_x_static[0].item(),decision_state[3::4],next_state[3::4])

                                assert decision_reward is not None, "reward= none"
                                self.agent.buffer.append(sample)
                                self.agent.memorize_td_error(0)

                            # print("fin to lane change and store lane change sample, decision =", decision)

                            if self.decision == 0 and self.can_lane_change:
                                # print("store straight sample")
                                # self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
                                #                              color=carla.Color(r=0, g=0, b=255), life_time=999)
                                sample = [state, x_static, self.decision, reward, next_state, next_x_static, done]
                                # assert self.decision == 0, 'decision ix_static : s not 0'
                                assert reward is not None, "reward= none"
                                # print(sample)

                                self.agent.buffer.append(sample)
                                # print(self.can_lane_change)
                                self.agent.memorize_td_error(0)
                                # self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
                                #                              color=carla.Color(r=255, g=255, b=255), life_time=99)

                                # print("2:  ",x_static)
                                # print("straight:  ", x_static[0], self.decision, next_x_static[0])



                        else: #is_lane_changing = True
                            if self.decision != 0:
                                # print("incomplete lane change and restart lane change")
                                # self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
                                #                              color=carla.Color(r=0, g=255, b=0), life_time=99)
                                if previous_reward is None:
                                    previous_reward = reward
                                    #previous_reward -> negative reward ..?
                                sample = [decision_state, decision_x_static, self.pre_decision, previous_reward-0.01, next_state, next_x_static, done]

                                assert previous_reward is not None, "reward= none"

                                # print("3:  ",x_static)

                                # print("3:  ",decision_x_static[0], self.pre_decision, x_static[0])
                                # print("3, x_static : ", x_static[0].item())

                                # print(sample)

                                self.agent.buffer.append(sample)
                                self.agent.memorize_td_error(0)
                            # else:
                            #     ## lane changing
                            #     self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
                            #                                  color=carla.Color(r=10, g=10, b=105), life_time=99)


                            # print("lane_valid_distance :",x_static[2]*self.ROI_length,"reward:",reward,"decision:",decision,"before decision",before_safety_decision)
##-------------------------------- 안전 거리 미확보 시 안전 데이터
                        # if self.side_leading_dr is not None:
                        #
                        #     if self.side_leading_dr>=0:
                        #         d = self.side_lead_safe_distance
                        #         x=  self.side_leading_dr.item()
                        #         reward =- 20 / (d ** 3) * x ** 3 + 30 / (d ** 2) * (x ** 2) - 2
                        #
                        #     else:
                        #         d = self.side_back_safe_distance
                        #         x = abs(self.side_leading_dr.item())
                        #         reward = -20/(d**3) * x**3+30/(d**2)*(x**2) -2

                            # print("dr:",x, "d: ",d, "reward: ",reward)
###----------------------------------
                            sample = [state, x_static, before_safety_decision, reward, None, None, done]
                            self.agent.buffer.append(sample)
                            self.agent.memorize_td_error(0)

                        if self.decision_changed:
                            # if x_static[0] <=1.5:
                            #     print("lane 1 :", before_safety_decision)
                            # elif x_static[0]>3.5:
                            #     print("lane 4 :", before_safety_decision)

                            sample = [state, x_static, before_safety_decision, -10, None, None, done]

                            # print("4:  ", x_static)
                            # print(sample)

                            self.agent.buffer.append(sample)
                            self.agent.memorize_td_error(0)
                            # print(self.decision, "can_lane_change:",self.can_lane_change)

                        if self.decision !=0:
                            # need to plot string
                            # print("decision static changed")
                            decision_state = state
                            decision_x_static = x_static
                            decision_reward = reward
                            self.pre_decision = self.decision

                        state = next_state
                        x_static = next_x_static

                else: #not training

                    self.decision_changed = False
                    # print(self.decision)

                    if self.mission_mode == False:
                        if self.can_lane_change == False:  # self.controller.is_lane_changing == True and self.controller.is_start_to_lane_change == False
                            self.decision = 0
                            # print(0)
                        else:
                            self.decision = self.agent.act(state, x_static)
                        self.decision = self.loose_safety_check(self.decision)
                    else:
                        #--
                        if self.can_lane_change == False:  # self.controller.is_lane_changing == True and self.controller.is_start_to_lane_change == False
                            self.decision = 0
                            # print(0)
                        else:
                            self.decision = self.agent.act(state, x_static)
                        self.decision = self.safety_check(self.decision)
                        #--
                        # self.rule_lane_change()
                        #--
                    # print(self.decision)

                    is_error = self.controller.apply_control(self.decision)
                    if is_error:
                        # print("validation fin")
                        if self.mission_mode == False:
                            self.scenario = "random"
                            self.agent.is_training = True
                            writer.add_scalar('누적 보상 ', self.accumulated_reward, epoch)
                            epoch += 1
                        self.restart()
                        self.start_epoch = False

                    tmp = self.step(self.decision)

                    if tmp is None:
                        print("get_state_error in step process")
                        self.restart()
                        self.start_epoch = False
                        break

                    else:
                        [__, _, decision, reward, next_state, next_x_static, done] = tmp

                    # if time.time() - self.simul_time >= 5:
                    #     print("screenshot comeplete")
                    #     pyautogui.screenshot('/home/a/Pictures/my_screenshot.png')
                    #     sample = [state, x_static, self.decision, reward, next_state, next_static, done]
                    #     f = open("/home/a/version_2_per_deepset/data/state_static.txt", 'w')
                    #     data = "%d\n" % epoch
                    #     f.write(data)
                    #     step = 0
                    #     for x in state:
                    #         step+=1
                    #         f.write("%f " % x)
                    #         if step==4:
                    #             step = 0
                    #             f.write('\n')
                    #     for x in x_static:
                    #         f.write("%f " % x)
                    #     f.write('\n')
                    #
                    #     f.close()
                    #     # self.plot_state_action(sample)
                    #     self.restart()
                    #     self.start_epoch = False

                    state = next_state
                    x_static = next_x_static

                    if done:
                        self.eplased_time = time.time() - self.simul_time
                        if self.mission_mode == True:
                            f = open("/home/a/version_2_per_deepset/data/new_mission_safe1.txt", 'a')
        #시나리오 반복 횟수, 미션 성공수    , 소요 시간,      평균 속도, 	차선 변경 횟수,	   left, 	    right,    exit colision   vehicle collision  퍙군속도 구하는데 들어간 iteration 수
                            data_list = [self.iter, self.mission_clear_num, self.eplased_time,self.controller.vel_history,self.controller.lane_change_history,self.left_col_num, self.right_col_num, self.exit_lane_col_num, self.collision_num, self.controller.history_num]
                            for data in data_list:
                                input = "%f \t" % data
                                f.write(input)
                            f.write("\n")
                            f.close()
                        else:
                            self.scenario = "random"
                            self.agent.is_training = True
                            print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)
                            writer.add_scalar('누적 보상 ', self.accumulated_reward, epoch)
                            epoch += 1
                        client.set_timeout(10)
                        self.restart()
                        self.start_epoch = False

world = None
actors=[]
sensors = []
if __name__ == '__main__':
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world =client.load_world('Town04_Opt')
        client.set_timeout(10.0)

        weather = carla.WeatherParameters(
            cloudiness=50.0,
            precipitation=0.0,
            sun_altitude_angle=0.0)

        world.unload_map_layer(carla.MapLayer.All)
        world.set_weather(weather)
        tm = client.get_trafficmanager(8000)
        tm_port = tm.get_port()
        env=CarlaEnv(world)
    # except:
    #
    #     if len(sensors) != 0:
    #         # print('destroying actors.')
    #         # print("actor 제거 :", self.actor_list)
    #         for sensor in sensors:
    #             if sensor.is_listening:
    #                 sensor.stop()
    #             sensor.destroy()
    #
    #     if len(actors) != 0:
    #         for x in actors:
    #             try:
    #                 client.apply_batch([carla.command.DestroyActor(x.id)])
    #             except:
    #                 print("destroy error ")



    except KeyboardInterrupt:
        if len(sensors) != 0:
            # print('destroying actors.')
            # print("actor 제거 :", self.actor_list)
            for sensor in sensors:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
        if len(actors) != 0:
            for x in actors:
                try:
                    client.apply_batch([carla.command.DestroyActor(x.id)])
                except:
                    print("destroy error ")

        print('\nCancelled by user. Bye!')

