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
import cv2
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
writer = SummaryWriter()
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
        self.safety_mode = True
        #화면 크기
        self.start_epoch = True
        self.input_size = 4  # dr dv da dl
        self.output_size = 3
        n_iters = 100
        self.width = 800
        self.height = 600
        #센서 종류
        self.actor_list = []
        self.extra_list = []
        self.extra_list = []
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
        self.hud = HUD(self.width, self.height)
        self.spawn_waypoints = self.map.generate_waypoints(3.0)
        self.lane_change_time = time.time()
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
        self.offline_learning_epoch = 0
        self.accumulated_loss = 0

        self.end_point = 0
        self.ROI_length = 100.0 #(meters)
        self.safe_lane_change_distance = 0
        self.search_radius = None
        self.left_search_radius = None
        self.right_search_radius = None
        self.decision_changed = False
        self.check = 0
        self.update_lane =0

        ## visualize all waypoints ##
        # for n, p in enumerate(self.spawn_waypoints):
        #     world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=999)

        settings = self.world.get_settings()
        # settings.no_rendering_mode = True
        # settings.synchronous_mode = True
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
        self.pre_max_Lane_num = self.max_Lane_num
        self.restart()
        self.main()

    def restart(self):
        self.decision = 0
        self.pre_decision = 0
        self.decision_changed = False
        self.simul_time = time.time()
        self.distance_memory = None
        self.lane_change_time = time.time()-100.0
        self.can_lane_change = True
        self.max_Lane_num = 4
        self.ego_Lane = 2
        self.pre_ego_lane = self.ego_Lane
        self.controller = None
        self.accumulated_reward = 0
        self.acummulated_loss = 0
        self.offline_learning_epoch = 0
        self.section = 0
        self.episode_start = time.time()
        self.pre_max_Lane_num = self.max_Lane_num
        self.index = 0
        # print('start destroying actors.')

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

        self.actor_list = []
        self.extra_list = []
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






        self.player = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')),
            start_pose)
        # print(self.player.bounding_box) # ego vehicle length

        self.actor_list.append(self.player)

        # self.camera_rgb =RGBSensor(self.player, self.hud)
        # self.actor_list.append(self.camera_rgb.sensor)

        # self.camera_depth =DepthCamera(self.player, self.hud)
        # self.actor_list.append(self.camera_depth.sensor)

        # self.camera_semseg = SegmentationCamera(self.player,self.hud)
        # self.actor_list.append(self.camera_semseg.sensor)

        self.collision_sensor = CollisionSensor(self.player, self.hud)  # 충돌 여부 판단하는 센서
        self.actor_list.append(self.collision_sensor.sensor)

        # self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)  # lane 침입 여부 확인하는 센서
        # self.actor_list.append(self.lane_invasion_sensor.sensor)

        # self.gnss_sensor = GnssSensor(self.player)
        # self.actor_list.append(self.gnss_sensor.sensor)

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
            distance_step = 20
            end = distance_step * (self.extra_num) + 1
            for i in range(distance_step, end, distance_step):
                dl=random.choice([-1,0,1])
                self.extra_dl_list.append(dl)
                if dl==-1:
                    spawn_point = self.spawn_waypoint.next(i)[0].get_left_lane().transform
                elif dl==0:
                    spawn_point = self.spawn_waypoint.next(i)[0].transform
                elif dl==1:
                    spawn_point = self.spawn_waypoint.next(i)[0].get_right_lane().transform
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                # print(blueprint_library.filter('vehicle.bmw.grandtourer'))
                # blueprint = random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))

                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))

                if blueprint.has_attribute('color'):
                        # color = random.choice(blueprint.get_attribute('color').recommended_values)
                        # print(blueprint.get_attribute('color').recommended_values)
                        color = '255,255,255'
                        blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint,spawn_point)
                self.extra_list.append(extra)

            spawn_point = carla.Transform(carla.Location(x=14.797643, y=-163.310318, z=2.000000),
                                          carla.Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000))
            blueprint = random.choice(blueprints)

            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    tm = client.get_trafficmanager(8000)
                    tm_port = tm.get_port()
                    tm.auto_lane_change(extra, False)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 30  # random.randrange(10, 40) # km/h
                    extra.set_target_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)

                    # extra.enable_constant_velocity(extra.get_transform().get_forward_vector() * target_velocity / 3.6)
                    extra.set_autopilot(True,tm_port)

                    self.world.constant_velocity_enabled = True





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
                if i < 4:
                    self.extra_list.append(extra)
                else:
                    trash_extra = extra
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.spawn_waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 20.0

                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
            trash_extra.enable_constant_velocity(extra.get_transform().get_right_vector() * 0.0 / 3.6)
            trash_extra.destroy()

                    # traffic_manager.auto_lane_change(extra,False)
        # print('Extra Genration Finished')


        self.spectator.set_transform(carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                            carla.Rotation(pitch=-90)))

        # extra_target_velocity = 10
        port = 8000
        # traffic_manager = client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(100.0)
        # traffic_manager.set_synchronous_mode(True) # for std::out_of_range eeror
        # tm_port = traffic_manager.get_port()

        # print("tm setting finished")
        # self.player.set_autopilot(True,tm_port)
        # traffic_manager.auto_lane_change(self.player, False)
        self.controller = Pure_puresuit_controller(self.player, self.spawn_waypoint, self.extra_list, 70)  # km/h
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


    def is_extra_front_than_ego(self,extra_pos):

        extra_pos_tensor = torch.tensor([[extra_pos.x, extra_pos.y, extra_pos.z, 1.0]])

        theta_z =  math.radians(self.player.get_transform().rotation.yaw)
        trans = self.player.get_transform().location

        n = torch.tensor([[math.cos(theta_z), math.sin(theta_z), 0]])
        o = torch.tensor([[-math.sin(theta_z),  math.cos(theta_z), 0]])
        a = torch.tensor([[0.0, 0.0, 1.0]])
        p = torch.tensor([[trans.x, trans.y, trans.z]])

        m41 = -torch.dot(torch.squeeze(n),torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
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

        plc = 0.005
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

        tmp = self.get_next_state(decision)  # get now state

        if tmp is None:
            return None
        else:
            next_state, next_x_static = tmp

        # end_length=math.sqrt((self.end_point.x - self.player.get_location().x)**2+(self.end_point.y - self.player.get_location().y)**2)
        done = False
        if len(self.collision_sensor.history) != 0:
            done = True
            print("collision")
            reward = -3
        elif time.time()-self.simul_time > 25:
            print("simultime done")
            done = True
            reward = 0
        # elif self.max_Lane_num < self.ego_Lane:
        #     done = True
        #     reward = -3
        else:
            reward = 0.09-1/2*abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel)-plc
            # print(abs(self.controller.desired_vel-self.controller.velocity)/(self.controller.desired_vel))
        if self.safety_mode == False:
            if decision == 1 and self.ego_Lane >= self.max_Lane_num - 0.5:  # dont leave max lane
                done = True
                print("lane right collision")
                reward = -3
            elif decision == -1 and self.ego_Lane <= 1.5:  # dont leave min lane
                done = True
                print("lane left collision")
                reward = -3
        else:
            if next_x_static[2] < 0 and self.ego_Lane >= self.max_Lane_num - 0.5:  # dont leave max lane
                done = True
                print("Agent get into exit, Done")
                reward = -3
            elif decision == -1 and self.ego_Lane <= 1.5:  # dont leave min lane
                assert False, "Agent leaves min lane"

        # print(reward)
        # if self.decision_changed == True:
        #     reward -= -1
        self.accumulated_reward += 1+reward

        #state length = 4 * num of extra vehicles + 1


        return None, None, decision ,reward, next_state, next_x_static , done
                                        # Next State 표현 필요
    def search_distance_between_vehicles(self,extra_forth_lane_waypoint,from_,to_):
        """
        :param search_raidus:
        :param extra: extra actor
        :param from_: data type = 'extra actor' or ego_forth_lane_waypoint
        :param to_:
        :return: distance from_ to to_
        """

        distance=9999
        sign = 0
        search_raidus = 5
        i = 0.01

        # self.world.debug.draw_string(ego_lane_waypoint.transform.location,
        #                                   'o', draw_shadow=True,
        #                                  color=carla.Color(r=0, g=0, b=255), life_time=-1)

        if from_ == extra_forth_lane_waypoint:
            from_waypoint = extra_forth_lane_waypoint
            to_waypoint =  to_
            sign = -1
        else:
            from_waypoint = to_
            to_waypoint = extra_forth_lane_waypoint
            sign = 1

        while search_raidus <= distance:
            # pre_distance = distance
            distance = self.uclidian_distance(from_waypoint.next(i)[0].transform.location, to_waypoint.transform.location)


            if round(distance - search_raidus) > 0:
                i += round(distance - search_raidus)
                # print("i : " ,i)
                # break
            else:
                # self.world.debug.draw_string(from_waypoint.next(distance + i)[0].transform.location,
                #                              'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255), life_time=-1)
                # print((distance + i)*sign)
                # print("finish")
                return (distance + i)*sign

    def get_dr(self):

        dr= np.zeros(len(self.extra_list))
        ego_forth_lane = self.get_waypoint_of_last_lane(self.player)

        # self.world.debug.draw_string(ego_forth_lane.transform.location,
        #                              'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=-1)
        for index, extra in enumerate(self.extra_list):

            extra_pos = extra.get_transform().location
            extra_forth_lane_waypoint = self.get_waypoint_of_last_lane(extra)
            self.world.debug.draw_string(extra_forth_lane_waypoint.transform.location,
                                             'o', draw_shadow=True,
                                             color=carla.Color(r=255, g=0, b=0), life_time=-1)
            if self.is_extra_front_than_ego(extra_pos) == False:
                dr[index] = self.search_distance_between_vehicles(extra_forth_lane_waypoint,extra_forth_lane_waypoint,ego_forth_lane)
            else:
                dr[index] = self.search_distance_between_vehicles(extra_forth_lane_waypoint,ego_forth_lane,extra_forth_lane_waypoint)
        return dr

                # print("lane id : ", forth_lane_waypoint.lane_id)

    def get_next_state(self,decision=None):
        """
        dl : relative lane num after ching the lane
        dr, dv, da : now state
        """
        state = []

        for x, extra in enumerate(self.extra_list):

            if decision == 1:
                self.extra_dl_list[x] =self.extra_dl_list[x]+1
                # self.ROI_extra_dl_list[x] =self.ROI_extra_dl_list[x]+1

            elif decision == -1:
                self.extra_dl_list[x] = self.extra_dl_list[x]-1
                # self.ROI_extra_dl_list[x] = self.ROI_extra_dl_list[x]-1

            else:
                pass

            extra_pos = extra.get_transform().location
            extra_vel = extra.get_velocity()
            # extra_acel = extra.get_acceleration()
            # sign = 0
            if self.is_extra_front_than_ego(extra_pos) == True:
                sign = 1
            else:
                sign = -1
            cur_agent_pos = self.player.get_transform().location
            dr = sign * self.uclidian_distance(cur_agent_pos,extra_pos) - abs(self.spawn_waypoint.lane_width*(self.extra_dl_list[x]))
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


        if self.can_lane_change == True:
            # print("finished lane change", self.pre_decision)
            if self.pre_decision == 1.0:
                # print("finish right")
                self.ego_Lane = math.ceil(self.ego_Lane) / 1.0
            elif self.pre_decision == -1.0:
                # print("finish left")
                self.ego_Lane = math.floor(self.ego_Lane) / 1.0

        self.update_lane = self.ego_Lane

        if decision == 1 :
            self.ego_Lane += 0.5
            self.pre_decision = 1.0
            self.can_lane_change = False
        elif decision ==-1:
            self.ego_Lane += -0.5
            self.pre_decision = -1.0
            self.can_lane_change = False
        else:
            pass

        # if self.ego_Lane % 1 == 0 and self.decision !=0:
        #     print("here")

        x_static= []
        lane_valid_distance = self.search_distance_valid()
        if lane_valid_distance is None:
            return None
        # else: ## for debug
        #     print("lane_valid_distance:", lane_valid_distance)

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
        distance1 = 0
        step = 6
        waypoint = starting_waypoint
        if self.distance_memory is not None:
            self.distance_memory -= self.uclidian_distance(waypoint.next(self.distance_memory)[0].trasnform.location,
                                                           goal_point)
            return self.distance_memory
        else:
            while self.uclidian_distance(waypoint.transform.location, goal_point) >= 2 * step:
                waypoint = waypoint.next(step)[0]
                distance1 += step
            return distance1 + self.uclidian_distance(waypoint.transform.location, goal_point)

    def get_waypoint_of_first_lane(self,actor):
        first_lane_waypoint = self.map.get_waypoint(actor.get_transform().location,
                                                   lane_type=carla.LaneType.Driving)  # self.controller.waypoint

        if first_lane_waypoint is not None:
            while first_lane_waypoint.lane_id > -1:  # get waypoint of forth's lane
                first_lane_waypoint = first_lane_waypoint.get_right_lane()
                if first_lane_waypoint is None:
                    return False

    def get_waypoint_of_last_lane(self,actor):
        last_lane_waypoint = self.map.get_waypoint(actor.get_transform().location, lane_type=carla.LaneType.Driving)  # self.controller.waypoint
        my_waypoint = last_lane_waypoint ## for debug

        if last_lane_waypoint is not None:
            while last_lane_waypoint.lane_id > -4:  # get waypoint of forth's lane
                #### for debug
                pre_last_lane_waypoint = last_lane_waypoint
                self.check = pre_last_lane_waypoint
                #### for debug
                last_lane_waypoint = last_lane_waypoint.get_right_lane()

                if last_lane_waypoint is None:
                    if abs(pre_last_lane_waypoint.lane_id) == -1 and abs(my_waypoint.lane_id) == -1:
                        return my_waypoint
                    else: ## for debug
                        print("pre_last_lane_waypoint1 : ", pre_last_lane_waypoint.lane_id, "my_waypoint_id1:",
                              my_waypoint.lane_id)  ## for debug
                        return False

            # print("lane id : ", last_lane_waypoint.lane_id)
        else:
            print("pre_last_lane_waypoint2 : ", self.check, "my_waypoint_id2:", my_waypoint.lane_id)

            return False
        # self.world.debug.draw_string(last_lane_waypoint.transform.location,
        #                                      'o', draw_shadow=True,
        #                                      color=carla.Color(r=255, g=255, b=0), life_time=9999)
        return last_lane_waypoint


    def search_distance_valid(self):
        # index= 4-self.ego_Lane
        # print("index :", index)

        last_lane_waypoint = self.get_waypoint_of_last_lane(self.player)
        if last_lane_waypoint == False:
            return None

        # print(self.ego_Lane)
        if self.max_Lane_num != self.pre_max_Lane_num:
            if self.index == len(self.lane_change_point) - 1:
                self.index = 0
            else:
                if self.index == 8 and self.distance_memory is None:  # it needs because of bug
                    pass
                else:
                    self.index += 1
                # if self.distance_memory is None or abs(self.distance_memory) <=20:

                self.pre_max_Lane_num = self.max_Lane_num
        # if self.index ==9: #bug not fixed completely. this is just a quick fix.
        #     self.index = 8

        # print(self.index)
        if self.index % 2 == 0:
            distance = self.get_distance_from_waypoint_to_goal_point(last_lane_waypoint,
                                                                     self.lane_change_point[self.index])
            return distance
        else:
            self.distance_memory = None
            distance = self.uclidian_distance(last_lane_waypoint.transform.location, self.lane_change_point[self.index])

            return -distance

            # if self.max_Lane_num ==4:
            #         while search_raidus <= distance:
            #             pre_distance = distance
            #             # distance = ((forth_lane_waypoint.next(i)[0].x-self.lane_start_point.x)**2+(forth_lane_waypoint.next(i)[0].y-self.lane_start_point.y)**2+(forth_lane_waypoint.next(i)[0].z-self.lane_start_point.z)**2)**0.5
            #             distance = ((forth_lane_waypoint.next(i)[0].transform.location.x-self.lane_start_point[self.section].x)**2+(forth_lane_waypoint.next(i)[0].transform.location.y-self.lane_start_point[self.section].y)**2+(forth_lane_waypoint.next(i)[0].transform.location.z-self.lane_start_point[self.section].z)**2)**0.5
            #
            #             # print("distance :",distance , "i :", i)
            #             if pre_distance <= distance:
            #                 print("pre_distance <= distance error")
            #                 break
            #             elif round(distance - search_raidus) > 0:
            #                 i += round(distance - search_raidus)
            #             else:
            #                 return min(distance + i, self.ROI_length)
            #             # print("i updated :", i)
            #         # self.max_Lane_num =3
            # elif self.max_Lane_num == 3:
            #     while search_raidus <= distance:
            #         distance = ((forth_lane_waypoint.next(i)[0].transform.location.x - self.lane_finished_point[
            #             self.section].x) ** 2 + (forth_lane_waypoint.next(i)[0].transform.location.y -
            #                                      self.lane_finished_point[self.section].y) ** 2 + (
            #                             forth_lane_waypoint.next(i)[0].transform.location.z -
            #                             self.lane_finished_point[self.section].z) ** 2) ** 0.5
            #         self.world.debug.draw_string(forth_lane_waypoint.next(i)[0].transform.location,
            #                                      'o', draw_shadow=True,
            #                                      color=carla.Color(r=255, g=255, b=0), life_time=9999)
            #         if pre_distance <= distance:
            #             print("pre_distance <= distance error")
            #             break
            #         elif round(distance - search_raidus) > 0:
            #             i += round(distance - search_raidus)
            #         else:
            #             return min(distance + i, self.ROI_length)
            #
            #         if round(distance - search_raidus) > 0:
            #             i += round(distance - search_raidus)
            #         else:
            #             return max(-(distance + i), -self.ROI_length)

    ## rule-based condition for lane change##
    # def can_lane_change(self,decision,state):
    #
    #     """
    #
    #     :param decision:
    #     :param state:  state_dyn = [dr/self.ROI_length, dv, length, self.extra_dl_list[x]]
    #     :return:
    #     """
    #     self.spawn_waypoint = self.map.get_waypoint(self.player.get_location(),lane_type=carla.LaneType.Driving)
    #     self.safe_lane_change_distance = int(self.controller.velocity/3.0+10.0)
    #     result = True
    #
    #     self.world.debug.draw_string(self.spawn_waypoint.next(self.safe_lane_change_distance)[0].transform.location, 'o',
    #                                  draw_shadow=True, color=carla.Color(r=255, g=0, b=0),
    #                                  life_time=-1)
    #
    #     if decision ==0:
    #         # print("tried strait in safety check")
    #         return True
    #     for i, state_dyn in enumerate(state):
    #
    #         if state_dyn[3] == decision:
    #             if state_dyn[0] >= 0.0 and state_dyn[0] <= self.safe_lane_change_distance:
    #                 print("not enough side front distance")
    #                 return False
    #             elif state_dyn[0] <= 0.0 and state_dyn[1] <= 0.0 and abs(state_dyn[0]) <= self.safe_lane_change_distance:
    #                 print("not enough behind distance")
    #                 return False
    #             elif state_dyn[0] <= 0 and abs(state_dyn[0]) <= self.safe_lane_change_distance/2.0:
    #                 print("not enough behind distance")
    #                 return False
    #         elif state_dyn[3] == 0:
    #
    #             if state_dyn[0] > 0.0 and state_dyn[0]  <= self.safe_lane_change_distance:
    #                 print("not engouth leading distance")
    #                 return False
    #     return True

    def loose_safety_check(self,decision,safe_lane_change_again_time = 3):
        if (time.time()-self.lane_change_time) <= safe_lane_change_again_time:
            self.can_lane_change = False
        else:
            self.can_lane_change = True

        if decision != 0:
            if self.can_lane_change == False:
                return 0
            else:
                self.can_lane_change = False
                self.lane_change_time = time.time()
                return decision
        else:
            return 0



    def safety_check(self,decision, safe_lane_change_again_time=3):
        action = decision ############### if this line deleted, not alert error properly

        if (time.time() - self.lane_change_time) <= safe_lane_change_again_time:
            self.can_lane_change = False
        else:
            self.can_lane_change = True

            # print("finished lane change", self.pre_decision)

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

            # if action !=0 and self.ego_Lane %1 !=0:
            #     self.check +=1
                # print("here")

            if action != 0:# and self.can_lane_change(action,state):
                self.lane_change_time = time.time()
                return action
            else:
                return 0
        else:
            return 0

    def get_max_lane(self,invalid_distances):
        """
        :param invalid_distances: int type list
        :return: thsi function change the variable, self.section and self.max_Lane_num
        condition : It returns a not properly computed value if a 'U' curve lane exists
                    in this case, it needs a vortual point
        """
        agent_cur_pos = self.player.get_transform().location
        if self.section < len(invalid_distances):

            self.lane_distance_between_start = self.uclidian_distance(agent_cur_pos,self.lane_start_point[self.section])

            self.lane_distance_between_end = self.uclidian_distance(agent_cur_pos,self.lane_finished_point[self.section])

            # print("self.lane_distance_between_start : ",self.lane_distance_between_start,"self.lane_distance_between_end :",self.lane_distance_between_end, "lane_distance_between_points[section]",lane_distance_between_points[self.section],"section :", self.section)
            if max(invalid_distances[self.section], self.lane_distance_between_start, self.lane_distance_between_end) == \
                    invalid_distances[self.section]:

                self.max_Lane_num = 3
                # world.debug.draw_string(self.player.get_transform().location, 'o', draw_shadow = True,
                #                                 color = carla.Color(r=255, g=255, b=0), life_time = 999)

            elif max(invalid_distances[self.section], self.lane_distance_between_start, self.lane_distance_between_end) == \
                    self.lane_distance_between_start and self.max_Lane_num == 3:

                self.section += 1
                self.max_Lane_num = 4
                # if virtual_point is None:
                if self.section > len(invalid_distances):
                    self.section = 0

                if self.section ==4:
                    self.max_Lane_num = 4


        # else:
            # distance = math.hypot(virtual_point.x-agent_cur_pos.x,virtual_point.y-agent_cur_pos.y)

    def visualize_virtual_invalid_forth_lane(self):
        for i in range(1,220,6):
            tmp = self.spawn_waypoint.next(2820)[0].get_right_lane().get_right_lane().next(i)
            if len(tmp)>1:
                waypoint = tmp[1]
            else:
                waypoint = tmp[0]
            self.world.debug.draw_string(waypoint.transform.location, 'o', draw_shadow=True, color=carla.Color(r=0, g=0, b=255),
                                    life_time=999)


        for i in range(1,300):
            tmp = self.spawn_waypoint.next(1180)[0].get_right_lane().get_right_lane().next(i)
            if len(tmp)>1:
                waypoint = tmp[1]
            else:
                waypoint = tmp[0]
            self.world.debug.draw_string(waypoint.transform.location, 'o', draw_shadow=True, color=carla.Color(r=0, g=0, b=255),
                                    life_time=999)


    def main_test(self):
        restart_time = time.time()
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) , carla.Location(x=382.441040, y=-212.488907, z=0.000000), carla.Location(x=161.251068, y=7.560803, z=8.935559),carla.Location(x=16.137751, y=143.156509, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=232.962860, y=-364.149139, z=0.000000) ,carla.Location(x=376.542816, y=-10.352980, z=0.000000), carla.Location(x=-136.986206, y=5.763320, z=7.994740), carla.Location(x=15.144917, y=-74.823540, z=0.000000) ]

        self.visualize_virtual_invalid_forth_lane()

        self.lane_change_point =[] #[carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) ,carla.Location(x=232.962860, y=-364.149139, z=0.000000), carla.Location(x=382.441040, y=-212.488907, z=0.000000),carla.Location(x=376.542816, y=-10.352980, z=0.000000)]
        for i in range(len(self.lane_start_point)):
            self.lane_change_point.append(self.lane_start_point[i])
            self.lane_change_point.append(self.lane_finished_point[i])

        for x in self.lane_change_point:
            self.world.debug.draw_string(x, 'o', draw_shadow=True, color=carla.Color(r=0, g=255, b=0), life_time=9999)

        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append(self.uclidian_distance(self.lane_start_point[i],self.lane_finished_point[i]))

            # ((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
            #         self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
            #          self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2) ** 0.5

        # virtual_point = carla.Location(x=-425.112549, y=405.182892, z=0.000000)
        # world.debug.draw_string(virtual_point, 'o', draw_shadow=True, color=carla.Color(r=0, g=0, b=255),
        #                         life_time=999)

        # for i in range(1,9000):
        #     tmp = self.spawn_waypoint.next(0.1)[0].get_right_lane().next(i)
            # if len(tmp)>1:
            #     waypoint = tmp[1]
            # else:
            #     waypoint = tmp[0]
            # waypoint = tmp[0]
            # self.world.debug.draw_string(waypoint.transform.location, 'o', draw_shadow=True, color=carla.Color(r=0, g=0, b=255),
            #                         life_time=999)
        d = None
        while True:
            # if time.time()-restart_time > 10:
            #     print("restart")
            #     self.restart()


            #     print(self.search_distance_valid())
            # print(self.controller.waypoint)


            if Keyboardcontrol.parse_events(client, self, clock):
                return

            self.spectator.set_transform(
                carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                                carla.Rotation(pitch=-90)))
            # self.camera_rgb.render(display)
            # self.hud.render(display)
            # pygame.display.flip()

            # extra_pos = self.extra_list[0].get_transform().location
            # print(self.is_extra_front_than_ego(extra_pos))

            ## Get max lane ##
            # print("start get lane")
            self.get_max_lane(lane_distance_between_points)
            d = self.search_distance_valid()

            print("d:", d, "section:", self.section, "index", self.index, "max_lane", self.max_Lane_num)

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

            # if time.time()-self.lane_change_time >10:
            #     self.lane_change_time = time.time()
                # if pre_decision is None:
                    # self.decision = -1
                    # self.decision = self.safety_check(self.decision)
            #
                    # self.ego_Lane-=1
                    # print("after decision: ", self.decision, "after lane", self.ego_Lane)

                    # pre_decision = -1
            #
            #
            #     elif pre_decision ==1:
            #         pre_decision = -1
            #         self.ego_Lane+=-1
            #         self.decision = -1
            #
            #     elif pre_decision ==-1:
            #         self.decision = 1
            #         self.ego_Lane+=1
            #         pre_decision =1
            #
            # else:
            #     self.decision = 0
            # self.decision= 0
            self.controller.apply_control(self.decision)
            self.decision=0

            # self.world.wait_for_tick(10.0)
            clock.tick(40)

            # self.hud.tick(self, clock)

    def main(self):

        PATH = "/home/a/carla_decision_making_with_RL-PER_DeepSet_DDQN/wegiths/"
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

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) , carla.Location(x=382.441040, y=-212.488907, z=0.000000),carla.Location(x=161.251068, y=7.560803, z=8.935559),carla.Location(x=16.137751, y=143.156509, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=232.962860, y=-364.149139, z=0.000000) ,carla.Location(x=376.542816, y=-10.352980, z=0.000000), carla.Location(x=-136.986206, y=5.763320, z=7.994740), carla.Location(x=15.144917, y=-74.823540, z=0.000000) ]
        self.visualize_virtual_invalid_forth_lane()
        self.lane_change_point = []

        for i in range(len(self.lane_start_point)):
            self.lane_change_point.append(self.lane_start_point[i])
            self.lane_change_point.append(self.lane_finished_point[i])
        for x in self.lane_change_point:
            self.world.debug.draw_string(x, 'o', draw_shadow=True, color=carla.Color(r=0, g=255, b=0), life_time=9999)

        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append(self.uclidian_distance(self.lane_start_point[i],self.lane_finished_point[i]))


        epoch_init = 0

        load_dir = PATH+'trained_info.pt'
        if(os.path.exists(load_dir)):

            print("저장된 가중치 불러옴")
            checkpoint = torch.load(load_dir)
            device = torch.device('cuda')
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.model.to(device)
            self.agent.target_model.load_state_dict((checkpoint['target_model_dict']))
            self.agent.target_model.to(device)
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.offline_learning_epoch = checkpoint['offline_learning_epoch']
            # self.agent.buffer.buffer = checkpoint['memorybuffer']
            epoch_init = checkpoint['epoch']
            self.agent.epsilon = checkpoint['epsilon']


        # self.agent.is_training = False
        print("epoch : ",epoch_init)
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

        n_iters = 9999000
        is_error = 0

        for epoch in range(epoch_init + 1, n_iters + 1):
            simulation_step = 0
            # print("start epoch!")
            self.start_epoch = True

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

                self.spectator.set_transform(
                    carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                                    carla.Rotation(pitch=-90)))

                ## Get max lane ##
                self.get_max_lane(lane_distance_between_points)
                # if self.pre_max_Lane_num!= self.max_Lane_num:
                #     print(self.max_Lane_num)

                ## finished get max lane ##
                ##visualize when, max lane ==3 ##
                # if self.max_Lane_num == 3:
                #     self.world.debug.draw_string(self.spawn_waypoint.transform.location,
                #                                  'o', draw_shadow=True,
                #                                  color=carla.Color(r=255, g=255, b=255), life_time=9999)
                # [self.extra_list, self.extra_dl_list] = self.agent.search_extra_in_ROI(self.extra_list,self.player,self.extra_dl_list)

                ## finished to visualize ##

                if self.agent.is_training:
                    ##dqn 과정##
                    # 가중치 초기화 (pytroch 내부)
                    # 입실론-그리디 행동 탐색 (act function)
                    # 메모리 버퍼에 MDP 튜플 얻기   ㅡ (step function)
                    # 메모리 버퍼에 MDP 튜플 저장   ㅡ
                    # optimal Q 추정             ㅡ   (learning function)
                    # Loss 계산                  ㅡ
                    # 가중치 업데이트              ㅡ

                    if epoch % 5 == 0:
                        # [w, b] = self.agent.model.parameters()  # unpack parameters
                        self.save_dir = torch.save({
                            'epoch': epoch,
                            'offline_learning_epoch': self.offline_learning_epoch,
                            'model_state_dict': self.agent.model.state_dict(),
                            'target_model_dict': self.agent.target_model.state_dict(),
                            'optimizer_state_dict': self.agent.optimizer.state_dict(),
                            # 'memorybuffer': self.agent.buffer.buffer,
                            'epsilon': self.agent.epsilon}, PATH+"trained_info"+".pt")#+str(epoch)+
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
                    else:
                        self.decision = self.agent.act(state, x_static)

                    before_safety_decision = self.decision
                    self.decision = self.safety_check(self.decision)
                    # print("before:", before_safety_decision)
                    # print("after:", self.decision)

                    is_error = self.controller.apply_control(self.decision)
                    if is_error:
                        print("controller error")
                        self.restart()
                        self.start_epoch = False
                    # print("extra_controller 개수 :", len(self.extra_controller_list))
                    # for i in range(len(self.extra_controller_list)):
                    #     self.extra_controller_list[i].apply_control()

                    # if self.decision == -1 and self.ego_Lane == 1:
                    #     print("decision :", tmp, "ego_lane : ", tmp2)

                    # print("after")

                    clock.tick(40)

                    tmp = self.step(self.decision)
#self.pre_ego_lane != self.ego_Lane
                    # if self.decision !=0:
                        # print("before decision:", self.decision, "before lane", self.pre_ego_lane)
                        # print("after decision: ", self.decision, "after lane", self.ego_Lane)

                    self.pre_ego_lane = self.ego_Lane


                    if tmp is None:
                        print("get_state_error in step process")
                        self.restart()
                        self.start_epoch = False
                        break

                    else:
                      [__, _, decision, reward, next_state, next_x_static, done] =  tmp

                    if done:
                        sample = [state, x_static, decision, reward, None, None, done]
                        x_static[0] = self.update_lane
                        # print("decision: ", decision, "ego_lane:", self.update_lane)

                        if epoch == epoch_init:
                            pass
                        else:
                            # if x_static[0]%1==0:
                            #     print("h")
                            # if decision !=0 and (next_x_static[0]%1 ==0 or x_static[0]%1 !=0):
                            #     print("h")
                            self.agent.buffer.append(sample)
                            # print("lane_valid_distance :",x_static[2]*self.ROI_length,"reward:",reward,"decision:",decision,"before decision",before_safety_decision)
                            self.agent.memorize_td_error(0)

                            if self.decision_changed:
                                # if x_static[0] <=1.5:+
                                #     print("lane 1 :", before_safety_decision)
                                # elif x_static[0]>3.5:
                                #     print("lane 4 :", before_safety_decision)

                                sample = [state, x_static, before_safety_decision, -3, None, None, done]

                                self.agent.buffer.append(sample)
                                self.agent.memorize_td_error(0)

                        print("buffer size : ", len(self.agent.buffer.size()))
                        n=300.0

                        print("epsilon :", self.agent.epsilon)
                        print("learning_rate:",self.agent.learning_rate)
                        print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)

                        # if epoch == 50:
                        #     self.agent.learning_rate = 0.0005
                        # elif epoch == 100:
                        #     self.agent.learning_rate/= 0.0001
                        # elif epoch == 500:
                        #     self.agent.learning_rate/= 0.00005

                        if epoch != epoch_init:
                            self.agent.update_td_error_memory(epoch)

                        #offline learning
                        if len(self.agent.buffer.size()) > self.agent.batch_size:
                            print("start learning")
                            self.agent.ddqn_learning()
                            for i in range(int(n)):
                                # self.offline_learning_epoch +=1
                                self.acummulated_loss += self.agent.loss
                            if epoch != epoch_init:
                                writer.add_scalar('Loss', self.acummulated_loss / n, epoch)
                        if epoch != epoch_init:
                            writer.add_scalar('누적 보상 ', self.accumulated_reward, epoch)

                        if epoch % 2 == 0:
                            self.agent.target_model.load_state_dict(self.agent.model.state_dict())

                        client.set_timeout(10)
                        self.restart()
                        self.start_epoch = False
                        # self.camera_rgb.toggle_camera()
                        # print("toggle_camera finished")

                    else:
                        sample = [state, x_static, self.decision, reward, next_state, next_x_static, done]
                        # sample = [state, x_static, pre_decision, reward, next_state, next_x_static, done]
                        x_static[0] = self.update_lane
                        # print("decision: ", decision, "ego_lane:", self.update_lane)
                        if epoch == epoch_init: #
                            pass
                        else:
                            # if x_static[0] % 1 == 0 and self.decision !=0:
                            #     print("h")
                            # if decision !=0 and (next_x_static[0]%1 ==0 or x_static[0]%1 !=0):
                            #     print("h")
                            self.agent.buffer.append(sample)
                            # print("lane_valid_distance :",x_static[2]*self.ROI_length,"reward:",reward,"decision:",decision,"before decision",before_safety_decision)

                            self.agent.memorize_td_error(0)
                            if self.decision_changed:
                                # if x_static[0] <=1.5:
                                #     print("lane 1 :", before_safety_decision)
                                # elif x_static[0]>3.5:
                                #     print("lane 4 :", before_safety_decision)
                                sample = [state, x_static, before_safety_decision, -3, None, None, done]
                                self.agent.buffer.append(sample)
                                self.agent.memorize_td_error(0)


                        #just generate space to put error

                        state = next_state
                        x_static = next_x_static


                else: #not traning


                    self.decision = self.agent.act(state, x_static)
                    # print(self.decision)
                    self.decision = self.safety_check(self.decision)

                    is_error = self.controller.apply_control(self.decision)
                    if is_error:
                        self.restart()
                        self.start_epoch = False

                    [__, _, decision, reward, next_state, next_static, done] = self.step(self.decision)

                    if done:
                        print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)
                        client.set_timeout(10)
                        self.restart()

        # finally:
        #     print('\ndestroying %d vehicles' % len(self.extra_list))
        #     # client.apply_batch([carla.command.DestroyActor(x) for x in self.extra_list])
        #
        #     print('destroying actors.')
        #     # client.apply_batch([carla.command.DestroyActors(x.id) for x in self.actor_list])
        #     # client.apply_batch([carla.command.DestroyActors(x.id) for x in self.extra_list])
        #     for actor in self.actor_list:
        #         # print("finally 에서 actor 제거 :", self.actor_list)
        #         if actor.is_alive:
        #             actor.destroy()
        #     for extra in self.extra_list:
        #         # print("finally 에서 actor 제거 :", self.extra_list)
        #         if extra.is_alive:
        #             extra.destroy()
        #
        #     # pygame.quit()
        #     print('done.')

if __name__ == '__main__':
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world =client.load_world('Town04_Opt')

        client.set_timeout(10.0)

        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=5.0)


        world.unload_map_layer(carla.MapLayer.All)

        world.set_weather(weather)

        CarlaEnv(world)


    except:
        pass




    # except KeyboardInterrupt:
        # print('\nCancelled by user. Bye!')

