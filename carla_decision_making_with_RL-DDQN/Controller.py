import carla
import math
import numpy as np
import time
import random
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class Pure_puresuit_controller:

    def __init__(self,player,waypoint=None,extra_actors=None,desired_vel=40):
        self.player = player
        self.player_length = math.hypot(self.player.get_physics_control().wheels[0].position.x-self.player.get_physics_control().wheels[2].position.x,
                                        self.player.get_physics_control().wheels[0].position.y-self.player.get_physics_control().wheels[2].position.y)/100.0 #unit : meters
        # self.player_length = ((self.player.get_physics_control().wheels[0].position.x - self.player.get_physics_control().wheels[
        #         2].position.x)**2+(self.player.get_physics_control().wheels[0].position.y - self.player.get_physics_control().wheels[
        #         2].position.y)**2)**0.5 / 100.0  # unit : meters
        # self.player_length = 2.9348175211493803
        self.world = self.player.get_world()
        self.map = self.world.get_map()
        if waypoint == None:
            self.waypoint = self.map.get_waypoint(self.player.get_location(),lane_type=carla.LaneType.Driving)
        else:
            self.waypoint = waypoint
        self.my_location_waypoint = waypoint
        # self.world.debug.draw_string(self.my_location_waypoint.transform.location, 'o', draw_shadow=True,
        #                              color=carla.Color(r=255, g=255, b=255), life_time=9999)
        self.pre_my_location_waypoint = self.my_location_waypoint
        self.desired_vel = desired_vel
        self.velocity = 0 #km/h

        self.error_v_pre = 0
        self.error_v = 0
        self.error_v_dot = 0
        self.error_v_int = 0

        self.error_a_pre = 0
        self.error_a = 0
        self.error_a_dot = 0
        self.error_a_int = 0

        self.leading_distance_pre = 0
        self.leading_distance = 0
        self.leading_distance_dot = 0
        self.leading_distance_int = 0

        # self.error_pre = 0
        # self.error = 0
        # self.error_dot = 0
        # self.error_int = 0
        self.k_v = [0.5, -0.004, 0.003] #60km/h : [0.03, 0.004, 0.003] #kp, kd, ki
        # self.k_acc = [0.2, -0.001, 0.001]
        self.cnt = 0
        self.t = time.time()
        # self.acc_start_time = 0
        self.pos_pre = self.player.get_location()
        self.pos = (self.player.get_physics_control().wheels[2].position+self.player.get_physics_control().wheels[3].position)/200.0
        # self.pos = carla.Vector3D(x=8.186243, y=-64.277382, z=0.359655)
        self.ld = 0
        self.heading = None
        self.extra_actors = extra_actors
        self.safe_distance = 20
        self.leading_vehicle = None
        self.search_radius = None
        self.left_search_radius = None
        self.right_search_radius = None
        # self.integral = 0
        self.throttle = 0
        self.y_ini = 0
        self.steer = 0
        self.h_constant = 1.3
        self.is_start_to_lane_change = False
        self.is_lane_changing = False
        self.is_fin_to_lane_change = False
        self.decision = None
        self.pre_decision = None
        self.check = 0
        self.vel_history = 0

        self.history_num = 0
        self.lane_change_history = 0

    def apply_control(self,decision):
        if decision !=0:
            self.lane_change_history += 1
        self.check = 0
        self.decision = decision
        dt = time.time() - self.t
        self.safe_distance = int(self.h_constant*self.velocity+20)

        self.is_fin_to_lane_change = False # when it doesnt get into the lane change waypoint still false

        self.history_num +=1
        self.vel_history = (self.vel_history * (self.history_num - 1) + self.velocity) / self.history_num

        if self.is_start_to_lane_change == True:
            self.is_lane_changing = True
            self.is_start_to_lane_change = False

        if decision !=0:
            self.is_start_to_lane_change = True

        ## waypoint update ##
        if self.waypoint is None:
            return False

        if self.ld < self.player_length+self.waypoint.lane_width:
            self.desired_vel = 70.0

            waypoints = self.waypoint.next(int(self.velocity / 3.6 * 0.3 + 3))
            if len(waypoints) ==0:
                return False
            else:
                self.waypoint = waypoints[0]

            if self.is_lane_changing == True:
                self.is_fin_to_lane_change = True
                self.is_lane_changing = False



                # self.world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
                #                          color=carla.Color(r=255, g=0, b=255), life_time=999)


            # self.world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
            #                          color=carla.Color(r=0, g=0, b=255), life_time=0.01)

        if self.decision == 1:
            print("right 차선 변경 수행")
            self.desired_vel = 50
            self.leading_vehicle = None

            waypoints = self.waypoint.next(int(self.velocity / 1.4 + 3))
            if len(waypoints) == 0:
                print("오른쪽 판단, waypoint 존재 x")
                return False
            else:
                self.waypoint = waypoints[0]

            self.waypoint = self.waypoint.get_right_lane()
            # self.world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
            #                              color=carla.Color(r=0, g=255, b=0), life_time=1)

        elif self.decision == -1:
            print("left 차선 변경 수행")
            self.desired_vel = 50
            self.leading_vehicle = None

            # self.player.set_autopilot(False)
            waypoints = self.waypoint.next(int(self.velocity / 1.4 + 3))
            if len(waypoints) == 0:
                print("왼쪽 판단, waypoint 존재 x")
                return False
            else:
                self.waypoint = waypoints[0]

            self.waypoint = self.waypoint.get_left_lane()

        if self.waypoint is None:
            return False

        # if self.is_lane_changing == True and self.is_start_to_lane_change == False:
        #     self.world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
        #                                  color=carla.Color(r=255, g=0, b=0), life_time=-1)

        if self.player.is_alive:
            self.pos =(self.player.get_physics_control().wheels[2].position+self.player.get_physics_control().wheels[3].position)/200.0#self.player.get_location()
            self.heading = self.player.get_transform().rotation.yaw #self.pos- self.pos_pre
            angle_waypoint = math.degrees(math.atan2(self.waypoint.transform.location.y-self.pos.y, self.waypoint.transform.location.x-self.pos.x))
            alpha = math.radians(angle_waypoint-self.heading)
            self.pos_pre = self.pos
        else:
            print("이미 죽은 actor")
            return -1
        # print(self.heading)

        ## lateral control ##
        self.ld = ((self.pos.x-self.waypoint.transform.location.x)**2+(self.pos.y-self.waypoint.transform.location.y)**2+(self.pos.z-self.waypoint.transform.location.z)**2)**0.5
        # eld = -(self.heading.y*self.waypoint.transform.location.x-self.heading.x*self.waypoint.transform.location.y+self.heading.x*self.pos.y-self.heading.y*self.pos.x)/(self.heading.x**2+self.heading.y**2+0.001)**0.5
        eld = abs(math.tan(self.heading)*self.waypoint.transform.location.x-self.waypoint.transform.location.y+self.waypoint.transform.location.y-math.tan(self.heading)*self.waypoint.transform.location.x)/(math.tan(self.heading)**2+1)**0.5
        self.velocity = (self.player.get_velocity().x**2+self.player.get_velocity().y**2+self.player.get_velocity().z**2)**0.5*3.6
        # km/h
        # self.steer = math.atan2(2 * self.player_length * eld, 3*self.velocity+0.01) * 180 / math.pi * 1 / 59  # 59 -> 1
        self.steer = math.atan2(2 * self.player_length * math.sin(alpha), self.ld) * 180.0 / math.pi * 1.0 / 59.0  # 59 -> 1
        # print("eld : ", eld, "ld : ", self.ld, "heading :", self.heading, "steer : ",self.steer)

        # self.safe_distance = self.h_constant * self.velocity+10
        ## 전방 차량 시각화 ##
        if self.leading_vehicle is not None:
            self.world.debug.draw_string(self.leading_vehicle.get_transform().location, 'o', draw_shadow=True,
                                         color=carla.Color(r=0, g=255, b=0), life_time=0.01)
        loop_break = False
        is_error = self.search_leading_vehicle()
        if is_error == False:
            return False

        if self.leading_vehicle is not None: #전방 차량이 없거나 없어졌을 때 다시 전방 차량을 찾아줌. 없으면 None값으로 초기화

            self.leading_distance = ((self.leading_vehicle.get_transform().location.x - self.player.get_transform().location.x) ** 2 + (
                                        self.leading_vehicle.get_transform().location.y - self.player.get_transform().location.y) ** 2 + (
                                        self.leading_vehicle.get_transform().location.z - self.player.get_transform().location.z) ** 2) ** 0.5\
                                        -self.safe_distance
            # print(self.leading_distance)
            # Finished when the leading vehicle is faster than desired_Vel or out of range
            if self.leading_distance > 0 or self.velocity > self.desired_vel:
                self.leading_vehicle = None

        if self.leading_vehicle is None:
            self.error_v_pre = self.error_v
            self.error_v = self.desired_vel - self.velocity
            self.error_v_dot = (self.error_v - self.error_v_pre) / dt
            if self.error_v < 0.1:
                self.cnt = 1
            if self.cnt != 1:
                self.error_v_int += self.error_v * dt
            else:
                pass

                # print("종방향 돔")
            self.error = [self.error_v, self.error_v_dot, self.error_v_int]
            # print("V error : " ,self.error[0])
            self.throttle = self.k_v[0] * self.error[0] + self.k_v[1] * self.error[1] + self.k_v[2] * self.error[2]

            self.control_input()
        else:
            ## ACC using CTG ##
            self.apply_ACC(dt)


            ## ACC using velocity ##
            # self.error_v_pre = self.error_v
            # leading_vel = (self.leading_vehicle.get_velocity().x**2+self.leading_vehicle.get_velocity().y**2+self.leading_vehicle.get_velocity().z**2)**0.5
            # self.error_v = leading_vel - self.velocity
            # self.error_v_dot = (self.error_v - self.error_v_pre) / dt
            # if self.error_v < 0.1:
            #     self.cnt = 1
            # if self.cnt != 1:
            #     self.error_v_int += self.error_v * dt
            # else:
            #     pass
            #     # print("종방향 돔")
            # self.error = [self.error_v, self.error_v_dot, self.error_v_int]
            # self.a = self.k_acc[0] * self.error[0] + self.k_acc[1] * self.error[1] + self.k_acc[2] * self.error[2]
            # self.control_input()



        # print(self.waypoint.section_id)
        # print(self.waypoint.lane_width)
        self.pre_my_location_waypoint = self.my_location_waypoint
        self.t = time.time()

        # if self.is_lane_changing:
        #     self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
        #                                  color=carla.Color(r=0, g=255, b=255), life_time=1)
        # elif self.is_start_to_lane_change:
        #     self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
        #                                  color=carla.Color(r=255, g=0, b=0), life_time=1)
        # elif self.is_fin_to_lane_change:
        #     self.world.debug.draw_string(self.player.get_location(), 'o', draw_shadow=True,
        #                                  color=carla.Color(r=0, g=0, b=255), life_time=1)
    def is_side_safe(self,decision):
        self.leading_vehicle()
        loop_break = False
        self.my_location_waypoint = self.map.get_waypoint(self.player.get_location(), lane_type=carla.LaneType.Driving)
        if decision == 1:
            self.my_location_waypoint = self.my_location_waypoint.get_right_lane()
        elif decision == -1:
            self.my_location_waypoint = self.my_location_waypoint.get_left_lane()

        # self.world.debug.draw_string(self.my_location_waypoint.transform.location, 'o', draw_shadow=True,
        #                         color=carla.Color(r=0, g=255, b=255), life_time=999)

        # tmp = self.waypoint.previous(10)[0].transform.location

        for actor in self.extra_actors:
            extra_pos = actor.get_transform().location
            next_waypoints = None
            for x in range(1, self.safe_distance + 1 - int(self.waypoint.lane_width), 1):

                # if x== self.safe_distance - int(self.waypoint.lane_width):
                #     print("h")
                if self.my_location_waypoint is None:
                    print(
                        self.pre_my_location_waypoint)  # Waypoint(Transform(Location(x=4.583873, y=-90.357193, z=0.000000), Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000)))
                    print(x)  # 1
                else:
                    next_waypoints = self.my_location_waypoint.next(x)
                    self.pre_my_location_waypoint = self.my_location_waypoint

                if next_waypoints is None or len(next_waypoints) == 0:
                    print("next waypoints is none")
                    return False
                else:
                    self.search_radius = ((extra_pos.x - next_waypoints[0].transform.location.x) ** 2 + (
                            extra_pos.y - next_waypoints[0].transform.location.y) ** 2) ** 0.5

                if self.search_radius <= self.waypoint.lane_width / 2:
                    # print("추종 시작")
                    self.leading_vehicle = actor
                    # self.acc_start_time = time.time()
                    loop_break = True
                    break
            if loop_break == True:
                loop_break = False
                break
    def search_leading_vehicle(self):
        loop_break = False
        self.my_location_waypoint = self.map.get_waypoint(self.player.get_location(),lane_type=carla.LaneType.Driving)
        if abs(self.waypoint.lane_id) > abs(self.my_location_waypoint.lane_id):
            self.my_location_waypoint = self.my_location_waypoint.get_right_lane()
        elif abs(self.waypoint.lane_id) < abs(self.my_location_waypoint.lane_id):
            self.my_location_waypoint = self.my_location_waypoint.get_left_lane()

        # self.world.debug.draw_string(self.my_location_waypoint.transform.location, 'o', draw_shadow=True,
        #                         color=carla.Color(r=0, g=255, b=255), life_time=999)

        # tmp = self.waypoint.previous(10)[0].transform.location

        for actor in self.extra_actors:
            if actor.is_alive:
                extra_pos = actor.get_transform().location
            else:
                continue
            next_waypoints = None
            for x in range(1, self.safe_distance + 1 - int(self.waypoint.lane_width), 1):

                # if x== self.safe_distance - int(self.waypoint.lane_width):
                #     print("h")
                if self.my_location_waypoint is None:
                    print(
                        self.pre_my_location_waypoint)  # Waypoint(Transform(Location(x=4.583873, y=-90.357193, z=0.000000), Rotation(pitch=0.000000, yaw=-450.224854, roll=0.000000)))
                    print(x)  # 1
                else:
                    next_waypoints = self.my_location_waypoint.next(x)
                    self.pre_my_location_waypoint = self.my_location_waypoint

                if next_waypoints is None or len(next_waypoints)==0:
                    print("next waypoints is none")
                    return False
                else:
                    self.search_radius = ((extra_pos.x - next_waypoints[0].transform.location.x) ** 2 + (
                            extra_pos.y - next_waypoints[0].transform.location.y) ** 2) ** 0.5

                if self.search_radius <= self.waypoint.lane_width / 2:
                    # print("추종 시작")
                    self.leading_vehicle = actor
                    # self.acc_start_time = time.time()
                    loop_break = True
                    break
            if loop_break == True:
                loop_break = False
                break



    def control_input(self):
        if self.throttle>=0:
            self.throttle =min(self.throttle,1)
            self.player.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=0.0))
            # print("self.a:", self.a)
            # print("self.accel:", (self.player.get_acceleration().x**2+self.player.get_acceleration().y**2+self.player.get_acceleration().z**2)**0.5)
            # print("self.vel:",(self.player.get_velocity().x**2+self.player.get_velocity().y**2+self.player.get_velocity().z**2)**0.5)


        else:
            b = -self.throttle
            b = min(b, 1)
            self.player.apply_control(carla.VehicleControl(throttle=0, steer=self.steer, brake=b))

    def apply_ACC(self,dt):


        # self.h_constant = 1.8
        lamda = 0.4
        # tau = 0.5
        leading_vehicle_length = self.leading_vehicle.bounding_box.extent.x # /2 = leading vehicle length
        # L_des = leading_vehicle_length*2+h*self.velocity
        epsilon = -(((self.leading_vehicle.get_transform().location.x - self.player.get_transform().location.x) ** 2 + (
                            self.leading_vehicle.get_transform().location.y - self.player.get_transform().location.y) ** 2 + (
                            self.leading_vehicle.get_transform().location.z - self.player.get_transform().location.z) ** 2) ** 0.5 \
                  -self.player_length/2-leading_vehicle_length)

        # print(epsilon)

        spacing_error = epsilon + self.h_constant*self.velocity+10
        # print(spacing_error)
        x_des_ddot = -1/self.h_constant * (epsilon + lamda * spacing_error)
        # print(x_des_ddot)
        accel = self.player.get_acceleration()
        accel_x = accel.x
        accel_y = accel.y
        accel_z = accel.z
        Max = max(abs(accel_x),abs(accel_y))
        sign = 0
        if accel_x ==Max or accel_y ==Max:
            sign = 1
        else:
            sign = -1

        current_accel =sign* (accel_x**2+accel_y**2+ accel_z**2)**0.5
        self.error_a = x_des_ddot - current_accel
        self.error_a_dot = (self.error_a - self.error_a_pre)/dt
        # self.error_a_int += self.error_a * dt
        self.error_a_pre = self.error_a

        # [0.5, -0.004, 0.003]
        # p=0.3
        # i=-0.004
        # d=0.003
        # self.throttle = p* self.error_a + d*self.error_a_dot
        # print(error)

        self.throttle = x_des_ddot

        # t= time.time()- self.acc_start_time
        # self.integral += math.exp(tau*t)*x_des_ddot*dt
        # print(self.integral)
        # self.a = (self.integral + self.y_ini)*math.exp(-tau*t)
        self.control_input()