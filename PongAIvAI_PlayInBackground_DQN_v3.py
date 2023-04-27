#   PongAIvAI
#   Omar & Will - USE THIS ONE AS OF 4/26/23
#   Authors: Michael Guerzhoy and Denis Begun, 2014-2016.
#   http://www.cs.toronto.edu/~guerzhoy/
#   Email: guerzhoy at cs.toronto.edu
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version. You must credit the authors
#   for the original parts of this code.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   Parts of the code are based on T. S. Hayden Dennison's PongClone (2011)
#   http://www.pygame.org/project-PongClone-1740-3032.html

#   This code runs with Python 2 and requires PyGame for Python 2
#   Download PyGame here: https://bitbucket.org/pygame/pygame/downloads


# import pygame
import sys, time, random, os
from pygame.locals import *

import math
import numpy as np
import argparse

white = [255, 255, 255]
black = [0, 0, 0]
# clock = pygame.time.Clock()

class fRect:
    """Like PyGame's Rect class, but with floating point coordinates"""

    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1 #self.size > 0 and other_frect.size > 0

class Paddle:
    def __init__(self, pos, size, speed, max_angle,  facing, timeout):
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.facing = facing
        self.max_angle = max_angle
        self.timeout = timeout

    def factor_accelerate(self, factor):
        self.speed = factor*self.speed

    def move(self, direction, table_size):
    # def move(self, enemy_frect, ball_frect, table_size):
        
        # The program crashes if move_getter crashes. The runtime of 
        # move_getter is not limited
        # direction = self.move_getter(self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size))
        
        # The program continues if move_getter crashes. The runtime of
        # move_getter is limited
        # direction = timeout(self.move_getter, (self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size)), {}, self.timeout)
        
        if direction == "up":
            self.frect.move_ip(0, -self.speed)
        elif direction == "down":
            self.frect.move_ip(0, self.speed)

        to_bottom = (self.frect.pos[1]+self.frect.size[1])-table_size[1]

        if to_bottom > 0:
            self.frect.move_ip(0, -to_bottom)
        to_top = self.frect.pos[1]
        if to_top < 0:
            self.frect.move_ip(0, -to_top)


    def get_face_pts(self):
        return ((self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1]),
                (self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1] + self.frect.size[1]-1)
                )

    def get_angle(self, y):
        center = self.frect.pos[1]+self.size[1]/2
        rel_dist_from_c = ((y-center)/self.size[1])
        rel_dist_from_c = min(0.5, rel_dist_from_c)
        rel_dist_from_c = max(-0.5, rel_dist_from_c)
        sign = 1-2*self.facing

        return sign*rel_dist_from_c*self.max_angle*math.pi/180

class Ball:
    def __init__(self, table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag):
        rand_ang = (.4+.4*random.random())*math.pi*(1-2*(random.random()>.5))+.5*math.pi # half chance positive or negative 
        speed = (init_speed_mag*math.cos(rand_ang), init_speed_mag*math.sin(rand_ang)) # random init speed at two directions 
        pos = (table_size[0]/2, table_size[1]/2)
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.prev_bounce = None

    def get_center(self):
        return (self.frect.pos[0] + .5*self.frect.size[0], self.frect.pos[1] + .5*self.frect.size[1])

    def get_speed_mag(self):
        return math.sqrt(self.speed[0]**2+self.speed[1]**2)

    def factor_accelerate(self, factor):
        self.speed = (factor*self.speed[0], factor*self.speed[1])

    def move(self, paddles, table_size, move_factor):
        moved = 0
        walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]

        for wall_rect in walls_Rects:
            if self.frect.get_rect().colliderect(wall_rect):
                c = 0
                
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                r1 = 1+2*(random.random()-.5)*self.dust_error
                r2 = 1+2*(random.random()-.5)*self.dust_error

                self.speed = (self.wall_bounce*self.speed[0]*r1, -self.wall_bounce*self.speed[1]*r2)
                
                while c > 0 or self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1              

        for paddle in paddles:
            if self.frect.intersect(paddle.frect):
                if (paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[0]/2) or \
                (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0]/2):
                    continue
                
                c = 0
                
                while self.frect.intersect(paddle.frect) and not self.frect.get_rect().colliderect(walls_Rects[0]) and not self.frect.get_rect().colliderect(walls_Rects[1]):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    
                    c += 1
                theta = paddle.get_angle(self.frect.pos[1]+.5*self.frect.size[1])
                
                v = self.speed

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]

                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to the  increase by 1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    self.speed = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    self.speed = (v[0], v[1])
                self.prev_bounce = paddle               

                while c > 0 or self.frect.intersect(paddle.frect):
                
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    
                    c -= 1
                
                moved = 1
                
        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        print("TIMEOUT")
        return default
    else:
        return it.result

def check_point(score, ball, table_size):
    if ball.frect.pos[0]+ball.size[0]/2 < 0:
        score[1] += 1
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        return (ball, score) # new ball, new start 
    elif ball.frect.pos[0]+ball.size[0]/2 >= table_size[0]:
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        score[0] += 1
        return (ball, score)

    return (ball, score)


"""
Add global variables here if needed. 
"""
class Pong:
    def __init__(self):
        """
        scores to win one episode. 
        """
        self.score_to_win = 10
        """
        DQN agent is always on the left side. 
        Replace chaser_ai by other ai as the enemy on the right side. 
        """
        self.right_ai = __import__("chaser_ai") 
        # self.right_ai = __import__("always_up_ai")

        self.prev_ball_pos = [0, 0]

    def reset(self):
        self.table_size = (440, 280)
        paddle_size = (10, 70)
        ball_size = (15, 15)
        paddle_speed = 1
        max_angle = 45

        paddle_bounce = 1.2
        wall_bounce = 1.00
        dust_error = 0.00
        init_speed_mag = 2
        timeout = 0.0003

        self.score = [0,0]

        # [left paddle, right paddle]
        # 20 units away from the wall.
        self.paddles = [Paddle((20, self.table_size[1]/2), paddle_size, paddle_speed, max_angle,  1, timeout),
               Paddle((self.table_size[0]-20, self.table_size[1]/2), paddle_size, paddle_speed, max_angle, 0, timeout)]
        self.ball = Ball(self.table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)

        self.paddles[1].move_getter = self.right_ai.pong_ai

        """
        This is the state in reset(). You can change it to pass along different data to the network. 
        """

        """
        My additions
        """

        # Our (Left) Paddle Vertical Midppint Position
        self.my_paddle_vert_pos = self.paddles[0].frect.pos[1] + self.paddles[0].frect.size[1]/2

        self.enemy_paddle_vert_pos = self.paddles[1].frect.pos[1] + self.paddles[1].frect.size[1] / 2

        self.ball_middle_pos = [self.ball.frect.pos[0]+self.ball.frect.size[0]/2, self.ball.frect.pos[1]+self.ball.frect.size[1]/2]

        self.prev_ball_pos = [0, 0]

        state = (self.my_paddle_vert_pos,
                 self.enemy_paddle_vert_pos,
                 self.ball_middle_pos[0], self.ball_middle_pos[1],
                 self.prev_ball_pos[0], self.prev_ball_pos[1],
                 self.table_size[0], self.table_size[1])

        return state, None   

    def step(self, action):   
        if action == 0:
            action = "up"
        elif action == 1:
            action = "down"

        self.paddles[0].move(action, self.table_size)
        action_enemy = select_action_enemy((self.paddles, self.ball, self.table_size))
        self.paddles[1].move(action_enemy, self.table_size)
        
        inv_move_factor = int((self.ball.speed[0]**2+self.ball.speed[1]**2)**.5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self.ball.move(self.paddles, self.table_size, 1./inv_move_factor)
        else:
            self.ball.move(self.paddles, self.table_size, 1)    
        
        old_score = self.score.copy()
        self.ball, self.score = check_point(self.score, self.ball, self.table_size)

        terminated = False
        if max(self.score) >= self.score_to_win: 
            terminated = True


        """
        Change the reward based on the current state. The current reward is -1 for every step. 

        variable old_score is score in previous step. 
        variable score is score in current step. 
        score is a list [left_score, right_score] and your DQN agent is always on the left side. 
        For example, if old_score and score are same, it means the current round of game continues. 
        If old_score is [3,4] and current score is [4,4], it means your agent wins this round. 
        The total rounds of game is 10. 

        You may use the these information as well to design your reward strategy. 
        You may re-visit the instructions in the step 1 of your project to understand these parameters. 
        - your paddle frect: self.paddles[0].frect
        - the enemy's paddle frect: self.paddles[1].frect
        - ball's frect: self.ball.frect
        - table size: self.table_size
        """
        if self.score[0] > old_score[0]:
            reward = 10000
        elif self.score[1] > old_score[1]:
            reward = -10000
        else:
            reward = -abs(self.my_paddle_vert_pos - self.ball_middle_pos[1]) + 100

        """
        This is the state in step(). You can change it to pass along different data to the network. 

        For example, you can pass the middle vertical position of the paddle instead of four parameters 
        of the paddle. You also can pass the previous positions of ball. 

        Please make sure that you need to change the definition of state in reset() function as well 
        if you change it here.
        """
        self.my_paddle_vert_pos = self.paddles[0].frect.pos[1] + self.paddles[0].frect.size[1] / 2

        self.enemy_paddle_vert_pos = self.paddles[1].frect.pos[1] + self.paddles[1].frect.size[1] / 2

        self.ball_middle_pos = [self.ball.frect.pos[0] + self.ball.frect.size[0] / 2,
                                self.ball.frect.pos[1] + self.ball.frect.size[1] / 2]

        state = (self.my_paddle_vert_pos,
                 self.enemy_paddle_vert_pos,
                 self.ball_middle_pos[0], self.ball_middle_pos[1],
                 self.prev_ball_pos[0], self.prev_ball_pos[1],
                 self.table_size[0], self.table_size[1])

        # Set prev ball pos after we set the state
        self.prev_ball_pos = [self.ball.frect.pos[0] + self.ball.frect.size[0] / 2,
                                self.ball.frect.pos[1] + self.ball.frect.size[1] / 2]

        truncated = False
        info = self.score.copy()
        return state, reward, terminated, truncated, info

def select_action_enemy(state_enemy):
    paddles, ball, table_size = state_enemy 

    direction = timeout(paddles[1].move_getter, (paddles[1].frect.copy(), paddles[0].frect.copy(), ball.frect.copy(), tuple(table_size)), {}, paddles[1].timeout)

    return direction

# for testing your code
if __name__ == '__main__':
    env = Pong()  
    obs, info = env.reset()
    while True:
        action = np.random.randint(2)
        obs, reward, terminated, truncated, score = env.step(action)
        
        if terminated or truncated:
            break