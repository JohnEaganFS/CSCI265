# Imports
import pygame
import pymunk
import numpy as np

def initialize_pygame(width, height):
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Cooperative Racing Environment")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return screen, clock

def draw_walls(screen, pl_set):
    for pl in pl_set:
        # Draw filled polygon (each pl is a list of points)
        pygame.draw.polygon(screen, (255, 0, 0), pl)

def draw_waypoint_segments(screen, points, current_waypoint, next_waypoint):
    for i in range(len(points) - 1):
        if i == current_waypoint:
            color = (255, 0, 255)
        else:
            color = (255, 255, 255)
        pygame.draw.line(screen, color, points[i], points[i + 1], 4)

def draw_test_waypoints(screen, points):
    for i, (p1, p2, p3, p4) in enumerate(points):
        pygame.draw.polygon(screen, (128,128,128), (p1, p2, p3, p4))

def draw_waypoints(screen, waypoints, current_waypoint, next_waypoint):
    for i, waypoint in enumerate(waypoints):
        # Make the current waypoint red
        if i == current_waypoint:
            color = (0, 0, 255)
        # Make the next waypoint green
        elif i == next_waypoint:
            color = (0, 255, 0)
        # Make completed waypoints blue
        elif i < current_waypoint:
            color = (0, 0, 255)
        # Make future waypoints white
        else:
            color = (255, 255, 255)
        pygame.draw.circle(screen, color, (int(waypoint[0]), int(waypoint[1])), 4)

def getNewHeading(heading_angle, steering_angle):
    '''
    Returns the new heading of the agent based on the current heading angle and steering angle.
    In other words, just adds the steering angle to the heading angle.
    '''
    # Add some noise to the steering angle
    noise = np.random.normal(0, 0.05)
    steering_angle += noise
    # Resize steering angle between -pi/128 and pi/128
    steering_angle = steering_angle * np.pi / 128
    new_angle = heading_angle - steering_angle
    # Keep the angle between -pi and pi
    if (new_angle > np.pi):
        new_angle -= 2 * np.pi
    elif (new_angle < -np.pi):
        new_angle += 2 * np.pi
    return new_angle

def getNewSpeed(speed, throttle, speed_limit):
    force = throttle

    force += np.random.normal(0, 0.05)

    new_speed = speed + force

    if new_speed > speed_limit:
        new_speed = speed_limit
    elif new_speed < 30:
        new_speed = 30
    
    return new_speed

def create_car(pos):
    car = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
    car.position = (pos[0], pos[1])
    car_shape = pymunk.Circle(car, 2)
    car_shape.elasticity = 1
    car_shape.collision_type = 1
    return car, car_shape