'''
Switching to a 2D physics engine library for the sake of simplicity. I will be using PyMunk. This will enable be me to
create the "walls" of the track and the "car" that will be racing around the track.
'''

import gym

import numpy as np
import matplotlib.pyplot as plt

import pygame
import pymunk

from read_gpx import read_gpx, removeDuplicatePoints, scaleData

### Miscellaneous Functions ###
def boundaries(x, y, scale):
    '''
    Returns the boundaries of the environment as a list of 4 points
    '''
    v = y - x
    
    p1 = np.array([v[1], -v[0]])
    p2 = np.array([-v[1], v[0]])

    # Scale to length of v
    p1 = p1 / np.linalg.norm(p1) * scale * 0.5
    p2 = p2 / np.linalg.norm(p2) * scale * 0.5

    p3 = p1 + v
    p4 = p2 + v

    points = np.array([p1, p2, p3, p4])
    points += x
    return points

def create_wall(space, x1, y1, x2, y2):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)
    segment = pymunk.Segment(body, (x1, y1), (x2, y2), 1)
    space.add(body, segment)
    return segment

def draw_walls(screen, walls):
    for wall in walls:
        pygame.draw.line(screen, (255, 255, 255), wall.a, wall.b, 1)

def game():
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update physics
        space.step(1 / FPS)

        # Draw stuff
        screen.fill((0, 0, 0))
        draw_walls(screen, walls)
        screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
        pygame.display.update()
        clock.tick(FPS)
    
def define_boundaries(points, scale):
    # Initialize boundary_points
    boundary_points = []

    # Create boundaries
    for i in range(len(points) - 1):
        A = np.array([points[i][0], points[i][1]])
        B = np.array([points[i + 1][0], points[i + 1][1]])
        a, b, c, d = boundaries(A, B, scale)
        # For the first point, define itself and the next point
        if i == 0:
            boundary_points.append((a, b))
        # For all other points, only define the next point (the current point has already been defined by the previous iteration)
        boundary_points.append((c, d))
    
    return boundary_points

if __name__ == "__main__":
    # Create pymunk space
    space = pymunk.Space()

    # Pull waypoints from .gpx file
    points = removeDuplicatePoints(read_gpx('../gpx/windy_road.gpx', 1))
    points = points[10:100]
    points = scaleData(points)
    points = points * 900 + 50

    min_vector_length = min([np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)])
    
    # Create boundaries
    boundary_points = define_boundaries(points, min_vector_length)
    # print(boundary_points)

    # Plot boundaries
    # plt.scatter(points[:,0], points[:,1], c='green', s=5)
    # for i, bp in enumerate(boundary_points):
    #     plt.plot([bp[0][0], bp[1][0]], [bp[0][1], bp[1][1]], c='black')
    #     # Label the points
    #     plt.text(bp[0][0], bp[0][1], str(i))
    # plt.show()

    # Create simple line segments
    walls = []
    for i in range(len(boundary_points) - 1):
        # Create segment from point i to point i + 1
        p1 = boundary_points[i][0]
        p2 = boundary_points[i + 1][0]
        p3 = boundary_points[i][1]
        p4 = boundary_points[i + 1][1]
        walls.append(create_wall(space, p1[0], p1[1], p2[0], p2[1]))
        walls.append(create_wall(space, p3[0], p3[1], p4[0], p4[1]))
        # Create segment between p1 and p3
        walls.append(create_wall(space, p1[0], p1[1], p3[0], p3[1]))

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    clock = pygame.time.Clock()
    FPS = 60

    # Run game
    game()