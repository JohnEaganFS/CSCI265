### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Simulation
import pygame
import pymunk
from pymunk.autogeometry import *
from pymunk.vec2d import Vec2d

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData
from CustomRacing2D_env import define_boundaries

### Global Variables ###
# Pygame parameters
FPS = 120
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# Waypoint stuff
start, amount = 30, 20
add_points_iterations = 5
total_waypoints = 50
track_width = 60

### Functions ###
def add_points(points):
    # Add points in between each set of points
    new_points = []
    for i in range(len(points) - 1):
        new_points.append(points[i])
        new_points.append((points[i] + points[i + 1]) / 2)
    new_points.append(points[-1])
    return np.array(new_points)

def select_points(points, total_points):
    # Select a subset of evenly spaced points from the points
    new_points = []
    for i in range(total_points):
        new_points.append(points[int(i * len(points) / total_points)])
    return np.array(new_points)

def sample_func(point):
    x, y = int(point[0]), int(point[1])
    return pixel_values[x, y]

# Pygame functions
def pygame_init():
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Map Creation")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock

def render(screen, clock):
    # Update the screen
    pygame.display.flip()
    clock.tick(FPS)

def draw_waypoints(screen, waypoints):
    for i in range(len(waypoints) - 1):
        # Draw points
        pygame.draw.circle(screen, (0, 255, 0), waypoints[i], 2)

def draw_boundaries(screen, boundary_points, size=2, fill_inside=False, color=(255, 255, 255), draw_sides=False):
    # Draw lines
    for i, bp in enumerate(boundary_points):
        # If first or last point, draw a line between bp[0] and bp[1]
        if i == 0:
            pygame.draw.line(screen, (255, 255, 255), bp[0], bp[1], size)
            pygame.draw.line(screen, (255, 255, 255), bp[0], boundary_points[i + 1][0], size)
            pygame.draw.line(screen, (255, 255, 255), bp[1], boundary_points[i + 1][1], size)
        elif i == len(boundary_points) - 1:
            pygame.draw.line(screen, (255, 255, 255), bp[0], bp[1], size)
        else:
            pygame.draw.line(screen, (255, 255, 255), bp[0], boundary_points[i + 1][0], size)
            pygame.draw.line(screen, (255, 255, 255), bp[1], boundary_points[i + 1][1], size)

    if fill_inside:
        # Make everything inside the boundaries green (unless it's a different color already)
        for i in range(len(boundary_points) - 1):
            # Quad
            pygame.draw.polygon(screen, color, (boundary_points[i][0], boundary_points[i][1], boundary_points[i + 1][1], boundary_points[i + 1][0]))
        
    if draw_sides:
        for bp in boundary_points:
            pygame.draw.line(screen, (0, 0, 0), bp[0], bp[1], 3)

### Classes ###
class Map:
    pass

if __name__ == "__main__":
    filename = "../gpx/windy_road.gpx"
    
    # Read the .gpx file
    gpx = read_gpx(filename, 1)

    # Remove duplicate points
    gpx = removeDuplicatePoints(gpx)

    # Select a subset of the points depending on the start and amount
    gpx_subset = gpx[start:start + amount]

    # Scale the lat/lon and ele coordinates to be in the range [0, 1]
    gpx_scaled = scaleData(gpx_subset)

    # Convert the scaled lat/lon and ele coordinates to pixel coordinates according to the screen size
    gpx_scaled = gpx_scaled * (SCREEN_WIDTH - 100) + 50

    # Remove the ele coordinates
    gpx_scaled = gpx_scaled[:, :2]

    ### Adding points process ###
    # For some number of iterations,
    for i in range(add_points_iterations):
        # Add more points in between each set of points
        gpx_scaled = add_points(gpx_scaled)
    
    # Select a subset of evenly spaced points from the gpx_scaled points to be the waypoints depending on the total_waypoints
    waypoints = select_points(gpx_scaled, total_waypoints)

    # Define the boundaries of the track
    boundary_points = define_boundaries(waypoints, track_width)

    # Add a point to beginning and end a little bit away from the beginning and end
    v_begin = waypoints[1] - waypoints[0]
    v_end = waypoints[-2] - waypoints[-1]
    v_begin = v_begin / np.linalg.norm(v_begin)
    v_end = v_end / np.linalg.norm(v_end)
    waypoints_with_added_points = np.copy(waypoints)
    waypoints_with_added_points = np.insert(waypoints_with_added_points, 0, waypoints[0] - v_begin * 20, axis=0)
    waypoints_with_added_points = np.insert(waypoints_with_added_points, len(waypoints_with_added_points), waypoints[-1] - v_end * 20, axis=0)

    # Define boundaries with bigger track width
    boundary_points_big = define_boundaries(waypoints_with_added_points, track_width + 50)
    boundary_points_bigger = define_boundaries(waypoints_with_added_points, track_width + 100)

    # Initialize pygame
    screen, clock = pygame_init()

    # Draw 
    # draw_waypoints(screen, waypoints)
    screen.fill((0, 0, 0))
    # draw_boundaries(screen, boundary_points_bigger, 5, fill_inside=True, color = (0,0,255))
    draw_boundaries(screen, boundary_points_big, 2, fill_inside=True, draw_sides=True)
    draw_boundaries(screen, boundary_points, 5, fill_inside=True, color = (0,0,0))
    # while True:
    #     # Check for quit event
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             exit()
    #     render(screen, clock)
    render(screen, clock)

    # Save the rendered image as a list of pixel values (0, 1)
    pixel_values = pygame.surfarray.array3d(screen)
    pixel_values = pixel_values[:, :, 0] / 255

    # Use autogeometry to create a polygon from the boundary points
    pl_set = march_soft(pymunk.BB(0, 0, SCREEN_WIDTH-1, SCREEN_HEIGHT-1), 500, 500, sample_func=sample_func, threshold=0.0001)

    pl_copy = []
    for pl in pl_set:
        pl_copy.append(pl.copy())

    # Create a pymunk space
    space = pymunk.Space()
    for pl in pl_set:
        # Create static polygons
        poly = pymunk.Poly(space.static_body, pl)
        poly.elasticity = 0.2
        space.add(poly)
    
    # Create a copy with just the static polygons (raw map)
    space_static = space.copy()

        # for i in range(len(pl) - 1):
        #     space.add(pymunk.Segment(space.static_body, pl[i], pl[i + 1], 1))
    
    # Create a ball to check collisions
    ball = pymunk.Body(1, 1, body_type=pymunk.Body.DYNAMIC)
    ball_shape = pymunk.Circle(ball, 5)
    ball_shape.color = (0, 0, 255)
    ball_shape.elasticity = 1
    ball.position = (waypoints[1][0], waypoints[1][1])
    ball.velocity = (200, 200)
    length = np.linalg.norm(ball.velocity)

    # Add the ball to the space
    space.add(ball, ball_shape)

    # Collision handler
    def begin(arbiter, space, data):
        print("Collision!")
        return True
    
    def separate(arbiter, space, data):
        print("Separation!")
        # Reset the ball velocity
        ball.velocity = ball.velocity / np.linalg.norm(ball.velocity) * length
        return True

    handler = space.add_collision_handler(0, 0)
    handler.begin = begin
    handler.separate = separate

    # Render the screen
    exit_render = False
    while not exit_render:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_render = True
        # Check for key presses
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
            exit_render = True
        elif pressed[pygame.K_y]:
            # Save the pymunk static space with pickle (filename should be map_{start, start+amount}_{screen_width}_{screen_height}.pkl)
            with open("../maps/map_" + str(start) + "_" + str(start + amount) + "_" + str(SCREEN_WIDTH) + "_" + str(SCREEN_HEIGHT) + ".pkl", "wb") as f:
                # Also save the waypoints and boundary points and screen width and height and the pl_set
                pickle.dump((space_static, waypoints, boundary_points, (SCREEN_WIDTH, SCREEN_HEIGHT), pl_copy), f)

        # Draw ball
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (0, 0, 255), (int(ball.position[0]), int(ball.position[1])), 5)

        # Draw the polygon
        for pl in pl_set:
            for i in range(len(pl) - 1):
                pygame.draw.line(screen, (255, 0, 0), pl[i], pl[i + 1], 1)

        # Check for collisions
        collisions = space.shape_query(ball_shape)
        for collision in collisions:
            if collision.shape == ball_shape:
                print("Collision with ball!")
            if collision.shape.body.body_type == pymunk.Body.STATIC:
                ball_shape.color = (255, 0, 0)
                print("Collision with static body!")
            else:
                ball_shape.color = (0, 255, 0)

        render(screen, clock)

        # Update the ball
        space.step(1 / FPS)

    # Quit pygame
    pygame.quit()

    # # Load the pymunk space with pickle
    # with open("../maps/map.pkl", "rb") as f:
    #     space = pickle.load(f)