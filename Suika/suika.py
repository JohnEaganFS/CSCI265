# Imports
import pyglet
import random
import numpy as np
import pymunk

def create_window():
    window = pyglet.window.Window(500, 500, "Suika")
    return window

if __name__ == "__main__":
    # Create window
    window = create_window()
    batch = pyglet.graphics.Batch()

    # Create space
    space = pymunk.Space()
    space.gravity = 0, -100

    # Define balls list
    balls = []
    score = 0

    # Define floor & walls
    floor = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor_shape = pymunk.Poly(floor, [(100, 50), (400, 50), (400, 60), (100, 60)])
    floor_shape.elasticity = 1.0
    floor_draw = pyglet.shapes.Line(100, 55, 400, 55, width=10, color=(255, 255, 255), batch=batch)
    # floor_draw = pyglet.shapes.Polygon([(100, 50), (400, 50), (400, 60), (100, 60)], color=(255, 255, 255), batch=batch)

    left_wall = pymunk.Body(body_type=pymunk.Body.STATIC)
    left_wall_shape = pymunk.Poly(left_wall, [(100, 50), (100, 400), (110, 400), (110, 50)])
    left_wall_shape.elasticity = 1.0
    left_wall_draw = pyglet.shapes.Line(105, 50, 105, 400, width=10, color=(255, 255, 255), batch=batch)
    # left_wall_draw = pyglet.shapes.Polygon([(100, 50), (100, 400), (110, 400), (110, 50)], color=(255, 255, 255), batch=batch)

    right_wall = pymunk.Body(body_type=pymunk.Body.STATIC)
    right_wall_shape = pymunk.Poly(right_wall, [(400, 50), (400, 400), (390, 400), (390, 50)])
    right_wall_shape.elasticity = 1.0
    right_wall_draw = pyglet.shapes.Line(395, 50, 395, 400, width=10, color=(255, 255, 255), batch=batch)
    # right_wall_draw = pyglet.shapes.Polygon([(400, 50), (400, 400), (390, 400), (390, 50)], color=(255, 255, 255), batch=batch)

    # Add objects to space
    space.add(floor, floor_shape)
    space.add(left_wall, left_wall_shape)
    space.add(right_wall, right_wall_shape)

    # Add sensor region at the top of the walls to detect balls that have gone out of bounds
    sensor_region = pymunk.Body(body_type=pymunk.Body.STATIC)
    sensor_region_shape = pymunk.Poly(sensor_region, [(100, 400), (400, 400), (400, 410), (100, 410)])
    sensor_region_shape.sensor = True
    sensor_region_shape.collision_type = 2
    space.add(sensor_region, sensor_region_shape)
    
    # Collision handlers
    handler_ball_ball = space.add_collision_handler(1, 1)
    handler_ball_sensor = space.add_collision_handler(1, 2)

    # When balls collide, they merge into one big ball (if they are the same mass)
    def begin(arbiter, space, data):
        if arbiter.shapes[0].body.mass == arbiter.shapes[1].body.mass and arbiter.shapes[0] in balls and arbiter.shapes[1] in balls:
            # New position is the midpoint of the two balls
            pos = (arbiter.shapes[0].body.position + arbiter.shapes[1].body.position) / 2
            print(pos.x, pos.y)
            # Get new mass (add the masses together)
            mass = arbiter.shapes[0].body.mass + arbiter.shapes[1].body.mass
            # New radius is 1.5 times the radius of the first ball
            radius = arbiter.shapes[0].radius * 1.5
            # New velocity is the average of the two balls' velocities
            vel = (arbiter.shapes[0].body.velocity + arbiter.shapes[1].body.velocity) / 2

            # If the new position is outside the bounds of the game, move it back inside (border + radius)
            x, y = pos.x, pos.y
            if pos.x < 100 + radius:
                x = 100 + radius
            elif pos.x > 400 - radius:
                x = 400 - radius
            if pos.y < 50 + radius:
                y = 50 + radius
            
            pos = pymunk.Vec2d(x, y)

            # Create new ball (add the masses together, double the radius, etc.)
            ball = pymunk.Body(mass, 1, body_type=pymunk.Body.DYNAMIC)
            ball.position = pos
            ball.velocity = vel
            ball_shape = pymunk.Circle(ball, radius)
            ball_shape.elasticity = 0.4
            ball_shape.collision_type = 1
            ball_shape.color = (255, 255, 255)
            ball_shape.new_ball = False
            ball_shape.circle = pyglet.shapes.Circle(x=ball.position.x, y=ball.position.y, radius=ball_shape.radius, color=(255, 255, 255))
            space.add(ball, ball_shape)
            balls.append(ball_shape)

            # Delete old balls
            space.remove(arbiter.shapes[0].body, arbiter.shapes[0])
            space.remove(arbiter.shapes[1].body, arbiter.shapes[1])
            if arbiter.shapes[0] in balls:
                balls.remove(arbiter.shapes[0])
            if arbiter.shapes[1] in balls:
                balls.remove(arbiter.shapes[1])

            # Update score
            global score
            score += mass

            # Apply "explosion" force to all balls originating from the collision
            # for ball in balls:
            #     if ball != ball_shape and ball.body.position.get_distance(pos) < radius + ball.radius + 20:
            #         print(ball.body.position.x, ball.body.position.y)
            #         # Calculate force vector
            #         force = ball.body.position - pos
            #         # Apply force (scaled by distance from center of explosion)
            #         ball.body.apply_impulse_at_local_point(force * 100 / (force.length*force.length), (0, 0))
        return True

    # When a ball hits the sensor region, if it has hit the sensor region before, print "Game Over"
    def sensor_begin(arbiter, space, data):
        if arbiter.shapes[0].new_ball:
            arbiter.shapes[0].new_ball = False
            return True
        print("Game Over")
        return True

    handler_ball_ball.begin = begin
    handler_ball_sensor.begin = sensor_begin

    # Update
    def update(dt):
        space.step(dt)
        for ball in balls:
            ball.circle.x = ball.body.position.x
            ball.circle.y = ball.body.position.y

    # Mouse click (add ball)
    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            ball = pymunk.Body(1, 1, body_type=pymunk.Body.DYNAMIC)
            ball.position = x, 480
            ball_shape = pymunk.Circle(ball, 10)
            ball_shape.elasticity = 0.4
            ball_shape.collision_type = 1
            ball_shape.color = (255, 255, 255)
            ball_shape.new_ball = True
            ball_shape.circle = pyglet.shapes.Circle(x=x, y=480, radius=10, color=(255, 255, 255))
            space.add(ball, ball_shape)
            balls.append(ball_shape)

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        # Draw balls
        for ball in balls:
            ball.circle.draw()
        # Draw label for score
        pyglet.text.Label("Score: " + str(int(score)), font_name="Times New Roman", font_size=16, x=10, y=10).draw()

    pyglet.clock.schedule_interval(update, 1/60.0)

    pyglet.app.run()

