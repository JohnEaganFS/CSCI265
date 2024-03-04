import numpy as np

def boundaries(a, b):
    # a and b are both points in space (x, y)
    # Start by getting the vector from a to b
    vector = b - a

    # point 1 is this vector rotated 90 degrees and scaled to the desired length
    point1 = np.array([vector[1], -vector[0]])

    # point 2 is the same vector rotated 270 degrees and scaled to the desired length
    point2 = np.array([-vector[1], vector[0]])

    # point 3 is point 1 + vector
    point3 = point1 + vector

    # point 4 is point 2 + vector
    point4 = point2 + vector

    # Return the four points
    return [point1, point2, point3, point4]

def defineBoundaries(boundary_points):
    ''' Define the "limits" produced by the boundary points.
    In other words, this is the possible range of values for (x,y) that are within the boundary.
    '''
    # Boundary not necessarily aligned with x and y axes so need to rotate the boundary points to align with x and y axes
    # It is a rectangle so can align with either x or y axis and the other axis will also be aligned
    # Rotate the boundary points so that the boundary is aligned with the x axis
    # Find the angle between the boundary and the x axis
    # Use first and third points to find vector representing direction of boundary
    vector = boundary_points[2] - boundary_points[0]
    print(vector)
    # Find the angle between this vector and the x axis
    angle = np.arctan(vector[1] / vector[0])
    print(angle*180/np.pi)

    # Rotate the boundary points by the negative of this angle to rotate the boundary to align with the x axis
    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    print(rotation_matrix)
    # Rotate the boundary points ()
    rotated_boundary_points = []
    for point in boundary_points:
        rotated_boundary_points.append(np.matmul(rotation_matrix, point))
    print(rotated_boundary_points)

    min_x, max_x, min_y, max_y = 0, 0, 0, 0

    # Find out which points are aligned with the x axis
    # They should have the same y value and the pairs have to be either 1 and 3 or 2 and 4
    if (rotated_boundary_points[0][1] == rotated_boundary_points[2][1]):
        # Points 1 and 3 are aligned with the x axis
        # Find the minimum and maximum x values
        min_x, max_x = min(rotated_boundary_points[0][0], rotated_boundary_points[2][0]), max(rotated_boundary_points[0][0], rotated_boundary_points[2][0])
        # Find the minimum and maximum y values based on the y values of points 2 and 4
        min_y, max_y = min(rotated_boundary_points[1][1], rotated_boundary_points[3][1]), max(rotated_boundary_points[1][1], rotated_boundary_points[3][1])
    else:
        # Points 2 and 4 are aligned with the x axis
        # Find the minimum and maximum x values
        min_x, max_x = min(rotated_boundary_points[1][0], rotated_boundary_points[3][0]), max(rotated_boundary_points[1][0], rotated_boundary_points[3][0])
        # Find the minimum and maximum y values based on the y values of points 1 and 3
        min_y, max_y = min(rotated_boundary_points[0][1], rotated_boundary_points[2][1]), max(rotated_boundary_points[0][1], rotated_boundary_points[2][1])

    rotated_limits = [min_x, max_x, min_y, max_y]

    print("Rotated Limits:", rotated_limits)

    # Rotate the limits back to the original orientation
    limits = []
    for point in rotated_limits:
        limits.append(np.matmul(rotation_matrix.T, np.array([point, 0])))
    print("Limits:", limits)


    return rotated_boundary_points, limits

def checkIfInRectangle(point, boundary_points):
    ''' Check if a point is in a rectangle defined by the boundary points.
    '''
    # Get the boundary points
    a, b, c, d = boundary_points[0], boundary_points[1], boundary_points[2], boundary_points[3]
    # Check if point is in rectangle
    vector_ap = point - a
    vector_ab = b - a
    vector_ad = d - a
    return 0 <= np.dot(vector_ap, vector_ab) <= np.dot(vector_ab, vector_ab) and 0 <= np.dot(vector_ap, vector_ad) <= np.dot(vector_ad, vector_ad)

def choosePoint(boundary_points, ax, ay):
    dx = boundary_points[2][0] - boundary_points[0][0]
    dy = boundary_points[1][1] - boundary_points[0][1]
    return np.array([boundary_points[0][0] + dx*ax, boundary_points[0][1] + dy*ay])

if __name__ == "__main__":
    a, b = np.array([0, 0]), np.array([1, 2])
    x = boundaries(a, b)

    print("Boundary Points", x)
    sample_point = choosePoint(x, 0.5, 0.5)
    print("Sample Point", sample_point)



    print("In Rectangle?", checkIfInRectangle(sample_point, x))

    # Plot the points
    import matplotlib.pyplot as plt

    # Plot the rotated boundary points
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    # Add grid lines
    plt.grid()
    # Make axes increment by 1
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    for i, point in enumerate(x):
        # label points 1 and 3
        if i == 0 or i == 2:
            plt.text(point[0], point[1], str(i+1))
        plt.plot(point[0], point[1], 'ro')
    plt.plot(sample_point[0], sample_point[1], 'bo')
    plt.show()



    # rotated_points, limits = defineBoundaries(x)

    
    # # Plot the rotated boundary points
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # # Add grid lines
    # plt.grid()
    # # Make axes increment by 1
    # plt.xticks(np.arange(-10, 10, 1))
    # plt.yticks(np.arange(-10, 10, 1))
    # for i, point in enumerate(rotated_points):
    #     # label points 1 and 3
    #     if i == 0 or i == 2:
    #         plt.text(point[0], point[1], str(i+1))
    #     plt.plot(point[0], point[1], 'ro')
    # # Plot original boundary points
    # for i, point in enumerate(x):
    #     if i == 0 or i == 2:
    #         plt.text(point[0], point[1], str(i+1))
    #     plt.plot(point[0], point[1], 'bo')
    # plt.show()



