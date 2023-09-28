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

if __name__ == "__main__":
    x = boundaries(np.array([0, 0]), np.array([1, 1]))
    print(x)