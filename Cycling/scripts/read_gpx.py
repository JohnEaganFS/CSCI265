# This script will read a .gpx file and return a list of lat/lon coordinates along with elevation data for each point.
# Essentially, I'm trying to capture and save the route data for future manipulation wiht numpy and matplotlib.

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def boundaries(a, b):
    # get a and b if translated such that a is at the origin
    vector = b - a

    point1 = np.array([vector[1], -vector[0]])
    point2 = np.array([-vector[1], vector[0]])
    point3 = point1 + vector
    point4 = point2 + vector

    # translate back to original position
    point1 = point1 + a
    point2 = point2 + a
    point3 = point3 + a
    point4 = point4 + a

    # Return the four points
    return [point1, point2, point3, point4]

def read_gpx(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    # Get the lat/lon coordinates (and elevation) for each point in the .gpx file
    points = []

    trkElement = root[1]
    trksegElement = trkElement[2]
    for trkpt in trksegElement:
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        ele = float(trkpt[0].text)
        points.append((lat, lon, ele))

    # Convert the list of points to a numpy array
    points = np.array(points)

    return points

def scaleData(data):
    min_lat = data[:,0].min()
    max_lat = data[:,0].max()
    min_lon = data[:,1].min()
    max_lon = data[:,1].max()
    min_ele = data[:,2].min()
    max_ele = data[:,2].max()
    for i in range(len(data)):
        data[i][0] = (data[i][0] - min_lat) / (max_lat - min_lat)
        data[i][1] = (data[i][1] - min_lon) / (max_lon - min_lon)
        data[i][2] = (data[i][2] - min_ele) / (max_ele - min_ele)
    
    return data

def drawPoints(points):
    # Make the markers go from red to blue as the ride progresses
    colors = np.linspace(0, 1, len(points))

    # Draw the points in 3d space with (x, y, z) representing (lat, lon, ele) and lines connecting each point
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors)
    ax.plot(points[:,0], points[:,1], points[:,2], c='black')

    
    # Set the elevation range to (min - 10, max + 10) to make the graph look better
    #ax.set_zlim(points[:,2].min() - 10, points[:,2].max() + 10)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Elevation')
    plt.show()

def drawBoundaryPoints(boundaryPoints, points):
    # Create a list of colors in the form (r, g, b) where r, g, and b are floats between 0 and 1
    colors = np.linspace(0, 1, len(boundaryPoints))

    # Draw the points in 3d space with (x, y, z) representing (lat, lon, ele) and lines connecting each point
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, bp in enumerate(boundaryPoints):
        # Create lines between points
        # 1 and 2
        ax.plot([bp[0][0], bp[1][0]], [bp[0][1], bp[1][1]], [bp[0][2], bp[1][2]], marker='o', markerfacecolor=(colors[i], 0.0, 0.0), markersize=0.5, color=(colors[i], 0.0, 0.0), linewidth=0.5)
        # 2 and 4
        ax.plot([bp[1][0], bp[3][0]], [bp[1][1], bp[3][1]], [bp[1][2], bp[3][2]], c='black')
        # 4 and 3
        ax.plot([bp[3][0], bp[2][0]], [bp[3][1], bp[2][1]], [bp[3][2], bp[2][2]], c='black')
        # 3 and 1
        ax.plot([bp[2][0], bp[0][0]], [bp[2][1], bp[0][1]], [bp[2][2], bp[0][2]], c='black')

    ax.scatter(points[:,0], points[:,1], points[:,2], c='green', s=5)

    plt.show()

def removeDuplicatePoints(data):
    newData = []
    for i in range(len(data) - 1):
        if data[i][0] == data[i+1][0] and data[i][1] == data[i+1][1]:
            continue
        newData.append(data[i])
    newData.append(data[-1])
    return np.array(newData)


if __name__ == "__main__":
    data = read_gpx("../gpx/Evening_Ride.gpx")

    # Remove duplicate points
    data = removeDuplicatePoints(data)

    # Scale the lat/lon and ele coordinates to be in the range [0, 1]
    data = scaleData(data)


    # drawPoints(data[:100])

    # for point in data:
    #     print(point.lat, point.lon, point.ele)

    boundaryPoints = []
    newData = []
    for i in range(len(data) - 1):
        # if equivalent points, skip
        if data[i][0] == data[i+1][0] and data[i][1] == data[i+1][1]:
            continue
        bp = boundaries(data[i][:2], data[i+1][:2])
        # ele will be the average of the two points
        ele = (data[i][2] + data[i+1][2]) / 2
        boundaryPoints.append(((bp[0][0], bp[0][1], ele), (bp[1][0], bp[1][1], ele), (bp[2][0], bp[2][1], ele), (bp[3][0], bp[3][1], ele)))
        newData.append(data[i])
    newData.append(data[-1])

    start = 500
    limit = 510

    #print(boundaryPoints[start:limit])
    
    drawBoundaryPoints(boundaryPoints[start:limit], data[start:limit])



    