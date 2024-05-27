import math
from combench.core.model import Model
import random

import numpy as np
import itertools


def centerMassCost(dimensions, locations):
    # Function to calculate the center of mass
    masses = []
    locx = []
    locy = []
    locz = []
    for i in range(len(dimensions)):
        # Assume constant density for all components. In the future add different densitites
        mass = dimensions[i][0] * dimensions[i][1] * dimensions[i][2]
        masses.append(mass)

        # Separate x,y,z components
        locx.append(locations[i][0])
        locy.append(locations[i][1])
        locz.append(locations[i][2])

    masses = np.array(masses)
    locx = np.array(locx)
    locy = np.array(locy)
    locz = np.array(locz)
    cmx = sum(np.multiply(masses, locx)) / sum(masses)
    cmy = sum(np.multiply(masses, locy)) / sum(masses)
    cmz = sum(np.multiply(masses, locz)) / sum(masses)
    cmMag = np.sqrt(cmx ** 2 + cmy ** 2 + cmz ** 2)
    return cmMag


def inertiaCost(dimensions, locations):
    # Function to calculate the moment of inertia
    # Method here is very simplified, will use full matrix and more complex analysis later
    # Treats each component as a point mass and minimizes sum of the magnitude of each inertia
    inertia = 0
    for i in range(len(dimensions)):
        mass = dimensions[i][0] * dimensions[i][1] * dimensions[i][2]
        distanceSquared = locations[i][0] ** 2 + locations[i][1] ** 2 + locations[i][2] ** 2
        inertia += mass * distanceSquared

    return inertia


def overlapCost(dimensions, locations):
    # Function to find how much overlap there are in all the elements
    overlap = 0

    # Find the min(x,y,z) and max(x,y,z) for each element
    elementCorners = []
    for i in range(len(dimensions)):
        minCorner = [locations[i][0] - dimensions[i][0] / 2,
                     locations[i][1] - dimensions[i][1] / 2,
                     locations[i][2] - dimensions[i][2] / 2]
        maxCorner = [locations[i][0] + dimensions[i][0] / 2,
                     locations[i][1] + dimensions[i][1] / 2,
                     locations[i][2] + dimensions[i][2] / 2]
        elCorners = [minCorner, maxCorner]
        elementCorners.append(elCorners)

    # Find the overlap between each pair of elements
    elCombList = itertools.combinations(elementCorners, 2)
    for comb in elCombList:
        (corners1, corners2) = comb
        xOverlap = min([corners1[1][0], corners2[1][0]]) - max([corners1[0][0], corners2[0][0]])
        yOverlap = min([corners1[1][1], corners2[1][1]]) - max([corners1[0][1], corners2[0][1]])
        zOverlap = min([corners1[1][2], corners2[1][2]]) - max([corners1[0][2], corners2[0][2]])
        if xOverlap >= 0 and yOverlap >= 0 and zOverlap >= 0:
            overlap += xOverlap * yOverlap * zOverlap

    return overlap


def wireCost(locations, types):
    # Extra cost from wires from here https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Customer+Drawing%7F10614%7FK%7Fpdf%7FEnglish%7FENG_CD_10614_K.pdf%7F865042-004
    # using manhattan distance for each component from pcu
    PCUInd = types.index("PCU")
    PCULoc = locations[PCUInd]
    totWireLen = 0
    for loc in locations:
        wireLen = np.abs(PCULoc[0] - loc[0]) + np.abs(PCULoc[1] - loc[1]) + np.abs(PCULoc[2] - loc[2])
        totWireLen += wireLen

    return totWireLen


def thermalCost(dimensions, locations):
    return 1


def vibrationsCost(dimensions, locatiions):
    return 1


def getCostComps(components):
    # pull parameters from the components
    locations = []
    dimensions = []
    types = []
    for comp in components:
        locations.append(comp.location)
        dimensions.append(comp.dimensions)
        types.append(comp.type)

    # Get the cost from each cost source
    overlapCostVal = overlapCost(dimensions, locations)
    cmCostCalVal = centerMassCost(dimensions, locations)
    inertiaCostVal = inertiaCost(dimensions, locations)
    wireCostVal = wireCost(locations, types)

    # Add all the costs together
    # In the future add weights for different costs
    cost = 1000 * overlapCostVal + cmCostCalVal + 3 * inertiaCostVal + wireCostVal
    return cost


def getCostParams(dimensions, locations, types):
    # Get the cost from each cost source
    overlapCostVal = overlapCost(dimensions, locations)
    cmCostCalVal = centerMassCost(dimensions, locations)
    inertiaCostVal = inertiaCost(dimensions, locations)
    wireCostVal = wireCost(locations, types)

    # Add all the costs together
    # In the future add weights for different costs
    cost = 1000 * overlapCostVal + cmCostCalVal + 3 * inertiaCostVal + wireCostVal
    return cost





class Zdt2Model:


    def __init__(self, problem_formulation):
        # super().__init__(problem_formulation)
        pass









if __name__ == '__main__':
    problem_formulation = {}
    model = Zdt2Model(problem_formulation)
