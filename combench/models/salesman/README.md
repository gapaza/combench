# Traveling Salesman Problem (TSP)

## Overview

The Traveling Salesman Problem (TSP) is a classic optimization problem where the goal is to find the shortest possible route that visits each city once and returns to the origin city. This implementation evaluates potential solutions to the TSP in terms of multiple objectives, specifically minimizing the total distance traveled and the total cost incurred during the tour.

## Problem Description

In the TSP, a salesman is given a list of cities and must determine the most efficient route to visit each city exactly once and return to the starting point. The challenge lies in the exponential number of possible routes as the number of cities increases. This implementation not only considers the total distance but also includes an evaluation of the cost associated with traveling between cities.

## Design Encoding Scheme

The design (tour) is represented as a list of integers, where each integer corresponds to a city. The order of the integers in the list defines the sequence in which the cities are visited.

For example, if there are 4 cities, a possible tour could be represented as:

    tour = [0, 2, 3, 1]

This denotes that the salesman starts at city 0, then goes to city 2, followed by city 3, and finally city 1 before returning to city 0.

## Objectives

The primary objectives in this TSP implementation are:

	1.	Minimize Total Distance Traveled: The total distance is calculated by summing the Euclidean distances between consecutive cities in the tour and from the last city back to the first.
	2.	Minimize Total Cost Incurred: The cost is evaluated similarly to the distance, considering a separate cost matrix.

## Real-Life Example

Imagine a delivery company that needs to deliver packages to a series of locations. The company’s goal is to minimize the distance traveled to reduce fuel costs and delivery time, as well as to minimize the overall cost of travel, which could include factors like tolls, road quality, or traffic congestion.

Given the coordinates of the cities (delivery locations) and the associated costs, the company can use this TSP implementation to find the optimal delivery route.


## Generator

This file generates city locations that create difficult instances of the TSP.

