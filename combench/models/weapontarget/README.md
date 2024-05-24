# Weapon Target Assigning Problem

## Overview

This repository contains a Python implementation of a weapon target assigning problem. The main objective is to allocate weapons to various targets in such a way that minimizes the total expected value of surviving targets. This is a classic problem in operations research and military logistics, where efficient resource allocation is critical.

## Problem Description

The weapon target assigning problem involves a set of targets, each with an associated value, and a set of weapons of different types, each with a probability of successfully destroying a target. The goal is to assign these weapons to the targets in a way that minimizes the total expected value of the targets that survive the attack.

Key Components

	•	Targets (V): Each target has a specific value representing its importance or the damage it could cause if it survives.
	•	Weapons (W): There are multiple types of weapons, each type with a different number of available units.
	•	Probabilities (p): Each weapon type has a probability of successfully destroying each target.

## Design Encoding Scheme

The design is encoded as a list where each element represents the target assigned to a particular weapon. The index of the list corresponds to the weapon, and the value at that index represents the target to which the weapon is assigned.

### Example

Given:

	•	V = [10, 20, 30, 40, 50, 60, 70, 80, 90]: Values of the targets.
	•	W = [4, 3]: Number of weapons of each type (4 of type 1, 3 of type 2).
	•	p = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3]]: Probability matrix for weapon types.

A design [0, 1, 8, 3, 4, 5, 6] means:

	•	Weapon 1 (type 1) is assigned to target 0.
	•	Weapon 2 (type 1) is assigned to target 1.
	•	Weapon 3 (type 1) is assigned to target 8.
	•	Weapon 4 (type 1) is assigned to target 3.
	•	Weapon 5 (type 2) is assigned to target 4.
	•	Weapon 6 (type 2) is assigned to target 5.
	•	Weapon 7 (type 2) is assigned to target 6.

## Objectives

The primary objective is to minimize the total expected survival value of the targets. This is calculated as the sum of the values of the targets multiplied by their respective survival probabilities after the assignment of weapons.

### Calculation

	1.	Survival Probability: For each target, compute the probability of it surviving the attack by all assigned weapons.
	2.	Expected Survival Value: Multiply the value of each target by its survival probability and sum these values to get the total expected survival value.

## Real-Life Example

Consider a scenario where a defense system has multiple types of weapons (e.g., missiles, drones) and needs to allocate them to a set of enemy targets (e.g., tanks, bunkers, command centers). Each weapon has a different effectiveness against each target. The defense system’s goal is to minimize the potential damage by ensuring the most valuable targets are destroyed with the highest probability.