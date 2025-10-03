# AI_Delivery_Shop 📦

A delivery shop application built with classic Artificial Intelligence (AI) to help delivery services find the least-cost paths, while taking package priority into account during delivery.

## Table Of Contents

* [Project Description](#project-description)
* [Problem Formulation](#problem-formulation)
* [Algorithms Used](#algorithms-used)
* [Test Samples](#test-samples)
* [Requirements](#requirements)

***

## Project Description

This is an application to help delivery shop owners find the least-cost way to deliver packages. The program uses heuristic search algorithms to solve a variant of the **Vehicle Routing Problem (VRP)**.

The program gives the user:

* **Complete GUI** for ease of use, built with `PySimpleGUI`.
* A **data-driven system** that reads package and vehicle information from the included `packages.csv` and `vehicles.csv` files.
* A choice of the algorithm to use: either **GA** (Genetic Algorithm) or **SA** (Simulated Annealing).
* A percentage setting for how much to focus on **package priority**, where zero is the least.

The core execution is managed by the Python script `delivery_shop.py`.

***

## Problem Formulation

### State:

A state represents which packages are assigned to each vehicle and the optimal delivery sequence for those packages (the route).
It is stored as a dictionary in the following format:

**State** = `{ V1: [P1, P2], V2: [P3, P4], ..., Vn: [Pn] }`

Where:
* `Vn` is a vehicle.
* `[Pn]` is the list of packages assigned to that vehicle, ordered by the determined delivery sequence.

### Objective Function:

The objective function evaluates how optimal a state is. The goal is to **minimize** this function. It considers two main factors:

1.  **Distance** – lower total distance results in lower cost.
2.  **Priority** – packages with higher priority (1 is highest) should be delivered earlier.

**Cost** = `Σ [ W1 × DirectCost + W2 × (1 / Priority) × PathCost ]`

Where:
* `W1`, `W2` are user-defined weights (related to the priority percentage setting).
* `DirectCost` is the straight-line cost (Euclidean distance from the shop).
* `Priority` is the priority value (1 is highest, 5 is lowest).
* `PathCost` is the path length or cost between delivery stops.

***

## Algorithms Used

### Simulated Annealing - SA

The algorithm iteratively explores the solution space. The way the next state is generated introduces stochastic variation:

**Next State**
The next state is generated randomly at each iteration by choosing one of the following methods:

1.  **Switching packages within the same vehicle** – two random packages from a randomly selected vehicle are swapped. This introduces variation in the route. No weight check is needed in this case.
2.  **Swapping packages between two different vehicles** – two random packages from two different vehicles are swapped. A weight constraint check is required.
3.  **Moving a package from one vehicle to another** – one random package is moved from a source vehicle to a different target vehicle. A weight constraint check is required.

### Genetic Algorithm - GA

The GA evolves a population of solutions using natural selection principles:

* **Chromosome (State):**
    A complete assignment of packages to vehicles. Each vehicle has its own route to deliver the assigned packages.
* **Population:**
    A set of 100 randomly generated chromosomes (based on the project specification).
* **Crossover:**
    Two parent chromosomes are selected. A fixed number of vehicles are randomly chosen from both parents, and their package lists are swapped, followed by the removal of duplicates and route reconstruction.
* **Mutation:**
    Some offspring are randomly selected for mutation. Mutation involves changing the tour (delivery route) of a randomly selected vehicle within the chromosome.

***
