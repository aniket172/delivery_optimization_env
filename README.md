---
title: Delivery Optimisation Environment Server
emoji: 🥁
colorFrom: pink
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Delivery Optimisation Environment

A reinforcement learning environment that simulates real-world food
delivery logistics similar to platforms like Swiggy, Zomato, Uber Eats,
and DoorDash.

The environment evaluates how well an AI agent can dispatch delivery
drivers, batch orders, and reposition fleets under dynamic conditions
such as traffic, weather, and demand fluctuations.

Built using the OpenEnv framework, this environment provides a
standardized interface (reset(), step(), state()) for training and
evaluating AI agents on complex operational decision-making tasks.

------------------------------------------------------------------------

Problem Overview

At its core, this environment simulates a city-scale logistics system.

Orders appear dynamically across a city graph while drivers move across
the network to pick up and deliver them.

An AI agent acts as the central dispatch system and must decide:

-   Which driver should handle which order
-   Whether multiple orders should be batched
-   When to reposition idle drivers to high-demand zones

The agent is rewarded for efficient delivery operations while penalized
for delays, cancellations, and fuel consumption.

------------------------------------------------------------------------

Environment Design

The city is modeled as a graph network using NetworkX.

Nodes represent intersections or city zones while edges represent roads
with travel time weights.

Drivers move along graph paths and must travel over time to reach pickup
and drop locations.

------------------------------------------------------------------------

State (Observation Space)

The observation returned to the agent includes:

-   Current simulation time
-   List of drivers with location and status
-   Pending delivery orders
-   Traffic level
-   Weather conditions
-   Demand heatmap across city zones

------------------------------------------------------------------------

Action Space

The agent can perform three types of actions.

Assignments: Assign a driver to a specific order.

Batch Assignments: Batch multiple compatible orders to the same driver.

Reposition: Move idle drivers to predicted demand zones.

------------------------------------------------------------------------

Environment Dynamics

Every step simulates approximately five minutes of real-world time.

During each step:

-   New orders appear across the city
-   Drivers complete deliveries if travel time has elapsed
-   Traffic and weather influence travel time
-   Orders expire if deadlines are missed

Drivers follow a state machine:

idle → assigned → delivering → idle

------------------------------------------------------------------------

Reward Function

The reward captures operational efficiency:

-   completed deliveries
-   on-time deliveries
-   priority order success
-   batching efficiency
-   cancelled orders
-   late deliveries
-   fuel cost

The final score is normalized between 0.0 and 1.0.

------------------------------------------------------------------------

Tasks

The environment contains three difficulty levels.

Easy: Small city graph, low traffic, fewer orders.

Medium: Larger graph, higher order arrival rate, increased traffic.

Hard: Large city, heavy traffic, driver failures, high order volume.

------------------------------------------------------------------------

Running the Environment

Start the server:

uvicorn server.app:app –host 0.0.0.0 –port 8000

Health check: http://localhost:8000/health

------------------------------------------------------------------------

Running Inference

python inference.py

The agent will reset the environment, query an LLM for decisions,
execute actions, and log rewards and scores.

------------------------------------------------------------------------

Project Structure

delivery_optimisation/

env/ - delivery_optimisation_environment.py - models.py - tasks.py -
graders.py - client.py

inference.py openenv.yaml Dockerfile README.txt

------------------------------------------------------------------------

Key Features

-   Graph-based city simulation using NetworkX
-   Multi-driver fleet management
-   Dynamic demand generation
-   Order batching
-   Priority deliveries
-   Driver travel-time simulation
-   Traffic and weather effects
-   Reward shaping for logistics optimization

------------------------------------------------------------------------

Applications

-   Reinforcement learning research
-   Multi-agent coordination
-   Fleet optimization
-   Agentic AI benchmarking
-   Logistics decision-making systems
