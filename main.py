import json
import random
import math
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Agent:
    """Represents a single agent in the simulation."""
    def __init__(self, id, x, y, speed, direction, commitment, eta):
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction  # Initial direction in radians
        self.commitment = commitment  # Initial commitment to a target (1, 2)
        self.eta = eta
        self.trajectory = [(x, y)]  # To store the trajectory of the agent
        self.received_commitments = {}  # Dictionary to store received commitments from neighbors {id: commitment}

    def update_position(self, target_x, target_y, no_turn_threshold, soft_turn_threshold, hard_turn_threshold):
        """Update the position of the agent based on its speed and direction towards the target."""
        target_direction = math.atan2(target_y - self.y, target_x - self.x)
        angle_diff = (target_direction - self.direction + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > hard_turn_threshold:
            # Hard turn: Turn in place
            self.direction += 0.05 * angle_diff  # Faster turning without moving
        elif abs(angle_diff) <= no_turn_threshold:
            # No turn: Move straight towards the target
            self.direction = target_direction
        else:
            # Soft turn: Adjust direction slightly while moving
            self.direction += 0.1 * angle_diff  # Scale turning speed
        if abs(angle_diff) <= hard_turn_threshold:
            # Update position if not turning in place
            self.x += self.speed * math.cos(self.direction)
            self.y += self.speed * math.sin(self.direction)


        self.trajectory.append((self.x, self.y))  # Store position in trajectory

    def update_commitment(self):

        if random.random() < self.eta:
            self.commitment = random.choice([1, 2])
        else:
            """Update the agent's commitment based on received commitments."""
            if self.received_commitments:
                unique_commitments = list(self.received_commitments.values())
                chosen_commitment = random.choice(unique_commitments)
                if chosen_commitment != 0:  # If 0, maintain current commitment
                    self.commitment = chosen_commitment
        self.received_commitments.clear()  # Clear the dictionary after update

    def __repr__(self):
        return f"Agent({self.id}, x={self.x:.2f}, y={self.y:.2f}, direction={self.direction:.2f})"

class LightSource:
    """Represents a light source in the simulation."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

class Simulation:
    """Manages the multi-agent simulation."""
    def __init__(self, config):
        self.num_agents = config["num_agents"]
        self.arena_size = config["arena_size"]
        self.time_steps = config["time_steps"]
        self.light_sources = [LightSource(**light) for light in config["light_sources"]]
        self.init_robot_bounds = config["init_robot_bounds"]
        self.robots_speed = config["robots_speed"]
        self.robots_direction = config["robots_direction"]
        self.no_turn_threshold = math.radians(config["no_turn_threshold"])
        self.soft_turn_threshold = math.radians(config["soft_turn_threshold"])
        self.hard_turn_threshold = math.radians(config["hard_turn_threshold"])
        self.steps_per_second = config["steps_per_second"]
        self.termination_radius = config["termination_radius"]
        self.visualize = config.get("visualize", True)  # Default to True if not specified
        self.commitment_update_time = config["commitment_update_time"]
        self.interval = 1000 // self.steps_per_second  # Milliseconds per frame
        self.agents = []
        self.history = []
        self.simulation_time = 0

    def initialize_agents(self):
        """Initialize agents using a default initializer."""
        half_agents = self.num_agents // 2
        commitments = [1] * half_agents + [2] * (self.num_agents - half_agents)
        random.shuffle(commitments)

        for i in range(self.num_agents):
            x = random.uniform(
                self.init_robot_bounds.get("x_min"),
                self.init_robot_bounds.get("x_max")
            )
            y = random.uniform(
                self.init_robot_bounds.get("y_min"),
                self.init_robot_bounds.get("y_max")
            )  # Start near the bottom of the arena
            direction = self.robots_direction #random.uniform(0, 2 * math.pi)
            speed = self.robots_speed#random.uniform(0.5, 2.0)
            self.agents.append(Agent(id=i, x=x, y=y, speed=speed, direction=direction, commitment=commitments[i], eta=0.05))

    def calculate_center_of_mass(self):
        """Calculate the center of mass of all agents."""
        x_com = sum(agent.x for agent in self.agents) / len(self.agents)
        y_com = sum(agent.y for agent in self.agents) / len(self.agents)
        return x_com, y_com

    def check_termination_condition(self):
        """Check if the center of mass is within the termination radius."""
        x_com, y_com = self.calculate_center_of_mass()
        for light in self.light_sources:
            distance_from_light = math.sqrt((x_com - light.x) ** 2 + (y_com - light.y) ** 2)
            #print('distance form light ', distance_from_light)
            if distance_from_light <= self.termination_radius:
                return True
        """arena_center = self.arena_size / 2
        distance_from_center = math.sqrt((x_com - arena_center) ** 2 + (y_com - arena_center) ** 2)
        return distance_from_center <= self.termination_radius"""

    def broadcast_commitments(self):
        """Agents broadcast their current commitments to neighbors."""
        for agent in self.agents:
            for neighbor in self.agents:
                if neighbor.id not in agent.received_commitments or agent.received_commitments[neighbor.id] != neighbor.commitment:
                    agent.received_commitments[neighbor.id] = neighbor.commitment

    def run(self, visualize=True):
        """Run the simulation for the specified number of time steps."""
        self.initialize_agents()

        if visualize:
            fig, ax = plt.subplots()
            ax.set_xlim(0, self.arena_size)
            ax.set_ylim(0, self.arena_size)

            # Plot light sources
            for light in self.light_sources:
                ax.scatter(light.x, light.y, color=light.color, s=100, label=f"Light ({light.color})")

            agent_scatter = ax.scatter([], [], s=50)
            trajectories = [ax.plot([], [], lw=1, alpha=0.6)[0] for _ in range(self.num_agents)]
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            def update_visuals():
                """Update the visualization at each timestep."""
                positions = self.history[-1]
                colors = ["green" if c == 1 else "yellow" for _, _, c in positions]
                agent_scatter.set_offsets([(x, y) for x, y, _ in positions])
                agent_scatter.set_color(colors)

                # Update trajectories
                for traj, agent in zip(trajectories, self.agents):
                    x_traj, y_traj = zip(*agent.trajectory)
                    traj.set_data(x_traj, y_traj)

                # Update time text
                time_text.set_text(f"Time: {self.simulation_time * 0.1:.1f} seconds")
                plt.pause(0.001)  # Pause to allow the plot to update

        for t in range(self.time_steps):
            start_time = time.time()  # Track the start time of the timestep
            self.simulation_time = t

            # Broadcast commitments
            self.broadcast_commitments()

            # Update commitments for all agents
            if t  % self.commitment_update_time == 0:
                for agent in self.agents:
                    agent.update_commitment()

            # Move all agents
            positions = []
            move_duration = 0.1
            step_start_time = time.time()
            #while time.time() - step_start_time < move_duration:
            for agent in self.agents:
                target_light = self.light_sources[agent.commitment - 1]  # Commitment 1 -> green, 2 -> yellow
                agent.update_position(
                    target_light.x, target_light.y,
                    self.no_turn_threshold, self.soft_turn_threshold, self.hard_turn_threshold
                )

                # Keep the agents within the bounds of the arena
                agent.x = max(0, min(agent.x, self.arena_size))
                agent.y = max(0, min(agent.y, self.arena_size))

                positions.append((agent.x, agent.y, agent.commitment))

            self.history.append(positions)

            # Update visuals if enabled
            if visualize:
                update_visuals()

            # Check termination condition
            #print("Time: {self.simulation_time * 0.1:.1f} seconds")
            if self.check_termination_condition():
                print(f"Termination condition met at simulation time: {self.simulation_time * 0.1:.1f} seconds")
                break

            # Ensure the loop runs for exactly 0.1 seconds
            elapsed_time = time.time() - start_time
            #print("elapsed time: ", elapsed_time)
            """if elapsed_time < 0.1:
                time.sleep(0.1 - elapsed_time)"""

        if visualize:
            plt.show()  # Keep the plot open after simulation ends if visualization is enabled

    def animate(self):
        """Visualize the simulation as it progresses."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.arena_size)
        ax.set_ylim(0, self.arena_size)

        # Plot light sources
        for light in self.light_sources:
            ax.scatter(light.x, light.y, color=light.color, s=100)#, label=f"Light ({light.color})")

        agent_scatter = ax.scatter([], [], s=50)
        trajectories = [ax.plot([], [], lw=1, alpha=0.6)[0] for _ in range(self.num_agents)]
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            positions = self.history[frame]
            colors = ["green" if c == 1 else "yellow" for _, _, c in positions]
            agent_scatter.set_offsets([(x, y) for x, y, _ in positions])
            agent_scatter.set_color(colors)

            # Update trajectories
            for traj, agent in zip(trajectories, self.agents):
                x_traj, y_traj = zip(*agent.trajectory[:frame+1])
                traj.set_data(x_traj, y_traj)

            # Update time text
            time_text.set_text(f"Time: {frame} steps")

            return agent_scatter, *trajectories, time_text

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=self.interval, blit=True)
        #plt.legend()
        plt.show()

def default_agents_initializer(num_agents, arena_size):
    """Default initializer for agents. Randomly assigns positions and directions within the arena."""
    agents = []
    half_agents = num_agents // 2
    commitments = [1] * half_agents + [2] * (num_agents - half_agents)
    random.shuffle(commitments)

    for i in range(num_agents):
        x = random.uniform((arena_size/2)-10, (arena_size/2)+10)
        y = random.uniform(0, 10)  # Start near the bottom of the arena
        direction = random.uniform(0, 2 * math.pi)  # Random direction
        speed = 2.0#random.uniform(0.5, 2.0)  # Random speed between 0.5 and 2.0
        agents.append(Agent(id=i, x=x, y=y, speed=speed, direction=direction, commitment=commitments[i], eta=0.05))
    return agents

if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Run the simulation
    simulation = Simulation(config)
    simulation.run()
