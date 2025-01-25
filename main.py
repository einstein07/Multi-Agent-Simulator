import csv
import json
import os
import random
import math
import time
from datetime import datetime

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
        #self.received_commitments = {}  # Dictionary to store received commitments from neighbors {id: commitment}
        self.received_broadcasts = {}  # Track received broadcasts at each timestep
        self.broadcast = None  # Current broadcast value
        self.my_opinions = [] #keeps track of this agent's opinion broadcasts
        self.log_file = None  # File handle for this agent's log file

    def initialize_log_file(self, simulation_start_time, run_folder, experiment_name):
        """Initialize the log file for this agent."""
        filename = os.path.join(run_folder, f"{experiment_name}_bot{self.id}_{simulation_start_time}.csv")
        self.log_file = open(filename, "w", newline="")
        writer = csv.writer(self.log_file)
        writer.writerow(["Time", "Commitment", "Opinion"])
        self.csv_writer = writer

    def close_log_file(self):
        """Close the agent's log file."""
        if self.log_file:
            self.log_file.close()

    def update_position(self, target_x, target_y, no_turn_threshold, soft_turn_threshold, hard_turn_threshold,
                        angular_speed):
        #Update the position of the agent based on its speed and direction towards the target.
        target_direction = math.atan2(target_y - self.y, target_x - self.x)
        angle_diff = (target_direction - self.direction + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > hard_turn_threshold:
            # Hard turn: Turn in place using angular speed
            self.direction += angular_speed * angle_diff
            #print('Hard-turn, new direction: ', math.degrees(self.direction))
        elif abs(angle_diff) <= no_turn_threshold:
            # No turn: Move straight towards the target
            self.direction = target_direction
            #print('No-turn, direction: ', math.degrees(self.direction), ' angle diff: ', math.degrees(angle_diff))
        else:
            # Soft turn: Adjust direction slightly while moving using angular speed
            self.direction += 0.5 * angular_speed * angle_diff  # Scale angular speed for soft turning
            #print('Soft-turn, new direction: ', math.degrees(self.direction))
        # Update position if not turning in place
        if abs(angle_diff) <= hard_turn_threshold:
            self.x += self.speed * math.cos(self.direction)
            self.y += self.speed * math.sin(self.direction)

        self.trajectory.append((self.x, self.y))  # Store position in trajectory

    def determine_broadcast(self, hard_turn_threshold, angle_diff):
        """Determine what the agent broadcasts."""
        if abs(angle_diff) > hard_turn_threshold:
            self.broadcast = 0  # Hard-turn: Broadcast 0
        else:
            self.broadcast = self.commitment  # Otherwise, broadcast commitment
        self.my_opinions.append(self.broadcast)

    def receive_broadcast(self, sender_id, broadcast_value):
        """Receive a broadcast from another agent."""
        self.received_broadcasts[sender_id] = broadcast_value

    def update_commitment(self):
        dice = random.random()
        if dice < self.eta:
            #print('value of dice: ', dice, ' value of eta: ', self.eta)
            self.commitment = random.choice([1, 2])
        else:
            """Update the agent's commitment based on received commitments."""
            if self.received_broadcasts:
                unique_commitments = list(self.received_broadcasts.values())
                chosen_commitment = random.choice(unique_commitments)
                if chosen_commitment != 0:  # If 0, maintain current commitment
                    self.commitment = chosen_commitment
        self.received_broadcasts.clear()  # Clear the dictionary after update

    def log_data(self, time_step):
        """Log the agent's data to its CSV file."""
        #opinions = "; ".join(str(self.received_broadcasts.get(neighbor_id, 0)) for neighbor_id in
        #                     self.received_broadcasts.keys())
        opinions = "; ".join(map(str, self.my_opinions))
        self.csv_writer.writerow([time_step, self.commitment, opinions])
        self.my_opinions.clear()  # Clear after logging

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
        self.start_time = datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss%fus")  # Simulation start time
        self.base_log_dir = config.get("log_directory", "./logs")  # Base directory for logs
        self.experiment_name = config.get("experiment_name")
        self.num_agents = config["num_agents"]
        self.num_runs = config["num_runs"]  # Number of experiment runs
        self.x_width = config["x_width"]
        self.y_height = config["y_height"]
        self.center = config["center"]
        self.time_steps = config["time_steps"]
        self.light_sources = [LightSource(**light) for light in config["light_sources"]]
        self.init_robot_bounds = config["init_robot_bounds"]
        self.robots_speed = config["robots_speed"]
        self.angular_speed = math.radians(config["angular_speed"])
        self.eta = config["eta"]
        self.robots_direction = math.radians(config["robots_direction"])
        self.no_turn_threshold = math.radians(config["no_turn_threshold"])
        self.soft_turn_threshold = math.radians(config["soft_turn_threshold"])
        self.hard_turn_threshold = math.radians(config["hard_turn_threshold"])
        self.steps_per_second = config["steps_per_second"]
        self.termination_radius = config["termination_radius"]
        self.visualize = config.get("visualize")  # Default to True if not specified
        self.commitment_update_time = config["commitment_update_time"]
        self.interval = 1000 // self.steps_per_second  # Milliseconds per frame
        self.agents = []
        self.history = []
        self.simulation_time = 0
        self.log_file = None  # File handle for this agent's log file


        # Compute arena bounds based on center and dimensions
        half_x = self.x_width / 2
        half_y = self.y_height / 2
        self.arena_bounds = {
            "x_min": self.center[0] - half_x,
            "x_max": self.center[0] + half_x,
            "y_min": self.center[1] - half_y,
            "y_max": self.center[1] + half_y
        }

    def create_run_folder(self, run_index):
        """Create a folder for each experiment run."""
        run_folder = os.path.join(self.base_log_dir, f"run{run_index}")
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def initialize_agents(self, run_folder):
        self.agents.clear()
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
            agent = Agent(id=i, x=x, y=y, speed=self.robots_speed, direction=self.robots_direction, commitment=commitments[i], eta=self.eta)

            agent.initialize_log_file(self.start_time, run_folder, self.experiment_name)  # Initialize log file
            self.agents.append(agent)

    def initialize_log_file(self, run_folder):
        """Initialize the log file for this agent."""
        filename = os.path.join(run_folder, f"{self.experiment_name}_positions_{self.start_time}.csv")
        self.log_file = open(filename, "w", newline="")
        writer = csv.writer(self.log_file)
        writer.writerow(["Time", "ID", "x", "y"])
        self.csv_writer = writer

    def log_positions_data(self, time_step, agent):
        """Log the agent's data to its CSV file."""
        self.csv_writer.writerow([time_step, agent.id, agent.x, agent.y])

    def log_agent_data(self, time_step):
        """Log data for each agent."""
        if time_step > 0 and time_step % self.commitment_update_time == 0:
            for agent in self.agents:
                agent.log_data(time_step)

    def get_light_source_in_sight(self, agent):
        """Check if the committed light source is in sight."""
        target_light = self.light_sources[agent.commitment - 1]
        # Placeholder logic: Always return the light source as 'in sight'
        return agent.commitment  # Modify this as needed

    def close_all_logs(self):
        """Close all agent log files."""
        for agent in self.agents:
            agent.close_log_file()

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
                #if neighbor.id not in agent.received_commitments or agent.received_commitments[neighbor.id] != neighbor.commitment:
                if agent.id != neighbor.id:  # Prevent self-broadcasting
                    neighbor.receive_broadcast(agent.id, agent.broadcast)
                    #agent.received_commitments[neighbor.id] = neighbor.commitment

    def run_experiments(self):
        """Run the specified number of experiments."""
        for run_index in range(self.num_runs):
            print(f"Starting experiment run {run_index}...")
            run_folder = self.create_run_folder(run_index)
            self.run(run_folder)
            print(f"Experiment run {run_index} completed.")

    def run(self, run_folder):
        """Run the simulation for the specified number of time steps."""
        self.initialize_agents(run_folder)
        self.initialize_log_file(run_folder)

        if self.visualize:
            fig, ax = plt.subplots(figsize=(7, 8))
            ax.set_xlim(self.arena_bounds["x_min"], self.arena_bounds["x_max"])
            ax.set_ylim(self.arena_bounds["y_min"], self.arena_bounds["y_max"])

            # Plot light sources
            for light in self.light_sources:
                ax.scatter(light.x, light.y, color=light.color, s=100, label=f"Light ({light.color})")

            # Initialize agent markers as triangle patches
            agent_markers = [
                ax.add_patch(plt.Polygon([(0, 0)], closed=True, alpha=0.7))
                for _ in range(self.num_agents)
            ]
            trajectories = [ax.plot([], [], lw=1, alpha=0.6)[0] for _ in range(self.num_agents)]
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            def update_visuals():
                """Update the visualization at each timestep."""
                positions = self.history[-1]

                # Update agent markers
                for agent_marker, (x, y, commitment), agent in zip(agent_markers, positions, self.agents):
                    color = "green" if commitment == 1 else "red"
                    size_long = 1.0  # Length of the longest side (tip pointing forward)
                    size_short = 0.1  # Length of the shorter base sides
                    direction = agent.direction

                    # Define triangle vertices based on position and direction
                    triangle_vertices = [
                        (x + size_long * math.cos(direction), y + size_long * math.sin(direction)),  # Tip
                        (x + size_short * math.cos(direction + 2.5 * math.pi / 3),
                         y + size_short * math.sin(direction + 2.5 * math.pi / 3)),  # Bottom left
                        (x + size_short * math.cos(direction - 2.5 * math.pi / 3),
                         y + size_short * math.sin(direction - 2.5 * math.pi / 3)),  # Bottom right
                    ]

                    # Update the triangle marker with new vertices and color
                    agent_marker.set_xy(triangle_vertices)
                    agent_marker.set_facecolor(color)


                # Update trajectories
                for traj, agent in zip(trajectories, self.agents):
                    x_traj, y_traj = zip(*agent.trajectory)
                    traj.set_data(x_traj, y_traj)

                # Update time text
                #time_text.set_text(f"Time: {self.simulation_time * 0.1:.1f} seconds")
                plt.pause(0.001)  # Pause to allow the plot to update

        for t in range(self.time_steps):
            start_time = time.time()  # Track the start time of the timestep
            self.simulation_time = t

            # Broadcast commitments every timestep
            for agent in self.agents:
                target_light = self.light_sources[agent.commitment - 1]
                target_direction = math.atan2(target_light.y - agent.y, target_light.x - agent.x)
                angle_diff = (target_direction - agent.direction + math.pi) % (2 * math.pi) - math.pi
                agent.determine_broadcast(self.hard_turn_threshold, angle_diff)

            self.broadcast_commitments()

            # Log data for each agent
            self.log_agent_data(t)
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
                    self.no_turn_threshold, self.soft_turn_threshold, self.hard_turn_threshold,
                    angular_speed=self.angular_speed
                )

                # Keep the agents within the bounds of the arena
                #agent.x = max(self.arena_bounds["x_min"]+2, min(agent.x, self.arena_bounds["x_max"]-2))
                #agent.y = max(self.arena_bounds["y_min"]+2, min(agent.y, self.arena_bounds["y_max"]-2))
                #print('ID: ', agent.id, ' x: ', agent.x, ' y: ', agent.y)
                positions.append((agent.x, agent.y, agent.commitment))

                if t % self.commitment_update_time == 0:
                    self.log_positions_data(t, agent)

            self.history.append(positions)

            # Update visuals if enabled
            if self.visualize:
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

        # Close all log files
        self.close_all_logs()

        if self.visualize and self.num_runs <= 1:
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
            colors = ["green" if c == 1 else "red" for _, _, c in positions]
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
        agents.append(Agent(id=i, x=x, y=y, speed=speed, direction=direction, commitment=commitments[i], eta=0.0))
    return agents

if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Run the simulation
    simulation = Simulation(config)
    simulation.run_experiments()  # Run multiple experiments
