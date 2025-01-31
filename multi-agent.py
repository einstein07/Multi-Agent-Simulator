import csv
import json
import os
import random
import math
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------------
# 1. States
# --------------------------------------------------------------------
NO_TURN   = "NO_TURN"
SOFT_TURN = "SOFT_TURN"
HARD_TURN = "HARD_TURN"

class WheelTurningParams:
    def __init__(self, turning_mechanism,
                 BaseSpeed,
                 hard_turn_on_angle_threshold,
                 soft_turn_on_angle_threshold,
                 no_turn_angle_threshold):
        self.BaseSpeed = BaseSpeed
        self.turning_mechanism = turning_mechanism
        self.HardTurnOnAngleThreshold = hard_turn_on_angle_threshold
        self.SoftTurnOnAngleThreshold = soft_turn_on_angle_threshold
        self.NoTurnAngleThreshold = no_turn_angle_threshold


class Agent:
    """Represents a single agent in the simulation."""

    def __init__(self, id, x, y, speed, track_width, direction, commitment, eta, light_ids, thresholds, turning_mechanism):
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed
        self.track_width = track_width
        self.direction = direction  # Initial direction in radians
        self.commitment = commitment
        self.eta = eta
        self.light_ids = light_ids
        self.trajectory = [(x, y)]
        self.received_broadcasts = {}
        self.broadcast = None
        self.my_opinions = []
        self.log_file = None
        self.wheel_turning_params = WheelTurningParams(
            BaseSpeed=self.speed,
            turning_mechanism=turning_mechanism,
            hard_turn_on_angle_threshold=thresholds['hard'],
            soft_turn_on_angle_threshold=thresholds['soft'],
            no_turn_angle_threshold=thresholds['none']
        )
        #print('received TM: ', turning_mechanism, ' stored: ', self.wheel_turning_params.turning_mechanism)

    def initialize_log_file(self, simulation_start_time, run_folder, experiment_name):
        """Initialize the agent's log file."""
        filename = os.path.join(run_folder, f"{experiment_name}_bot{self.id}_{simulation_start_time}.csv")
        self.log_file = open(filename, "w", newline="")
        writer = csv.writer(self.log_file)
        writer.writerow(["Time", "Commitment", "Opinion"])
        self.csv_writer = writer

    def close_log_file(self):
        """Close the agent's log file."""
        if self.log_file:
            self.log_file.close()

    def normalize_angle(sef, angle):
        """
        Returns angle in [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    # --------------------------------------------------------------------
    # 3. State Transition Function
    # --------------------------------------------------------------------
    def update_turning_mechanism(self, c_heading_angle, params):
        #print('init TM: ', params.turning_mechanism)
        """
        Update the turning mechanism based on the absolute heading angle
        and threshold values, replicating the logic from the original C++ code.

        :param current_mechanism: One of (NO_TURN, SOFT_TURN, HARD_TURN)
        :param c_heading_angle: (float) Current heading angle difference we need to correct
        :param params: WheelTurningParams instance
        :return: new_mechanism (string)
        """
        abs_angle = abs(c_heading_angle)

        # The original code did these checks in sequence.
        # We replicate that structure:

        # 1) If currently HARD_TURN, check possible switch to SOFT_TURN
        if params.turning_mechanism == HARD_TURN:
            if abs_angle <= params.SoftTurnOnAngleThreshold:
                params.turning_mechanism = SOFT_TURN

        # 2) If currently SOFT_TURN, check possible switch to HARD_TURN or NO_TURN
        if params.turning_mechanism == SOFT_TURN:
            if abs_angle > params.HardTurnOnAngleThreshold:
                params.turning_mechanism = HARD_TURN
            elif abs_angle <= params.NoTurnAngleThreshold:
                params.turning_mechanism = NO_TURN

        # 3) If currently NO_TURN, check possible switch to HARD_TURN or SOFT_TURN
        if params.turning_mechanism == NO_TURN:
            if abs_angle > params.HardTurnOnAngleThreshold:
                params.turning_mechanism = HARD_TURN
            elif abs_angle > params.NoTurnAngleThreshold:
                params.turning_mechanism = SOFT_TURN
        #print('new turning mechanism: ', params.turning_mechanism)
        return params.turning_mechanism

    # --------------------------------------------------------------------
    # 4. Compute Wheel Speeds for Each Mechanism
    # --------------------------------------------------------------------
    def compute_wheel_speeds(self, turning_mechanism, c_heading_angle, params):
        """
        Given the turning mechanism (NO_TURN, SOFT_TURN, HARD_TURN),
        compute the left/right wheel linear speeds in m/s.

        :param turning_mechanism: One of (NO_TURN, SOFT_TURN, HARD_TURN)
        :param c_heading_angle: The heading angle difference to correct
        :param params: WheelTurningParams
        :return: (v_left, v_right) in m/s
        """
        abs_angle = abs(c_heading_angle)
        #print('TM, ANGLE, PARAMS: ', turning_mechanism, c_heading_angle, params)
        if turning_mechanism == NO_TURN:
            # Both wheels run at the same base speed => go straight
            fSpeed1 = params.BaseSpeed
            fSpeed2 = params.BaseSpeed

        elif turning_mechanism == HARD_TURN:
            # Turn in place: left = -MaxSpeed, right = +MaxSpeed (pivot turn)
            fSpeed1 = -params.BaseSpeed
            fSpeed2 = params.BaseSpeed

        elif turning_mechanism == SOFT_TURN:
            # Turn while moving forward
            # Reproduce the "soft turn" logic from your snippet:
            #   fSpeedFactor = (HardTurnOnAngleThreshold - abs_angle) / HardTurnOnAngleThreshold

            fSpeedFactor = (params.HardTurnOnAngleThreshold - abs_angle) / params.HardTurnOnAngleThreshold

            # For clarity:
            #   fSpeed1 = base - base * (1 - factor) = base * factor
            #   fSpeed2 = base + base * (1 - factor) = base * (2 - factor)
            # We'll treat left = fSpeed1, right = fSpeed2
            fSpeed1 = params.BaseSpeed * fSpeedFactor
            fSpeed2= params.BaseSpeed * (2.0 - fSpeedFactor)

        else:
            # Default fallback (should not happen if code is correct)
            fSpeed1 = 0.0
            fSpeed2 = 0.0
            print('something wrong. turning mech: ', turning_mechanism)
        fLeftWheelSpeed = 0
        fRightWheelSpeed = 0
        if c_heading_angle > 0:
            #Turn Left * /
            fLeftWheelSpeed  = fSpeed1
            fRightWheelSpeed = fSpeed2
        else:
            #Turn Right * /
            fLeftWheelSpeed  = fSpeed2
            fRightWheelSpeed = fSpeed1

        return fLeftWheelSpeed, fRightWheelSpeed

    # --------------------------------------------------------------------
    # 5. Differential Drive Pose Update
    # --------------------------------------------------------------------
    def update_pose(self, x, y, theta, v_left, v_right, track_width, dt):
        """
        Given the current pose (x, y, theta) and the left/right wheel speeds,
        update the robot's pose over a time step dt using the standard
        differential drive equations.

        :param x:       current x position
        :param y:       current y position
        :param theta:   current heading (radians)
        :param v_left:  left wheel linear speed (m/s)
        :param v_right: right wheel linear speed (m/s)
        :param track_width: distance between the two wheels (m)
        :param dt:      time step (s)
        :return: (x_new, y_new, theta_new)
        """

        # 1) Forward speed of the robot's center
        v = 0.5 * (v_left + v_right)

        # 2) Yaw rate
        dot_theta = (v_right - v_left) / track_width

        # 3) Update heading
        theta_new = theta + dot_theta * dt

        # 4) Update position
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt

        return x_new, y_new, theta_new

    def update_position(self, target_x, target_y, dt):
        """Update the agent's position based on movement rules."""
        target_direction = math.atan2(target_y - self.y, target_x - self.x)
        angle_diff = (target_direction - self.direction + math.pi) % (2 * math.pi) - math.pi
        #print('angle diff: ', angle_diff)
        # (A) Update turning mechanism
        self.wheel_turning_params.turning_mechanism = self.update_turning_mechanism(
            angle_diff,
            self.wheel_turning_params
        )
        # (B) Compute wheel speeds for that mechanism
        v_left, v_right = self.compute_wheel_speeds(
            self.wheel_turning_params.turning_mechanism,
            angle_diff,
            self.wheel_turning_params
        )
        # (C) Do the differential-drive pose update
        self.x, self.y, self.direction = self.update_pose(
            self.x,
            self.y, self.direction,
            v_left,
            v_right,
            self.track_width,
            dt
        )

        self.trajectory.append((self.x, self.y))

    def determine_broadcast(self, hard_turn_threshold, angle_diff):
        """Determine the broadcast value based on turn type."""
        self.broadcast = 0 if abs(angle_diff) > hard_turn_threshold else self.commitment
        self.my_opinions.append(self.broadcast)

    def receive_broadcast(self, sender_id, broadcast_value):
        """Store received broadcasts only if different from previous."""
        if self.received_broadcasts.get(sender_id, None) != broadcast_value:
            self.received_broadcasts[sender_id] = broadcast_value

    def update_commitment(self):
        """Update the agent's commitment based on received messages."""
        if random.random() < self.eta:
            self.commitment = random.choice(self.light_ids)
        else:
            if self.received_broadcasts:
                valid_commitments = [v for v in self.received_broadcasts.values() if v != 0]
                if valid_commitments:
                    self.commitment = random.choice(valid_commitments)
        self.received_broadcasts.clear()

    def log_data(self, time_step):
        """Log the agent's current state."""
        opinions = ";".join(map(str, self.my_opinions))
        self.csv_writer.writerow([time_step, self.commitment, opinions])
        self.my_opinions.clear()


class LightSource:
    """Represents a target light source in the environment."""

    def __init__(self, id, x, y, color):
        self.id = id
        self.x = x
        self.y = y
        self.color = color


class Simulation:
    """Main simulation controller."""

    def __init__(self, config):
        self.start_time = datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss%fus")
        self.config = config
        self.validate_config()
        self.initialize_from_config()
        self.agents = []
        self.history = []
        self.position_log = None

    def validate_config(self):
        """Ensure all required configuration parameters are present."""
        required_keys = ['num_agents', 'x_width', 'y_height', 'center',
                         'time_steps', 'light_sources', 'init_robot_bounds',
                         'robots_speed', 'track_width', 'eta', 'robots_direction',
                         'no_turn_threshold', 'soft_turn_threshold', 'hard_turn_threshold',
                         'steps_per_second', 'termination_radius', 'commitment_update_time']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def initialize_from_config(self):
        """Initialize simulation parameters from config."""
        # Light sources
        self.light_sources = {light['id']: LightSource(**light)
                              for light in self.config['light_sources']}
        self.light_ids = list(self.light_sources.keys())

        # Simulation parameters
        self.num_agents = self.config['num_agents']
        self.num_runs = self.config.get('num_runs', 1)
        self.base_log_dir = self.config.get('log_directory', './logs')
        self.experiment_name = self.config.get('experiment_name', 'experiment')

        # Movement thresholds
        self.thresholds = {
            'none': math.radians(self.config['no_turn_threshold']),
            'soft': math.radians(self.config['soft_turn_threshold']),
            'hard': math.radians(self.config['hard_turn_threshold'])
        }

        # Timing control
        self.steps_per_second = self.config['steps_per_second']
        self.step_duration = 1.0 / self.steps_per_second

        # Agent parameters
        self.track_width = self.config['track_width']
        self.robots_speed = self.config['robots_speed']
        self.eta = self.config['eta']

        # Arena bounds
        half_x = self.config['x_width'] / 2
        half_y = self.config['y_height'] / 2
        self.arena_bounds = {
            'x_min': self.config['center'][0] - half_x,
            'x_max': self.config['center'][0] + half_x,
            'y_min': self.config['center'][1] - half_y,
            'y_max': self.config['center'][1] + half_y
        }

    def create_run_folder(self, run_index):
        """Create a directory for the current run's logs."""
        run_folder = os.path.join(self.base_log_dir, f"run{run_index}")
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def initialize_agents(self, run_folder):
        """Initialize agents with random positions and commitments."""
        self.agents = []
        commitments = random.choices(self.light_ids, k=self.num_agents)

        for i in range(self.num_agents):
            x = random.uniform(self.config['init_robot_bounds']['x_min'],
                               self.config['init_robot_bounds']['x_max'])
            y = random.uniform(self.config['init_robot_bounds']['y_min'],
                               self.config['init_robot_bounds']['y_max'])

            agent = Agent(
                id=i, x=x, y=y,
                speed=self.robots_speed,
                track_width=self.track_width,
                direction=math.radians(self.config['robots_direction']),
                commitment=commitments[i],
                eta=self.eta,
                light_ids=self.light_ids,
                thresholds = self.thresholds,
                turning_mechanism = NO_TURN
            )
            agent.initialize_log_file(self.start_time, run_folder, self.experiment_name)
            self.agents.append(agent)

    def initialize_position_log(self, run_folder):
        """Initialize the position log file."""
        filename = os.path.join(run_folder, f"{self.experiment_name}_positions_{self.start_time}.csv")
        self.position_log = open(filename, "w", newline="")
        writer = csv.writer(self.position_log)
        writer.writerow(["Time", "ID", "x", "y"])
        self.position_writer = writer

    def log_positions_data(self, time_step):
        """Log all agents' positions for the current timestep."""
        for agent in self.agents:
            self.position_writer.writerow([time_step, agent.id, agent.x, agent.y])

    def setup_visualization(self):
        """Initialize the visualization window."""
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])

        # Plot light sources
        for light in self.light_sources.values():
            ax.scatter(light.x, light.y, color=light.color, s=100, label=f"Light {light.id}")

        # Initialize agent markers
        self.agent_markers = [
            ax.add_patch(plt.Polygon([(0, 0)], closed=True, alpha=0.7))
            for _ in range(self.num_agents)
        ]

        # Initialize trajectories
        self.trajectories = [ax.plot([], [], lw=1, alpha=0.6)[0]
                             for _ in range(self.num_agents)]

        self.time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        plt.legend()
        return fig, ax

    def update_visualization(self, ax, timestep):
        """Update the visualization elements."""
        positions = self.history[-1]

        # Update agent markers
        for marker, (x, y, commitment), agent in zip(self.agent_markers, positions, self.agents):
            color = self.light_sources[commitment].color
            size_long = 1.0
            size_short = 0.1
            direction = agent.direction

            triangle_vertices = [
                (x + size_long * math.cos(direction), y + size_long * math.sin(direction)),
                (x + size_short * math.cos(direction + 2.5 * math.pi / 3),
                 y + size_short * math.sin(direction + 2.5 * math.pi / 3)),
                (x + size_short * math.cos(direction - 2.5 * math.pi / 3),
                 y + size_short * math.sin(direction - 2.5 * math.pi / 3)),
            ]
            marker.set_xy(triangle_vertices)
            marker.set_facecolor(color)

        # Update trajectories
        for traj, agent in zip(self.trajectories, self.agents):
            x_traj, y_traj = zip(*agent.trajectory)
            traj.set_data(x_traj, y_traj)

        # Update time display
        self.time_text.set_text(f"Time: {timestep / self.steps_per_second:.1f}s")
        plt.pause(0.001)

    def check_termination(self):
        """Check if center of mass is within termination radius of any light."""
        x_com = sum(a.x for a in self.agents) / len(self.agents)
        y_com = sum(a.y for a in self.agents) / len(self.agents)
        return any(math.hypot(x_com - l.x, y_com - l.y) <= self.config['termination_radius']
                   for l in self.light_sources.values())

    def enforce_timing(self, step_start):
        """Maintain consistent timestep duration."""
        elapsed = time.perf_counter() - step_start
        sleep_time = self.step_duration - elapsed
        if sleep_time > 0 and not self.config.get('visualize', False):
            time.sleep(sleep_time)

    def cleanup(self):
        """Close all open resources."""
        # Close agent logs
        for agent in self.agents:
            agent.close_log_file()

        # Close position log
        if self.position_log:
            self.position_log.close()

        # Keep visualization open if enabled
        if self.config.get('visualize', False) and self.num_runs <= 1:
            plt.show()

    def run_experiments(self):
        """Run all configured experiment runs."""
        for run_idx in range(self.num_runs):
            run_folder = self.create_run_folder(run_idx)
            print(f"Starting run {run_idx + 1}/{self.num_runs}")
            self.run(run_folder)
            print(f"Completed run {run_idx + 1}")

    def run(self, run_folder):
        """Execute a single simulation run."""
        self.initialize_agents(run_folder)
        self.initialize_position_log(run_folder)

        if self.config.get('visualize', False):
            fig, ax = self.setup_visualization()

        try:
            for t in range(self.config['time_steps']):
                step_start = time.perf_counter()

                # Process simulation step
                self.process_timestep(t)

                # Update visualization if enabled
                if self.config.get('visualize', False):
                    self.update_visualization(ax, t)

                # Check termination condition
                if self.check_termination():
                    print(f"Termination condition met at step {t}")
                    break

                # Maintain timing
                #self.enforce_timing(step_start)

        finally:
            self.cleanup()

    def process_timestep(self, t):
        """Process a single simulation timestep."""
        # Determine broadcasts
        for agent in self.agents:
            light = self.light_sources[agent.commitment]
            target_dir = math.atan2(light.y - agent.y, light.x - agent.x)
            angle_diff = (target_dir - agent.direction + math.pi) % (2 * math.pi) - math.pi
            agent.determine_broadcast(self.thresholds['hard'], angle_diff)

        # Broadcast to neighbors
        self.broadcast_commitments()

        # Update commitments at specified interval
        if t % self.config['commitment_update_time'] == 0:
            for agent in self.agents:
                agent.update_commitment()

        # Move agents and log data
        self.move_agents()
        self.log_data(t)

    def broadcast_commitments(self):
        """Exchange broadcasts between all agents."""
        for agent in self.agents:
            for neighbor in self.agents:
                if agent.id != neighbor.id:
                    neighbor.receive_broadcast(agent.id, agent.broadcast)

    def move_agents(self):
        """Update positions of all agents."""
        positions = []
        for agent in self.agents:
            light = self.light_sources[agent.commitment]
            agent.update_position(
                light.x, light.y,
                (1/self.steps_per_second)
            )
            positions.append((agent.x, agent.y, agent.commitment))
        self.history.append(positions)

    def log_data(self, time_step):
        """Log data for all agents."""
        self.log_positions_data(time_step)
        if time_step % self.config['commitment_update_time'] == 0:
            for agent in self.agents:
                agent.log_data(time_step)


if __name__ == "__main__":
    with open("config_4_targets.json") as f:
        config = json.load(f)

    simulation = Simulation(config)
    simulation.run_experiments()