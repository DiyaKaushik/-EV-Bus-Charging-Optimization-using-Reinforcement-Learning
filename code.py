"""
COMPLETE FIXED EV SIMULATION WITH REALISTIC CHARGING BEHAVIOR
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# -----------------------------
# REALISTIC Configuration
# -----------------------------
SIM_SECONDS = 60 * 90  # 90 minutes for meaningful behavior
DT = 2
NUM_BUSES = 5  # More buses create competition
SEED = 42

BATTERY_KWH = 120.0
CONSUMPTION_KWH_PER_KM = 3.5  # Higher consumption
CRUISE_SPEED_KMH = 30.0
INIT_SOC_RANGE = (0.15, 0.25)  # Start much lower to force charging
DOCK_SOC_THRESHOLD = 0.40  # More aggressive charging threshold
CHARGE_TARGET_SOC = 0.85
DOCK_PROXIMITY_KM = 0.25  # Larger proximity range

# Create charger scarcity to force queues
CHARGER_LOCATIONS = [(0.0, 0.0), (1.0, 1.0)]
PORTS_PER_CHARGER = 1  # Only 1 port per charger to create competition
CHARGING_POWER_KW = 120.0

ROUTE_WAYPOINTS = [
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)
]

# ALL 4 STRATEGIES
STRATEGIES = ["baseline", "heuristic", "predictive", "rl"]
OUTPUT_DIR = "ai_ev_outputs_realistic"
EMA_ALPHA = 0.3

# RL Configuration
RL_STATE_SOC_BUCKETS = 5
RL_ALPHA = 0.2
RL_GAMMA = 0.95
RL_EPS_START = 0.3
RL_EPS_DECAY = 0.995
RL_MIN_EPS = 0.05

# -----------------------------
# RL Agent Implementation
# -----------------------------
class RLAgent:
    def __init__(self, num_chargers: int):
        self.num_chargers = num_chargers
        self.eps = RL_EPS_START
        self.q_table: Dict[Any, np.ndarray] = {}

    def state_from_bus(self, bus, sim):
        """Simplified state that actually works for learning"""
        # 1. SOC level (0-4 buckets)
        soc_bucket = min(4, int(bus.soc_fraction * 5))
        
        # 2. Nearest charger index (0 or 1)
        nearest_idx, nearest_dist = sim._nearest_charger(bus.position_km)
        
        # 3. Simple charger availability (0=has free ports, 1=full but short queue, 2=long queue)
        charger_status = []
        for ch in sim.chargers:
            if ch.has_free_port():
                status = 0
            elif ch.queue_length <= 1:
                status = 1  
            else:
                status = 2
            charger_status.append(status)
        
        # Simple state tuple
        return (soc_bucket, nearest_idx, charger_status[0], charger_status[1])

    def ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_chargers)

    def select_action(self, state, explore=True):
        self.ensure_state(state)
        if explore and random.random() < self.eps:
            return random.randint(0, self.num_chargers - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)
        best_next = np.max(self.q_table[next_state])
        td_target = reward + RL_GAMMA * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += RL_ALPHA * td_error

    def decay_epsilon(self):
        self.eps = max(RL_MIN_EPS, self.eps * RL_EPS_DECAY)

    def save(self, path):
        np.save(path, self.q_table, allow_pickle=True)

    def load(self, path):
        self.q_table = np.load(path, allow_pickle=True).item()

def train_rl_agent(episodes=150, episode_seconds=400):
    """RL training that actually works"""
    print(" Training RL agent (REALISTIC VERSION)...")
    agent = RLAgent(len(CHARGER_LOCATIONS))
    
    best_performance = float('inf')
    
    for episode in range(episodes):
        sim = Simulation(strategy='rl', seed=SEED + episode)
        total_reward = 0
        
        steps = episode_seconds // DT
        for step in range(steps):
            step_decisions = []
            
            for bus in sim.buses:
                # More aggressive charging - higher threshold
                if bus.soc_fraction < 0.45 and not bus.is_charging:  # Increased threshold
                    state = agent.state_from_bus(bus, sim)
                    action = agent.select_action(state, explore=True)
                    
                    step_decisions.append({
                        'bus': bus,
                        'state': state, 
                        'action': action,
                        'prev_soc': bus.soc_fraction,
                        'prev_wait_time': bus.time_waiting_at_charger_s
                    })
                    
                    # Attempt connection (will create queues due to port scarcity)
                    sim.chargers[action].connect(bus.bus_id)
            
            sim.step(DT)
            
            for decision in step_decisions:
                bus = decision['bus']
                state = decision['state']
                action = decision['action']
                
                reward = 0
                
                # STRONG REWARDS:
                if bus.is_charging:
                    reward += 30.0
                    soc_increase = bus.soc_fraction - decision['prev_soc']
                    if soc_increase > 0:
                        reward += soc_increase * 100.0
                
                elif any(bus.bus_id in ch.waiting_queue for ch in sim.chargers):
                    reward -= 25.0
                    wait_increase = bus.time_waiting_at_charger_s - decision['prev_wait_time']
                    reward -= wait_increase * 2.0
                
                if bus.soc_fraction < 0.15:
                    reward -= 40.0
                
                chosen_charger = sim.chargers[action]
                dist = math.hypot(bus.position_km[0] - chosen_charger.position_km[0],
                                 bus.position_km[1] - chosen_charger.position_km[1])
                reward -= dist * 2.0
                
                utilization = len(chosen_charger.connected_bus_ids) / chosen_charger.num_ports
                if utilization < 0.4:
                    reward += 8.0
                
                next_state = agent.state_from_bus(bus, sim)
                agent.update(state, action, reward, next_state)
                total_reward += reward
        
        current_wait = np.mean([b.time_waiting_at_charger_s for b in sim.buses])
        if current_wait < best_performance:
            best_performance = current_wait
            agent.save(os.path.join(OUTPUT_DIR, "q_table_best.npy"))
            print(f"   Episode {episode+1}: NEW BEST! Wait time: {current_wait:.1f}s")
        
        agent.decay_epsilon()
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}/{episodes}, epsilon: {agent.eps:.3f}")
    
    best_path = os.path.join(OUTPUT_DIR, "q_table_best.npy")
    if os.path.exists(best_path):
        best_agent = RLAgent(len(CHARGER_LOCATIONS))
        best_agent.load(best_path)
        print(f" Loaded BEST RL agent (wait time: {best_performance:.1f}s)")
        return best_agent
    
    return agent

# -----------------------------
# Core Simulation - FIXED FOR REALISTIC BEHAVIOR
# -----------------------------
@dataclass
class ChargingStation:
    position_km: Tuple[float, float]
    num_ports: int
    power_kw: float
    connected_bus_ids: List[int] = field(default_factory=list)
    waiting_queue: List[int] = field(default_factory=list)

    def has_free_port(self): 
        return len(self.connected_bus_ids) < self.num_ports
    
    def connect(self, bus_id: int) -> bool:
        if self.has_free_port() and bus_id not in self.connected_bus_ids:
            if bus_id in self.waiting_queue:
                self.waiting_queue.remove(bus_id)
            self.connected_bus_ids.append(bus_id)
            return True
        if bus_id not in self.waiting_queue and bus_id not in self.connected_bus_ids:
            self.waiting_queue.append(bus_id)
        return False

    def disconnect(self, bus_id: int):
        if bus_id in self.connected_bus_ids:
            self.connected_bus_ids.remove(bus_id)

    @property
    def total_load_kw(self): 
        return len(self.connected_bus_ids) * self.power_kw
    
    @property
    def queue_length(self): 
        return len(self.waiting_queue)

@dataclass
class Bus:
    bus_id: int
    battery_kwh: float
    consumption_kwh_per_km: float
    cruise_speed_kmh: float
    route_waypoints: List[Tuple[float, float]]
    start_offset_km: float = 0.0

    soc_kwh: float = field(init=False)
    route_index: int = field(default=0, init=False)
    segment_progress_km: float = field(default=0.0, init=False)
    charging_station_id: Optional[int] = field(default=None, init=False)
    is_charging: bool = field(default=False, init=False)
    time_waiting_at_charger_s: float = field(default=0.0, init=False)
    total_energy_consumed_kwh: float = field(default=0.0, init=False)
    energy_charged_kwh: float = field(default=0.0, init=False)
    distance_traveled_km: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.soc_kwh = self.battery_kwh * random.uniform(*INIT_SOC_RANGE)

    @property
    def soc_fraction(self): 
        return max(0.0, min(1.0, self.soc_kwh / self.battery_kwh))

    @property
    def position_km(self) -> Tuple[float, float]:
        a = self.route_waypoints[self.route_index]
        b = self.route_waypoints[(self.route_index + 1) % len(self.route_waypoints)]
        seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
        t = 0.0 if seg_len == 0 else self.segment_progress_km / seg_len
        return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)

    def update_drive(self, dt_s: float) -> float:
        if self.is_charging: 
            return 0.0
        
        distance_this_step_km = (self.cruise_speed_kmh / 3600.0) * dt_s
        remaining = distance_this_step_km
        traveled_this_step = 0.0
        
        while remaining > 1e-9:
            a = self.route_waypoints[self.route_index]
            b = self.route_waypoints[(self.route_index + 1) % len(self.route_waypoints)]
            seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
            seg_left = max(0.0, seg_len - self.segment_progress_km)
            step = min(seg_left, remaining)
            
            self.segment_progress_km += step
            traveled_this_step += step
            self.distance_traveled_km += step
            
            if self.segment_progress_km >= seg_len - 1e-9:
                self.route_index = (self.route_index + 1) % len(self.route_waypoints)
                self.segment_progress_km = 0.0
            
            remaining -= step
            if step <= 0: 
                break
        
        energy_used = traveled_this_step * self.consumption_kwh_per_km
        self.total_energy_consumed_kwh += energy_used
        self.soc_kwh = max(0.0, self.soc_kwh - energy_used)
        return traveled_this_step

    def start_charging(self, station_id: int):
        self.is_charging = True
        self.charging_station_id = station_id

    def stop_charging(self):
        self.is_charging = False
        self.charging_station_id = None

    def update_charge(self, dt_s: float, station_power_kw: float):
        if not self.is_charging: 
            return
        added_kwh = station_power_kw * (dt_s / 3600.0)
        self.soc_kwh = min(self.battery_kwh, self.soc_kwh + added_kwh)
        self.energy_charged_kwh += added_kwh

class Simulation:
    def __init__(self, strategy: str = "baseline", seed: Optional[int] = SEED, rl_agent: Optional[RLAgent] = None):
        self.strategy = strategy
        self.rl_agent = rl_agent
        if seed: 
            random.seed(seed)
            np.random.seed(seed)

        self.chargers = [ChargingStation(pos, PORTS_PER_CHARGER, CHARGING_POWER_KW) for pos in CHARGER_LOCATIONS]
        
        # Create buses with staggered starting positions
        loop_len = sum(math.hypot(ROUTE_WAYPOINTS[i][0]-ROUTE_WAYPOINTS[(i+1)%len(ROUTE_WAYPOINTS)][0],
                                 ROUTE_WAYPOINTS[i][1]-ROUTE_WAYPOINTS[(i+1)%len(ROUTE_WAYPOINTS)][1]) 
                      for i in range(len(ROUTE_WAYPOINTS)))
        offsets = np.linspace(0, loop_len, NUM_BUSES, endpoint=False)
        
        # Add consumption variation for more realistic behavior
        consumption_rates = [CONSUMPTION_KWH_PER_KM * random.uniform(0.9, 1.1) for _ in range(NUM_BUSES)]
        
        self.buses = [Bus(i, BATTERY_KWH, consumption_rates[i], CRUISE_SPEED_KMH, 
                         ROUTE_WAYPOINTS, float(offsets[i])) 
                     for i in range(NUM_BUSES)]

        self.t_s = 0
        self.times = []
        self.soc_hist = [[] for _ in self.buses]
        self.pos_hist = []
        self.load_hist = []
        self.queue_hist = []
        self.ema_loads = [0.0 for _ in self.chargers]
        self.alpha = EMA_ALPHA

    def _nearest_charger(self, pos: Tuple[float, float]) -> Tuple[int, float]:
        best_idx, best_d = 0, float('inf')
        for i, ch in enumerate(self.chargers):
            d = math.hypot(ch.position_km[0]-pos[0], ch.position_km[1]-pos[1])
            if d < best_d: 
                best_idx, best_d = i, d
        return best_idx, best_d

    def _choose_charger(self, bus: Bus) -> int:
        if self.strategy == "baseline":
            idx, _ = self._nearest_charger(bus.position_km)
            return idx
            
        elif self.strategy == "heuristic":
            best_score = float('inf')
            best_idx = 0
            for i, ch in enumerate(self.chargers):
                dist = math.hypot(ch.position_km[0]-bus.position_km[0], ch.position_km[1]-bus.position_km[1])
                score = ch.queue_length * 15 + dist * 2  # Higher queue penalty
                if not ch.has_free_port(): 
                    score += 25  # Higher penalty for no free ports
                if score < best_score: 
                    best_score, best_idx = score, i
            return best_idx
            
        elif self.strategy == "predictive":
            best_score = float('inf')
            best_idx = 0
            for i, ch in enumerate(self.chargers):
                dist = math.hypot(ch.position_km[0]-bus.position_km[0], ch.position_km[1]-bus.position_km[1])
                # Update EMA
                self.ema_loads[i] = self.alpha * ch.total_load_kw + (1-self.alpha) * self.ema_loads[i]
                score = self.ema_loads[i] + dist * 1.5 + ch.queue_length * 8  # Higher queue weight
                if score < best_score: 
                    best_score, best_idx = score, i
            return best_idx
            
        elif self.strategy == "rl" and self.rl_agent:
            state = self.rl_agent.state_from_bus(bus, self)
            return self.rl_agent.select_action(state, explore=False)
        else:
            idx, _ = self._nearest_charger(bus.position_km)
            return idx

    def step(self, dt_s: int):
        # 1. Update charging for connected buses
        for bus in self.buses:
            if bus.is_charging and bus.charging_station_id is not None:
                station = self.chargers[bus.charging_station_id]
                bus.update_charge(dt_s, station.power_kw)

        # 2. Update driving for non-charging buses
        for bus in self.buses:
            if not bus.is_charging:
                bus.update_drive(dt_s)

        # 3. Update waiting times for queued buses
        for ch in self.chargers:
            for bus_id in ch.waiting_queue:
                bus = next((b for b in self.buses if b.bus_id == bus_id), None)
                if bus: 
                    bus.time_waiting_at_charger_s += dt_s

        # 4. Charging decisions - MORE AGGRESSIVE
        for bus in self.buses:
            if bus.is_charging:
                # Leave charger if target SOC reached
                if bus.soc_fraction >= CHARGE_TARGET_SOC:
                    station = self.chargers[bus.charging_station_id]
                    station.disconnect(bus.bus_id)
                    bus.stop_charging()
            else:
                # MORE AGGRESSIVE: Charge if SOC below threshold OR if critically low
                if bus.soc_fraction < DOCK_SOC_THRESHOLD or bus.soc_fraction < 0.25:
                    chosen_idx = self._choose_charger(bus)
                    # Check proximity (but with larger range)
                    _, dist = self._nearest_charger(bus.position_km)
                    if dist <= DOCK_PROXIMITY_KM:
                        connected = self.chargers[chosen_idx].connect(bus.bus_id)
                        if connected:
                            bus.start_charging(chosen_idx)

        # 5. Assign queued buses to freed ports
        for ch_idx, ch in enumerate(self.chargers):
            while ch.has_free_port() and ch.waiting_queue:
                next_bus_id = ch.waiting_queue.pop(0)
                if next_bus_id not in ch.connected_bus_ids:
                    ch.connected_bus_ids.append(next_bus_id)
                bus = next((b for b in self.buses if b.bus_id == next_bus_id), None)
                if bus and not bus.is_charging:
                    bus.start_charging(ch_idx)

        # 6. Record state
        self.t_s += dt_s
        self.times.append(self.t_s)
        self.pos_hist.append([b.position_km for b in self.buses])
        for i, b in enumerate(self.buses):
            self.soc_hist[i].append(b.soc_fraction * 100.0)
        self.load_hist.append(sum(ch.total_load_kw for ch in self.chargers))
        self.queue_hist.append(sum(ch.queue_length for ch in self.chargers))

    def run(self, total_seconds: int, dt_s: int):
        steps = total_seconds // dt_s
        for _ in range(steps):
            self.step(dt_s)
        
        # Calculate final metrics
        total_queue_time = sum(ch.queue_length for ch in self.chargers) * DT  # Estimate queue time
        total_wait_time = sum(b.time_waiting_at_charger_s for b in self.buses)
        
        self.metrics = {
            'avg_wait_time_s': np.mean([b.time_waiting_at_charger_s for b in self.buses]),
            'max_wait_time_s': max([b.time_waiting_at_charger_s for b in self.buses]),
            'total_wait_time_s': total_wait_time,
            'avg_soc_percent': np.mean([np.mean(soc) for soc in self.soc_hist]),
            'min_soc_percent': min([min(soc) for soc in self.soc_hist if soc]),
            'total_energy_used_kwh': sum(b.total_energy_consumed_kwh for b in self.buses),
            'total_energy_charged_kwh': sum(b.energy_charged_kwh for b in self.buses),
            'total_queue_time_s': total_queue_time,
            'buses_that_charged': sum(1 for b in self.buses if b.energy_charged_kwh > 0),
        }
        return self.metrics

    def export_metrics_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'total_load_kw', 'total_queue'] + 
                           [f'bus_{i}_soc' for i in range(len(self.buses))] +
                           [f'bus_{i}_charging' for i in range(len(self.buses))])
            for i, t in enumerate(self.times):
                row = [t, self.load_hist[i], self.queue_hist[i] if i < len(self.queue_hist) else 0]
                row += [self.soc_hist[j][i] for j in range(len(self.buses))]
                row += [1 if self.buses[j].is_charging else 0 for j in range(len(self.buses))]
                writer.writerow(row)

# -----------------------------
# Visualization Functions - ADDED BACK
# -----------------------------
def plot_soc_history(sim: Simulation, filename: str):
    plt.figure(figsize=(10, 5))
    for i in range(len(sim.buses)):
        plt.plot(sim.times, sim.soc_hist[i], label=f'Bus {i}', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('SOC (%)')
    plt.title(f'SOC History - {sim.strategy} Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Saved SOC plot: {filename}")

def plot_load_history(sim: Simulation, filename: str):
    plt.figure(figsize=(10, 4))
    plt.plot(sim.times, sim.load_hist, 'r-', linewidth=2, label='Total Load')
    plt.xlabel('Time (s)')
    plt.ylabel('Load (kW)')
    plt.title(f'Charger Load - {sim.strategy} Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Saved load plot: {filename}")

def create_animation(sim: Simulation, gif_path: str, fps: int = 5):
    """Create a simple animation showing bus movements"""
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    if len(sim.pos_hist) == 0:
        print("  No position data for animation")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot route
    route_x, route_y = zip(*ROUTE_WAYPOINTS)
    ax.plot(route_x + (route_x[0],), route_y + (route_y[0],), 'k--', alpha=0.5, label='Route')

    # Plot chargers
    charger_x, charger_y = zip(*CHARGER_LOCATIONS)
    ax.scatter(charger_x, charger_y, s=200, c='orange', marker='s',
               edgecolors='black', label='Chargers')

    # Create initial positions
    initial_positions = sim.pos_hist[0]
    init_x = [p[0] for p in initial_positions]
    init_y = [p[1] for p in initial_positions]

    colors = plt.cm.Set1(np.linspace(0, 1, len(sim.buses)))
    
    bus_scatter = ax.scatter(init_x, init_y, s=100, c=colors, edgecolors='black')

    # Text overlay
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'EV Simulation - {sim.strategy} Strategy')
    ax.legend()

    # Animation function
    def animate(frame):
        if frame >= len(sim.pos_hist):
            return bus_scatter, time_text

        positions = sim.pos_hist[frame]
        x_pos = [p[0] for p in positions]
        y_pos = [p[1] for p in positions]

        bus_scatter.set_offsets(np.column_stack([x_pos, y_pos]))

        soc_info = " | ".join([
            f"B{i}:{sim.soc_hist[i][frame]:.0f}%"
            for i in range(len(sim.buses))
        ])
        time_text.set_text(f'Time: {sim.times[frame]}s\n{soc_info}')

        return bus_scatter, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate,
        frames=min(100, len(sim.pos_hist)),
        interval=1000 // fps,
        blit=False
    )

    # Save GIF
    try:
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
        print(f"[OK] Saved animation: {gif_path}")
    except Exception as e:
        print(f"[ERROR] Animation failed: {e}")
    finally:
        plt.close()

# -----------------------------
# Main Function
# -----------------------------
def main():
    print("EV Bus Charging Optimization Simulation - REALISTIC")
    print("===================================================")
    
    # Create output directories
    base_dir = OUTPUT_DIR
    stat_dir = os.path.join(base_dir, "statistical_analysis")
    demo_dir = os.path.join(base_dir, "strategy_demonstrations")
    comp_dir = os.path.join(base_dir, "comparisons")
    
    for d in [base_dir, stat_dir, demo_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Train RL agent once
    print("\n Training RL agent...")
    rl_agent = train_rl_agent(episodes=150, episode_seconds=400)
    
    # Test ALL 4 strategies with individual demonstrations
    print("\n Creating individual strategy demonstrations...")
    all_metrics = {}
    
    for strategy in STRATEGIES:
        print(f"\n--- Testing {strategy} strategy ---")
        
        # Create individual simulation
        sim = Simulation(strategy=strategy, seed=SEED, 
                        rl_agent=rl_agent if strategy == 'rl' else None)
        
        # Run simulation
        metrics = sim.run(SIM_SECONDS, DT)
        all_metrics[strategy] = metrics
        
        # Save INDIVIDUAL outputs
        strategy_dir = os.path.join(demo_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # CSV metrics
        csv_path = os.path.join(strategy_dir, f"metrics_{strategy}.csv")
        sim.export_metrics_csv(csv_path)
        
        # SOC plot
        soc_path = os.path.join(strategy_dir, f"soc_{strategy}.png")
        plot_soc_history(sim, soc_path)
        
        # Load plot  
        load_path = os.path.join(strategy_dir, f"load_{strategy}.png")
        plot_load_history(sim, load_path)
        
        # Animation
        anim_path = os.path.join(strategy_dir, f"animation_{strategy}.gif")
        create_animation(sim, anim_path)
        
        print(f" {strategy}: Wait {metrics['avg_wait_time_s']:.1f}s, SOC {metrics['avg_soc_percent']:.1f}%")
        print(f"   Max wait: {metrics['max_wait_time_s']:.1f}s, Buses charged: {metrics['buses_that_charged']}")
    
    # Create strategy comparison
    print("\n Creating strategy comparison...")
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Wait times comparison
    plt.subplot(2, 3, 1)
    wait_times = [all_metrics[s]['avg_wait_time_s'] for s in STRATEGIES]
    plt.bar(STRATEGIES, wait_times, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Average Wait Time (s)')
    plt.title('Average Wait Time')
    plt.xticks(rotation=45)
    
    # Plot 2: Max wait times
    plt.subplot(2, 3, 2)
    max_waits = [all_metrics[s]['max_wait_time_s'] for s in STRATEGIES]
    plt.bar(STRATEGIES, max_waits, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Max Wait Time (s)')
    plt.title('Maximum Wait Time')
    plt.xticks(rotation=45)
    
    # Plot 3: SOC comparison
    plt.subplot(2, 3, 3)
    soc_values = [all_metrics[s]['avg_soc_percent'] for s in STRATEGIES]
    plt.bar(STRATEGIES, soc_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Average SOC (%)')
    plt.title('Average SOC Level')
    plt.xticks(rotation=45)
    
    # Plot 4: Energy usage
    plt.subplot(2, 3, 4)
    energy_used = [all_metrics[s]['total_energy_used_kwh'] for s in STRATEGIES]
    plt.bar(STRATEGIES, energy_used, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Energy Used (kWh)')
    plt.title('Total Energy Consumption')
    plt.xticks(rotation=45)
    
    # Plot 5: Energy charged
    plt.subplot(2, 3, 5)
    energy_charged = [all_metrics[s]['total_energy_charged_kwh'] for s in STRATEGIES]
    plt.bar(STRATEGIES, energy_charged, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Energy Charged (kWh)')
    plt.title('Total Energy Charged')
    plt.xticks(rotation=45)
    
    # Plot 6: Buses that charged
    plt.subplot(2, 3, 6)
    buses_charged = [all_metrics[s]['buses_that_charged'] for s in STRATEGIES]
    plt.bar(STRATEGIES, buses_charged, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.ylabel('Buses That Charged')
    plt.title('Successful Charging Events')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    comparison_path = os.path.join(comp_dir, "strategy_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n REALISTIC SIMULATION COMPLETE!")
    print(f" Outputs saved to: {OUTPUT_DIR}")
    print(f" Individual strategies: {demo_dir}")
    print(f" Comparisons: {comp_dir}")
    print(f"\n Strategy performance summary:")
    for strategy in STRATEGIES:
        m = all_metrics[strategy]
        print(f"   {strategy:12} | Avg Wait: {m['avg_wait_time_s']:5.1f}s | "
              f"Max Wait: {m['max_wait_time_s']:5.1f}s | SOC: {m['avg_soc_percent']:5.1f}% | "
              f"Charged: {m['buses_that_charged']}/{NUM_BUSES}")

if __name__ == "__main__":
    main()