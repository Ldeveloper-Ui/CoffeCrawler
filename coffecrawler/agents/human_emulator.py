"""
🧠 HUMAN EMULATOR - Advanced Human Behavior Simulation for CoffeCrawler
Revolutionary human-like behavior patterns with biometric simulation and adaptive learning
"""

import random
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
# import bezier
from collections import deque
import hashlib

from ..exceptions import EmulationError


@dataclass
class HumanProfile:
    """Advanced human behavior profile"""
    id: str
    behavior_type: str
    speed_variance: float
    accuracy_level: float
    patience_factor: float
    curiosity_level: float
    learning_style: str
    biometric_pattern: Dict[str, Any]


@dataclass
class MouseTrajectory:
    """Advanced mouse trajectory data"""
    points: List[Tuple[float, float]]
    timestamps: List[float]
    velocity_profile: List[float]
    acceleration_profile: List[float]
    bezier_curve: Any


class HumanEmulator:
    """
    🧠 ADVANCED HUMAN EMULATOR - Revolutionary Human Behavior Simulation
    
    Features:
    - Biometric mouse movement simulation
    - Realistic typing patterns and errors
    - Adaptive scrolling behavior
    - Gaze simulation and attention spans
    - Learning and habit formation
    - Personality-based behavior variants
    - Environmental factor simulation
    - Memory and pattern retention
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.emulator_id = self._generate_emulator_id()
        self.active_profile = self._create_human_profile()
        self.mouse_tracker = MouseTracker()
        self.attention_engine = AttentionEngine()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.biometric_simulator = BiometricSimulator()
        
        # Advanced emulation settings
        self.realism_level = 'high'  # low, medium, high, extreme
        self.learning_enabled = True
        self.adaptive_behavior = True
        self.biometric_variance = True
        
        # State tracking
        self.current_session = None
        self.behavior_history = deque(maxlen=1000)
        self.performance_metrics = EmulationMetrics()
        
        # Environmental factors
        self.fatigue_level = 0.0
        self.distraction_probability = 0.1
        self.attention_span = random.uniform(45.0, 180.0)  # seconds
        
        if crawler.debug_mode:
            print(f"🧠 Human Emulator {self.emulator_id} initialized")
            print(f"   Profile: {self.active_profile.behavior_type}")
            print(f"   Realism: {self.realism_level} | Learning: {self.learning_enabled}")
    
    def _generate_emulator_id(self) -> str:
        """Generate unique emulator ID"""
        timestamp = str(time.time())
        random_component = str(random.randint(1000, 9999))
        return hashlib.md5(f"human_{timestamp}_{random_component}".encode()).hexdigest()[:8]
    
    def _create_human_profile(self) -> HumanProfile:
        """Create advanced human behavior profile"""
        behavior_types = ['cautious', 'confident', 'curious', 'efficient', 'methodical']
        learning_styles = ['visual', 'auditory', 'kinesthetic', 'reading', 'mixed']
        
        behavior_type = random.choice(behavior_types)
        
        # Behavior parameters based on type
        behavior_params = {
            'cautious': {'speed_var': 0.3, 'accuracy': 0.9, 'patience': 0.8, 'curiosity': 0.4},
            'confident': {'speed_var': 0.6, 'accuracy': 0.7, 'patience': 0.5, 'curiosity': 0.7},
            'curious': {'speed_var': 0.5, 'accuracy': 0.6, 'patience': 0.9, 'curiosity': 0.9},
            'efficient': {'speed_var': 0.4, 'accuracy': 0.8, 'patience': 0.6, 'curiosity': 0.5},
            'methodical': {'speed_var': 0.2, 'accuracy': 0.95, 'patience': 0.7, 'curiosity': 0.3}
        }
        
        params = behavior_params[behavior_type]
        
        return HumanProfile(
            id=self._generate_emulator_id(),
            behavior_type=behavior_type,
            speed_variance=params['speed_var'],
            accuracy_level=params['accuracy'],
            patience_factor=params['patience'],
            curiosity_level=params['curiosity'],
            learning_style=random.choice(learning_styles),
            biometric_pattern=self._generate_biometric_pattern(behavior_type)
        )
    
    def _generate_biometric_pattern(self, behavior_type: str) -> Dict[str, Any]:
        """Generate biometric behavior patterns"""
        patterns = {
            'cautious': {
                'mouse_steadiness': 0.9,
                'click_pressure': 0.7,
                'reaction_time': 1.2,
                'gaze_stability': 0.8
            },
            'confident': {
                'mouse_steadiness': 0.6,
                'click_pressure': 0.9,
                'reaction_time': 0.8,
                'gaze_stability': 0.6
            },
            'curious': {
                'mouse_steadiness': 0.5,
                'click_pressure': 0.6,
                'reaction_time': 1.0,
                'gaze_stability': 0.4
            },
            'efficient': {
                'mouse_steadiness': 0.8,
                'click_pressure': 0.8,
                'reaction_time': 0.9,
                'gaze_stability': 0.7
            },
            'methodical': {
                'mouse_steadiness': 0.95,
                'click_pressure': 0.5,
                'reaction_time': 1.3,
                'gaze_stability': 0.9
            }
        }
        
        return patterns.get(behavior_type, patterns['confident'])
    
    def simulate_behavior(self, strategy: Dict):
        """
        🎯 MAIN BEHAVIOR SIMULATION - Advanced Human-like Actions
        
        Args:
            strategy: Current crawling strategy
        """
        if not self._should_simulate_behavior(strategy):
            return
        
        session_start = time.time()
        self.current_session = {
            'start_time': session_start,
            'strategy': strategy,
            'actions': [],
            'attention_level': 1.0
        }
        
        try:
            # 1. Initial attention phase
            self._simulate_initial_attention()
            
            # 2. Pre-action observation
            self._simulate_observation_phase()
            
            # 3. Main interaction sequence
            interaction_sequence = self._generate_interaction_sequence(strategy)
            
            for interaction in interaction_sequence:
                if time.time() - session_start > self.attention_span:
                    self._simulate_attention_break()
                    session_start = time.time()  # Reset attention span
                
                self._execute_interaction(interaction, strategy)
            
            # 4. Post-action reflection
            self._simulate_reflection_phase()
            
            # 5. Session cleanup and learning
            self._finalize_session()
            
        except Exception as e:
            raise EmulationError(f"Behavior simulation failed: {e}") from e
    
    def _should_simulate_behavior(self, strategy: Dict) -> bool:
        """Determine if behavior simulation should occur"""
        if strategy.get('method') in ['stealth', 'adaptive', 'intelligent']:
            return True
        
        if self.crawler.agent_type in ['stealth', 'adaptive']:
            return True
        
        return False
    
    def _simulate_initial_attention(self):
        """Simulate initial attention and focus phase"""
        focus_time = random.uniform(1.0, 3.0) * self.active_profile.patience_factor
        
        if self.crawler.debug_mode:
            print(f"   🧠 Initial focus: {focus_time:.2f}s")
        
        time.sleep(focus_time * 0.3)  # Reduced for practical purposes
    
    def _simulate_observation_phase(self):
        """Simulate observation and scanning behavior"""
        # Simulate page scanning
        scan_pattern = self._generate_scan_pattern()
        
        for scan_move in scan_pattern:
            self.mouse_tracker.record_movement(scan_move)
            time.sleep(random.uniform(0.05, 0.2))
    
    def _generate_interaction_sequence(self, strategy: Dict) -> List[Dict]:
        """Generate realistic interaction sequence"""
        sequence = []
        
        # Base interactions based on strategy
        if strategy.get('method') == 'stealth':
            sequence.extend(self._generate_stealth_interactions())
        elif strategy.get('method') == 'adaptive':
            sequence.extend(self._generate_adaptive_interactions())
        else:
            sequence.extend(self._generate_default_interactions())
        
        # Add personality-based variations
        sequence = self._apply_personality_variations(sequence)
        
        # Add random variations
        sequence = self._add_random_variations(sequence)
        
        return sequence
    
    def _generate_stealth_interactions(self) -> List[Dict]:
        """Generate stealth-optimized interactions"""
        return [
            {'type': 'scroll', 'amount': random.randint(200, 500), 'speed': 'slow'},
            {'type': 'mouse_move', 'target': 'random', 'style': 'natural'},
            {'type': 'pause', 'duration': random.uniform(1.0, 3.0)},
            {'type': 'scroll', 'amount': random.randint(100, 300), 'speed': 'medium'},
            {'type': 'mouse_move', 'target': 'content_area', 'style': 'purposeful'},
            {'type': 'pause', 'duration': random.uniform(0.5, 2.0)},
            {'type': 'scroll', 'amount': random.randint(50, 200), 'speed': 'slow'}
        ]
    
    def _generate_adaptive_interactions(self) -> List[Dict]:
        """Generate adaptive intelligent interactions"""
        return [
            {'type': 'scan_page', 'intensity': 'thorough'},
            {'type': 'mouse_move', 'target': 'interesting_element', 'style': 'curious'},
            {'type': 'scroll', 'amount': random.randint(300, 600), 'speed': 'variable'},
            {'type': 'pause', 'duration': random.uniform(0.8, 2.5)},
            {'type': 'mouse_move', 'target': 'navigation', 'style': 'exploratory'},
            {'type': 'scroll', 'amount': random.randint(200, 400), 'speed': 'medium'},
            {'type': 'read_simulation', 'duration': random.uniform(2.0, 5.0)}
        ]
    
    def _generate_default_interactions(self) -> List[Dict]:
        """Generate default interaction sequence"""
        return [
            {'type': 'scroll', 'amount': random.randint(100, 400), 'speed': 'normal'},
            {'type': 'mouse_move', 'target': 'random', 'style': 'basic'},
            {'type': 'pause', 'duration': random.uniform(0.5, 1.5)},
            {'type': 'scroll', 'amount': random.randint(50, 200), 'speed': 'normal'}
        ]
    
    def _apply_personality_variations(self, sequence: List[Dict]) -> List[Dict]:
        """Apply personality-based variations to interactions"""
        varied_sequence = []
        
        for interaction in sequence:
            varied_interaction = interaction.copy()
            
            # Apply speed variations based on profile
            if 'speed' in varied_interaction:
                speed_multiplier = self._get_speed_multiplier()
                if varied_interaction['speed'] == 'slow':
                    varied_interaction['speed'] = 'very_slow' if speed_multiplier < 0.7 else 'slow'
                elif varied_interaction['speed'] == 'fast':
                    varied_interaction['speed'] = 'very_fast' if speed_multiplier > 1.3 else 'fast'
            
            # Apply duration variations
            if 'duration' in varied_interaction:
                varied_interaction['duration'] *= random.uniform(0.8, 1.2) * self.active_profile.patience_factor
            
            varied_sequence.append(varied_interaction)
        
        return varied_sequence
    
    def _add_random_variations(self, sequence: List[Dict]) -> List[Dict]:
        """Add realistic random variations"""
        if random.random() < 0.3:  # 30% chance to add extra action
            extra_actions = [
                {'type': 'micro_pause', 'duration': random.uniform(0.1, 0.5)},
                {'type': 'mouse_wiggle', 'intensity': 'subtle'},
                {'type': 'scroll_correction', 'amount': random.randint(-50, 50)}
            ]
            insert_pos = random.randint(1, len(sequence) - 1)
            sequence.insert(insert_pos, random.choice(extra_actions))
        
        return sequence
    
    def _execute_interaction(self, interaction: Dict, strategy: Dict):
        """Execute a single interaction"""
        interaction_type = interaction['type']
        
        if self.crawler.debug_mode:
            print(f"   🖱️  Executing: {interaction_type}")
        
        try:
            if interaction_type == 'scroll':
                self._simulate_scroll(interaction)
            elif interaction_type == 'mouse_move':
                self._simulate_mouse_move(interaction)
            elif interaction_type == 'pause':
                self._simulate_pause(interaction)
            elif interaction_type == 'scan_page':
                self._simulate_page_scan(interaction)
            elif interaction_type == 'read_simulation':
                self._simulate_reading(interaction)
            elif interaction_type == 'micro_pause':
                self._simulate_micro_pause(interaction)
            elif interaction_type == 'mouse_wiggle':
                self._simulate_mouse_wiggle(interaction)
            elif interaction_type == 'scroll_correction':
                self._simulate_scroll_correction(interaction)
            
            # Record action
            if self.current_session:
                self.current_session['actions'].append({
                    'type': interaction_type,
                    'timestamp': time.time(),
                    'details': interaction
                })
            
            # Update attention level
            self._update_attention_level()
            
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Interaction failed: {e}")
    
    def _simulate_scroll(self, interaction: Dict):
        """Simulate realistic scrolling behavior"""
        amount = interaction['amount']
        speed_setting = interaction.get('speed', 'normal')
        
        # Convert speed setting to actual parameters
        speed_params = {
            'very_slow': {'duration': 2.0, 'smoothness': 0.9},
            'slow': {'duration': 1.5, 'smoothness': 0.8},
            'normal': {'duration': 1.0, 'smoothness': 0.7},
            'medium': {'duration': 0.8, 'smoothness': 0.6},
            'fast': {'duration': 0.5, 'smoothness': 0.5},
            'very_fast': {'duration': 0.3, 'smoothness': 0.4},
            'variable': {'duration': random.uniform(0.5, 1.5), 'smoothness': random.uniform(0.5, 0.8)}
        }
        
        params = speed_params.get(speed_setting, speed_params['normal'])
        duration = params['duration'] * random.uniform(0.8, 1.2)
        smoothness = params['smoothness']
        
        # Simulate scroll with realistic variations
        scroll_steps = max(3, int(amount / 100))
        step_size = amount / scroll_steps
        
        for step in range(scroll_steps):
            # Add slight variations to each step
            actual_step = step_size * random.uniform(0.8, 1.2)
            step_duration = duration / scroll_steps * random.uniform(0.7, 1.3)
            
            # Simulate the scroll step
            time.sleep(step_duration * 0.1)  # Reduced for practical purposes
            
            # Occasionally add micro-pauses
            if random.random() < 0.1:
                time.sleep(random.uniform(0.05, 0.2))
    
    def _simulate_mouse_move(self, interaction: Dict):
        """Simulate realistic mouse movement"""
        target_type = interaction.get('target', 'random')
        movement_style = interaction.get('style', 'natural')
        
        # Generate trajectory based on style and target
        trajectory = self.mouse_tracker.generate_trajectory(
            target_type=target_type,
            movement_style=movement_style,
            profile=self.active_profile
        )
        
        # Execute the trajectory
        for point in trajectory.points:
            # Simulate mouse movement to point
            movement_duration = random.uniform(0.01, 0.05)  # Reduced for practical purposes
            time.sleep(movement_duration)
            
            # Record the movement
            self.mouse_tracker.record_movement(point)
    
    def _simulate_pause(self, interaction: Dict):
        """Simulate realistic pausing behavior"""
        base_duration = interaction['duration']
        
        # Apply personality-based duration adjustment
        actual_duration = base_duration * self.active_profile.patience_factor
        
        # Add slight random variation
        actual_duration *= random.uniform(0.9, 1.1)
        
        # Simulate the pause
        time.sleep(actual_duration * 0.3)  # Reduced for practical purposes
    
    def _simulate_page_scan(self, interaction: Dict):
        """Simulate page scanning behavior"""
        intensity = interaction.get('intensity', 'normal')
        
        scan_params = {
            'quick': {'duration': 1.0, 'coverage': 0.3},
            'normal': {'duration': 2.0, 'coverage': 0.6},
            'thorough': {'duration': 3.5, 'coverage': 0.9},
            'extensive': {'duration': 5.0, 'coverage': 1.0}
        }
        
        params = scan_params.get(intensity, scan_params['normal'])
        
        # Generate scan pattern
        scan_pattern = self._generate_scan_pattern(
            duration=params['duration'],
            coverage=params['coverage']
        )
        
        # Execute scan
        for scan_point in scan_pattern:
            self.mouse_tracker.record_movement(scan_point)
            time.sleep(random.uniform(0.02, 0.08))
    
    def _simulate_reading(self, interaction: Dict):
        """Simulate reading behavior"""
        duration = interaction['duration']
        
        # Simulate reading with occasional micro-movements
        read_segments = int(duration / 0.5)  # Segment reading into 0.5s chunks
        
        for segment in range(read_segments):
            # Occasionally move mouse slightly (like real reading)
            if random.random() < 0.3:
                wiggle_trajectory = self.mouse_tracker.generate_trajectory(
                    target_type='micro_wiggle',
                    movement_style='subtle',
                    profile=self.active_profile
                )
                for point in wiggle_trajectory.points[:2]:  # Just a couple of points
                    self.mouse_tracker.record_movement(point)
                    time.sleep(0.02)
            
            # Pause for reading segment
            segment_duration = duration / read_segments * random.uniform(0.8, 1.2)
            time.sleep(segment_duration * 0.3)  # Reduced for practical purposes
    
    def _simulate_micro_pause(self, interaction: Dict):
        """Simulate micro-pause behavior"""
        duration = interaction.get('duration', random.uniform(0.1, 0.5))
        time.sleep(duration)
    
    def _simulate_mouse_wiggle(self, interaction: Dict):
        """Simulate subtle mouse wiggling"""
        intensity = interaction.get('intensity', 'subtle')
        
        wiggle_params = {
            'subtle': {'amplitude': 2, 'duration': 0.3},
            'noticeable': {'amplitude': 5, 'duration': 0.5},
            'obvious': {'amplitude': 10, 'duration': 0.8}
        }
        
        params = wiggle_params.get(intensity, wiggle_params['subtle'])
        
        # Generate wiggle pattern
        wiggle_points = self.mouse_tracker.generate_wiggle_pattern(
            amplitude=params['amplitude'],
            duration=params['duration']
        )
        
        for point in wiggle_points:
            self.mouse_tracker.record_movement(point)
            time.sleep(0.02)
    
    def _simulate_scroll_correction(self, interaction: Dict):
        """Simulate scroll correction (overscroll/underscroll)"""
        amount = interaction['amount']
        
        # Simulate correction scroll
        correction_duration = random.uniform(0.2, 0.6)
        steps = max(2, int(abs(amount) / 20))
        
        for step in range(steps):
            step_duration = correction_duration / steps
            time.sleep(step_duration * 0.1)  # Reduced for practical purposes
    
    def _simulate_attention_break(self):
        """Simulate attention break or distraction"""
        if self.crawler.debug_mode:
            print("   😴 Attention break simulated")
        
        break_duration = random.uniform(2.0, 8.0) * (1.0 - self.active_profile.patience_factor)
        
        # Simulate break activities
        break_type = random.choice(['look_away', 'stretch', 'micro_task', 'distraction'])
        
        if break_type == 'look_away':
            time.sleep(break_duration * 0.3)
        elif break_type == 'stretch':
            # Simulate stretching with mouse movements
            stretch_trajectory = self.mouse_tracker.generate_trajectory(
                target_type='stretch',
                movement_style='expansive',
                profile=self.active_profile
            )
            for point in stretch_trajectory.points:
                time.sleep(0.1)
        
        # Reset attention level after break
        if self.current_session:
            self.current_session['attention_level'] = min(1.0, 
                self.current_session['attention_level'] + 0.3)
    
    def _simulate_reflection_phase(self):
        """Simulate post-action reflection"""
        reflection_time = random.uniform(0.5, 2.0) * self.active_profile.curiosity_level
        
        if self.crawler.debug_mode:
            print(f"   🤔 Reflection: {reflection_time:.2f}s")
        
        time.sleep(reflection_time * 0.2)  # Reduced for practical purposes
    
    def _finalize_session(self):
        """Finalize behavior session and update learning"""
        if self.current_session:
            session_duration = time.time() - self.current_session['start_time']
            self.current_session['duration'] = session_duration
            
            # Record session for learning
            self.behavior_history.append(self.current_session)
            self.performance_metrics.record_session(self.current_session)
            
            # Update fatigue level
            self.fatigue_level = min(1.0, self.fatigue_level + (session_duration / 3600))
            
            # Adaptive learning
            if self.learning_enabled:
                self._update_behavior_models()
    
    def _update_attention_level(self):
        """Update attention level based on activity"""
        if not self.current_session:
            return
        
        # Gradual attention decay
        decay_rate = 0.1 * (1.0 - self.active_profile.patience_factor)
        self.current_session['attention_level'] = max(0.3, 
            self.current_session['attention_level'] - decay_rate)
        
        # Random attention spikes
        if random.random() < 0.05:
            self.current_session['attention_level'] = min(1.0,
                self.current_session['attention_level'] + 0.2)
    
    def _generate_scan_pattern(self, duration: float = 2.0, coverage: float = 0.6) -> List[Tuple[float, float]]:
        """Generate realistic page scan pattern"""
        points = []
        num_points = max(5, int(duration * 10))
        
        # Generate scan points covering the viewport
        for i in range(num_points):
            # Bias toward content areas
            if random.random() < coverage:
                x = random.gauss(0.5, 0.2)  # Center-biased
                y = random.gauss(0.4, 0.3)
            else:
                x = random.uniform(0.0, 1.0)
                y = random.uniform(0.0, 1.0)
            
            # Clamp to [0, 1] range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            points.append((x, y))
        
        return points
    
    def _get_speed_multiplier(self) -> float:
        """Get speed multiplier based on profile and state"""
        base_speed = 1.0 / (self.active_profile.speed_variance + 0.5)
        
        # Adjust for fatigue
        fatigue_impact = 1.0 - (self.fatigue_level * 0.3)
        
        # Adjust for attention
        attention_impact = self.current_session['attention_level'] if self.current_session else 1.0
        
        return base_speed * fatigue_impact * attention_impact * random.uniform(0.9, 1.1)
    
    def _update_behavior_models(self):
        """Update behavior models based on recent experiences"""
        if len(self.behavior_history) < 5:
            return
        
        recent_sessions = list(self.behavior_history)[-10:]
        
        # Analyze patterns and adjust behavior
        patterns = self.behavior_analyzer.analyze_patterns(recent_sessions)
        
        # Update biometric patterns based on success
        successful_sessions = [s for s in recent_sessions if s.get('success', True)]
        if successful_sessions:
            self._refine_biometric_patterns(successful_sessions)
    
    def _refine_biometric_patterns(self, successful_sessions: List[Dict]):
        """Refine biometric patterns based on successful sessions"""
        # Simple refinement - in production would use more sophisticated ML
        avg_attention = np.mean([s.get('attention_level', 0.5) for s in successful_sessions])
        
        # Adjust patience based on attention
        self.active_profile.patience_factor = min(0.95, 
            max(0.3, self.active_profile.patience_factor * (0.9 + avg_attention * 0.2)))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get human emulator performance statistics"""
        return self.performance_metrics.get_stats()
    
    def change_profile(self, new_profile_type: str):
        """Change to a different human profile"""
        self.active_profile = self._create_specific_profile(new_profile_type)
        
        if self.crawler.debug_mode:
            print(f"🧠 Profile changed to: {new_profile_type}")
    
    def _create_specific_profile(self, profile_type: str) -> HumanProfile:
        """Create a specific type of human profile"""
        behavior_types = ['cautious', 'confident', 'curious', 'efficient', 'methodical']
        
        if profile_type not in behavior_types:
            profile_type = random.choice(behavior_types)
        
        return self._create_human_profile_with_type(profile_type)
    
    def _create_human_profile_with_type(self, behavior_type: str) -> HumanProfile:
        """Create human profile with specific behavior type"""
        behavior_params = {
            'cautious': {'speed_var': 0.3, 'accuracy': 0.9, 'patience': 0.8, 'curiosity': 0.4},
            'confident': {'speed_var': 0.6, 'accuracy': 0.7, 'patience': 0.5, 'curiosity': 0.7},
            'curious': {'speed_var': 0.5, 'accuracy': 0.6, 'patience': 0.9, 'curiosity': 0.9},
            'efficient': {'speed_var': 0.4, 'accuracy': 0.8, 'patience': 0.6, 'curiosity': 0.5},
            'methodical': {'speed_var': 0.2, 'accuracy': 0.95, 'patience': 0.7, 'curiosity': 0.3}
        }
        
        params = behavior_params[behavior_type]
        
        return HumanProfile(
            id=self._generate_emulator_id(),
            behavior_type=behavior_type,
            speed_variance=params['speed_var'],
            accuracy_level=params['accuracy'],
            patience_factor=params['patience'],
            curiosity_level=params['curiosity'],
            learning_style=random.choice(['visual', 'auditory', 'kinesthetic', 'reading', 'mixed']),
            biometric_pattern=self._generate_biometric_pattern(behavior_type)
        )


class MouseTracker:
    """Advanced mouse movement tracking and simulation"""
    
    def __init__(self):
        self.trajectory_history = deque(maxlen=100)
        self.movement_patterns = {}
        self.last_position = (0.5, 0.5)  # Center of screen
    
    def generate_trajectory(self, target_type: str, movement_style: str, profile: HumanProfile) -> MouseTrajectory:
        """Generate realistic mouse trajectory"""
        # Determine target based on type
        if target_type == 'random':
            target = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
        elif target_type == 'content_area':
            target = (random.gauss(0.5, 0.15), random.gauss(0.4, 0.2))
        elif target_type == 'interesting_element':
            target = (random.gauss(0.6, 0.1), random.gauss(0.3, 0.1))
        elif target_type == 'navigation':
            target = (random.gauss(0.1, 0.05), random.gauss(0.1, 0.05))
        elif target_type == 'micro_wiggle':
            target = (self.last_position[0] + random.uniform(-0.02, 0.02),
                     self.last_position[1] + random.uniform(-0.02, 0.02))
        elif target_type == 'stretch':
            target = (random.uniform(0.8, 0.95), random.uniform(0.8, 0.95))
        else:
            target = (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))
        
        # Generate trajectory based on style
        if movement_style == 'natural':
            points = self._generate_natural_trajectory(self.last_position, target, profile)
        elif movement_style == 'purposeful':
            points = self._generate_purposeful_trajectory(self.last_position, target, profile)
        elif movement_style == 'curious':
            points = self._generate_curious_trajectory(self.last_position, target, profile)
        elif movement_style == 'exploratory':
            points = self._generate_exploratory_trajectory(self.last_position, target, profile)
        elif movement_style == 'subtle':
            points = self._generate_subtle_trajectory(self.last_position, target, profile)
        elif movement_style == 'expansive':
            points = self._generate_expansive_trajectory(self.last_position, target, profile)
        else:
            points = self._generate_natural_trajectory(self.last_position, target, profile)
        
        # Update last position
        if points:
            self.last_position = points[-1]
        
        # Create trajectory object
        timestamps = [i * 0.1 for i in range(len(points))]  # Simulated timestamps
        velocities = self._calculate_velocities(points, timestamps)
        accelerations = self._calculate_accelerations(velocities, timestamps)
        
        # Generate Bezier curve for smoothness
        curve = self._generate_bezier_curve(points)
        
        return MouseTrajectory(
            points=points,
            timestamps=timestamps,
            velocity_profile=velocities,
            acceleration_profile=accelerations,
            bezier_curve=curve
        )
    
    def _generate_natural_trajectory(self, start: Tuple[float, float], end: Tuple[float, float], 
                                   profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate natural human-like mouse trajectory"""
        num_points = random.randint(8, 15)
        points = [start]
        
        # Create curved path using Bezier-like interpolation
        control_point = (
            (start[0] + end[0]) / 2 + random.uniform(-0.1, 0.1),
            (start[1] + end[1]) / 2 + random.uniform(-0.1, 0.1)
        )
        
        for i in range(1, num_points - 1):
            t = i / (num_points - 1)
            
            # Quadratic Bezier interpolation
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_point[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_point[1] + t ** 2 * end[1]
            
            # Add natural jitter based on profile
            jitter_scale = 0.01 * (1.0 - profile.biometric_pattern['mouse_steadiness'])
            x += random.gauss(0, jitter_scale)
            y += random.gauss(0, jitter_scale)
            
            points.append((x, y))
        
        points.append(end)
        return points
    
    def _generate_purposeful_trajectory(self, start: Tuple[float, float], end: Tuple[float, float],
                                      profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate purposeful, direct trajectory"""
        num_points = random.randint(5, 10)  # Fewer points for direct movement
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            
            # Minimal jitter for purposeful movement
            jitter_scale = 0.005
            x += random.gauss(0, jitter_scale)
            y += random.gauss(0, jitter_scale)
            
            points.append((x, y))
        
        return points
    
    def _generate_curious_trajectory(self, start: Tuple[float, float], end: Tuple[float, float],
                                   profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate curious, exploratory trajectory"""
        num_points = random.randint(12, 20)  # More points for exploration
        points = [start]
        
        # Create meandering path
        current = start
        for i in range(num_points - 2):
            # Calculate direction to target
            dir_x = end[0] - current[0]
            dir_y = end[1] - current[1]
            
            # Add exploration bias
            explore_bias = random.uniform(-0.1, 0.1)
            dir_x += explore_bias
            dir_y += explore_bias
            
            # Normalize and scale
            dist = math.sqrt(dir_x ** 2 + dir_y ** 2)
            if dist > 0:
                dir_x /= dist
                dir_y /= dist
            
            step_size = 0.1 * random.uniform(0.8, 1.2)
            new_x = current[0] + dir_x * step_size
            new_y = current[1] + dir_y * step_size
            
            # Add significant jitter for curiosity
            jitter_scale = 0.02
            new_x += random.gauss(0, jitter_scale)
            new_y += random.gauss(0, jitter_scale)
            
            points.append((new_x, new_y))
            current = (new_x, new_y)
        
        points.append(end)
        return points
    
    def _generate_exploratory_trajectory(self, start: Tuple[float, float], end: Tuple[float, float],
                                       profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate exploratory trajectory with detours"""
        # Start with natural trajectory
        base_trajectory = self._generate_natural_trajectory(start, end, profile)
        
        # Add some detour points
        if len(base_trajectory) > 4:
            detour_index = random.randint(2, len(base_trajectory) - 3)
            detour_point = base_trajectory[detour_index]
            
            # Create detour
            detour_offset = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
            detour_point = (detour_point[0] + detour_offset[0], detour_point[1] + detour_offset[1])
            
            base_trajectory.insert(detour_index + 1, detour_point)
        
        return base_trajectory
    
    def _generate_subtle_trajectory(self, start: Tuple[float, float], end: Tuple[float, float],
                                  profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate subtle, minimal trajectory"""
        num_points = random.randint(3, 6)  # Very few points
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            
            # Very minimal jitter
            jitter_scale = 0.001
            x += random.gauss(0, jitter_scale)
            y += random.gauss(0, jitter_scale)
            
            points.append((x, y))
        
        return points
    
    def _generate_expansive_trajectory(self, start: Tuple[float, float], end: Tuple[float, float],
                                     profile: HumanProfile) -> List[Tuple[float, float]]:
        """Generate expansive, dramatic trajectory"""
        num_points = random.randint(15, 25)  # Many points for dramatic movement
        points = [start]
        
        # Create exaggerated curve
        control_point = (
            (start[0] + end[0]) / 2 + random.uniform(-0.2, 0.2),
            (start[1] + end[1]) / 2 + random.uniform(-0.2, 0.2)
        )
        
        for i in range(1, num_points - 1):
            t = i / (num_points - 1)
            
            # Quadratic Bezier with exaggeration
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_point[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_point[1] + t ** 2 * end[1]
            
            # Significant jitter for expansiveness
            jitter_scale = 0.03
            x += random.gauss(0, jitter_scale)
            y += random.gauss(0, jitter_scale)
            
            points.append((x, y))
        
        points.append(end)
        return points
    
    def generate_wiggle_pattern(self, amplitude: float, duration: float) -> List[Tuple[float, float]]:
        """Generate mouse wiggle pattern"""
        num_points = max(3, int(duration * 20))
        points = []
        
        center_x, center_y = self.last_position
        
        for i in range(num_points):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, amplitude / 100.0)  # Convert to screen ratio
            
            x = center_x + math.cos(angle) * distance
            y = center_y + math.sin(angle) * distance
            
            points.append((x, y))
        
        return points
    
    def _calculate_velocities(self, points: List[Tuple[float, float]], timestamps: List[float]) -> List[float]:
        """Calculate velocities along trajectory"""
        if len(points) < 2:
            return [0.0]
        
        velocities = [0.0]  # Initial velocity
        
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                distance = math.sqrt(dx**2 + dy**2)
                velocity = distance / dt
            else:
                velocity = 0.0
            
            velocities.append(velocity)
        
        return velocities
    
    def _calculate_accelerations(self, velocities: List[float], timestamps: List[float]) -> List[float]:
        """Calculate accelerations along trajectory"""
        if len(velocities) < 2:
            return [0.0]
        
        accelerations = [0.0]  # Initial acceleration
        
        for i in range(1, len(velocities)):
            dv = velocities[i] - velocities[i-1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                acceleration = dv / dt
            else:
                acceleration = 0.0
            
            accelerations.append(acceleration)
        
        return accelerations
    
    def _generate_bezier_curve(self, points: List[Tuple[float, float]]):
        """Generate Bezier curve for smooth trajectory"""
        if len(points) < 3:
            return None
        
        # Simple implementation - in production would use proper Bezier curves
        return f"bezier_curve_{len(points)}_points"
    
    def record_movement(self, point: Tuple[float, float]):
        """Record mouse movement"""
        self.trajectory_history.append({
            'timestamp': time.time(),
            'position': point
        })


class AttentionEngine:
    """Advanced attention and focus simulation engine"""
    
    def __init__(self):
        self.attention_model = self._initialize_attention_model()
        self.distraction_patterns = self._initialize_distraction_patterns()
    
    def _initialize_attention_model(self) -> Dict[str, Any]:
        """Initialize attention simulation model"""
        return {
            'base_attention_span': random.uniform(120.0, 300.0),  # 2-5 minutes
            'attention_decay_rate': random.uniform(0.1, 0.3),
            'focus_recovery_rate': random.uniform(0.2, 0.5),
            'distraction_susceptibility': random.uniform(0.1, 0.4)
        }
    
    def _initialize_distraction_patterns(self) -> List[Dict[str, Any]]:
        """Initialize distraction patterns"""
        return [
            {'type': 'external_notification', 'probability': 0.1, 'duration': 5.0},
            {'type': 'multitasking', 'probability': 0.2, 'duration': 10.0},
            {'type': 'environmental', 'probability': 0.15, 'duration': 3.0},
            {'type': 'cognitive_wander', 'probability': 0.3, 'duration': 8.0}
        ]
    
    def simulate_attention_cycle(self, duration: float) -> float:
        """Simulate attention cycle over duration"""
        # Simplified attention simulation
        attention_level = 1.0
        elapsed = 0.0
        
        while elapsed < duration:
            time_chunk = min(10.0, duration - elapsed)  # Process in 10s chunks
            
            # Apply attention decay
            decay = self.attention_model['attention_decay_rate'] * time_chunk / 60.0
            attention_level = max(0.3, attention_level - decay)
            
            # Check for distractions
            if self._should_distract():
                distraction = random.choice(self.distraction_patterns)
                attention_level = max(0.1, attention_level - 0.3)
                # distraction_duration would be used in real implementation
            
            elapsed += time_chunk
        
        return attention_level
    
    def _should_distract(self) -> bool:
        """Determine if distraction should occur"""
        return random.random() < self.attention_model['distraction_susceptibility']


class BehaviorAnalyzer:
    """Advanced behavior pattern analysis"""
    
    def analyze_patterns(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze behavior patterns from sessions"""
        if not sessions:
            return {}
        
        # Analyze interaction patterns
        interaction_counts = {}
        total_duration = 0.0
        
        for session in sessions:
            total_duration += session.get('duration', 0)
            
            for action in session.get('actions', []):
                action_type = action['type']
                interaction_counts[action_type] = interaction_counts.get(action_type, 0) + 1
        
        # Calculate rates
        interaction_rates = {}
        if total_duration > 0:
            for action_type, count in interaction_counts.items():
                interaction_rates[action_type] = count / total_duration
        
        return {
            'total_sessions_analyzed': len(sessions),
            'total_duration': total_duration,
            'interaction_counts': interaction_counts,
            'interaction_rates': interaction_rates,
            'avg_attention_level': np.mean([s.get('attention_level', 0.5) for s in sessions])
        }


class BiometricSimulator:
    """Advanced biometric simulation"""
    
    def __init__(self):
        self.heart_rate = random.randint(65, 75)
        self.breathing_rate = random.uniform(12.0, 16.0)
        self.cognitive_load = 0.5
    
    def update_biometrics(self, activity_intensity: float, duration: float):
        """Update biometric readings based on activity"""
        # Simulate heart rate increase
        hr_increase = activity_intensity * 20.0
        self.heart_rate = min(120, self.heart_rate + hr_increase)
        
        # Simulate breathing rate increase
        br_increase = activity_intensity * 4.0
        self.breathing_rate = min(25.0, self.breathing_rate + br_increase)
        
        # Update cognitive load
        self.cognitive_load = min(1.0, self.cognitive_load + activity_intensity * 0.3)
        
        # Gradual recovery
        recovery_rate = 0.1 * duration
        self.heart_rate = max(65, self.heart_rate - recovery_rate * 5)
        self.breathing_rate = max(12.0, self.breathing_rate - recovery_rate * 0.5)
        self.cognitive_load = max(0.3, self.cognitive_load - recovery_rate * 0.2)
    
    def get_biometric_readings(self) -> Dict[str, Any]:
        """Get current biometric readings"""
        return {
            'heart_rate': round(self.heart_rate),
            'breathing_rate': round(self.breathing_rate, 1),
            'cognitive_load': round(self.cognitive_load, 2),
            'stress_level': self._calculate_stress_level()
        }
    
    def _calculate_stress_level(self) -> float:
        """Calculate stress level from biometrics"""
        hr_stress = max(0, (self.heart_rate - 70) / 50.0)
        br_stress = max(0, (self.breathing_rate - 14) / 11.0)
        cognitive_stress = self.cognitive_load
        
        return min(1.0, (hr_stress + br_stress + cognitive_stress) / 3.0)


class EmulationMetrics:
    """Advanced emulation performance metrics"""
    
    def __init__(self):
        self.session_history = deque(maxlen=100)
        self.performance_data = {
            'total_sessions': 0,
            'total_duration': 0.0,
            'avg_attention_level': 0.0,
            'success_rate': 0.0
        }
    
    def record_session(self, session: Dict):
        """Record session metrics"""
        self.session_history.append(session)
        self.performance_data['total_sessions'] += 1
        self.performance_data['total_duration'] += session.get('duration', 0)
        
        # Update averages
        if self.session_history:
            attention_levels = [s.get('attention_level', 0.5) for s in self.session_history]
            self.performance_data['avg_attention_level'] = np.mean(attention_levels)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emulation performance statistics"""
        stats = self.performance_data.copy()
        
        if self.session_history:
            recent_sessions = list(self.session_history)[-10:]
            stats['recent_sessions'] = len(recent_sessions)
            stats['recent_avg_attention'] = np.mean([s.get('attention_level', 0.5) for s in recent_sessions])
        else:
            stats['recent_sessions'] = 0
            stats['recent_avg_attention'] = 0.0
        
        return stats


# Factory function
def create_human_emulator(crawler, realism_level: str = 'high') -> HumanEmulator:
    """Factory function to create human emulator instance"""
    emulator = HumanEmulator(crawler)
    emulator.realism_level = realism_level
    return emulator


print("🧠 Human Emulator loaded successfully - Behavioral AI Activated!")
