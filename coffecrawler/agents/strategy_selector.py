"""
🎯 STRATEGY SELECTOR - AI-Powered Adaptive Strategy Management.
Next-generation strategy selection with machine learning and real-time adaptation.
Bro I'm adaptating right now.
"""

import random
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import threading
from dataclasses import dataclass

# Import exceptions
from ..exceptions import StrategyError, ConfigurationError, AIError

class StrategyType(Enum):
    STEALTH = "stealth"
    AGGRESSIVE = "aggressive"
    SMART = "smart"
    SAFE = "safe"
    HYBRID = "hybrid"
    TERMUX_OPTIMIZED = "termux_optimized"
    PREDATOR = "predator"
    GHOST = "ghost"
    ADAPTIVE = "adaptive"
    NEURAL = "neural"

@dataclass
class StrategyPerformance:
    strategy: StrategyType
    success_rate: float
    avg_speed: float
    detection_rate: float
    resource_usage: float
    last_used: datetime
    usage_count: int

class AIPredictor:
    """
    🤖 AI Prediction Engine for strategy optimization
    """
    
    def __init__(self):
        self.learning_data = {}
        self.pattern_cache = {}
        self.prediction_model = self._init_model()
    
    def _init_model(self):
        """Initialize AI prediction model"""
        # In real implementation, this would be a ML model
        # For now, we use rule-based AI with pattern recognition
        return {
            'pattern_detector': self._pattern_detector,
            'risk_assessor': self._risk_assessor,
            'performance_predictor': self._performance_predictor
        }
    
    def _pattern_detector(self, target_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Detect patterns in target behavior"""
        patterns = {
            'anti_bot_detected': 0.0,
            'rate_limiting': 0.0,
            'javascript_challenge': 0.0,
            'captcha_present': 0.0,
            'behavior_analysis': 0.0
        }
        
        # Analyze target characteristics
        if target_analysis.get('response_headers', {}).get('server', '').lower() in ['cloudflare', 'akamai']:
            patterns['anti_bot_detected'] = 0.8
        
        if target_analysis.get('response_time', 0) > 5.0:
            patterns['rate_limiting'] = 0.6
            
        if target_analysis.get('content_type', '') == 'application/javascript':
            patterns['javascript_challenge'] = 0.7
            
        return patterns
    
    def _risk_assessor(self, patterns: Dict[str, float], environment: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        base_risk = sum(patterns.values()) / len(patterns) if patterns else 0.0
        
        # Environment modifiers
        if environment.get('is_termux', False):
            base_risk *= 0.8  # Lower risk for mobile
        
        if environment.get('network_stability', 'stable') == 'unstable':
            base_risk *= 1.2
            
        return min(1.0, base_risk)
    
    def _performance_predictor(self, strategy: StrategyType, risk_score: float) -> float:
        """Predict performance score for strategy"""
        strategy_scores = {
            StrategyType.STEALTH: max(0.7, 1.0 - risk_score * 0.5),
            StrategyType.AGGRESSIVE: max(0.3, 1.0 - risk_score * 1.5),
            StrategyType.SMART: 0.8 - risk_score * 0.3,
            StrategyType.SAFE: 0.6 + risk_score * 0.2,
            StrategyType.PREDATOR: 0.9 - risk_score * 0.8,
            StrategyType.GHOST: 0.5 + (1.0 - risk_score) * 0.4
        }
        
        return strategy_scores.get(strategy, 0.5)

class AdaptiveLearner:
    """
    🧠 Machine Learning component for continuous strategy optimization
    """
    
    def __init__(self):
        self.learning_data = {}
        self.performance_history = []
        self.adaptation_rules = self._init_adaptation_rules()
    
    def _init_adaptation_rules(self) -> Dict[str, Any]:
        """Initialize adaptation rules based on historical performance"""
        return {
            'high_detection_switch': {
                'threshold': 0.3,
                'action': 'switch_to_stealth',
                'confidence': 0.85
            },
            'slow_performance_boost': {
                'threshold': 2.0,
                'action': 'increase_aggression', 
                'confidence': 0.75
            },
            'high_success_maintain': {
                'threshold': 0.9,
                'action': 'maintain_strategy',
                'confidence': 0.95
            }
        }
    
    def record_performance(self, strategy: StrategyType, metrics: Dict[str, Any]):
        """Record strategy performance for learning"""
        learning_entry = {
            'strategy': strategy,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'success': metrics.get('success', False),
            'response_time': metrics.get('response_time', 0),
            'detection_flagged': metrics.get('detection_flagged', False)
        }
        
        self.performance_history.append(learning_entry)
        
        # Keep only last 1000 entries for memory efficiency
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_optimized_strategy(self, target_patterns: Dict[str, float]) -> StrategyType:
        """Get optimized strategy based on learned patterns"""
        if not self.performance_history:
            return StrategyType.SMART  # Default fallback
        
        # Analyze historical performance
        recent_performance = [p for p in self.performance_history 
                            if datetime.now() - p['timestamp'] < timedelta(hours=24)]
        
        if not recent_performance:
            return StrategyType.SMART
        
        # Calculate success rates for each strategy
        success_rates = {}
        for strategy in StrategyType:
            strategy_data = [p for p in recent_performance if p['strategy'] == strategy]
            if strategy_data:
                success_rate = sum(1 for p in strategy_data if p['success']) / len(strategy_data)
                success_rates[strategy] = success_rate
        
        if not success_rates:
            return StrategyType.SMART
        
        # Return strategy with highest success rate
        return max(success_rates.items(), key=lambda x: x[1])[0]

class StrategySelector:
    """
    🎯 Main Strategy Selector with AI-Powered Decision Making
    """
    
    def __init__(self, auto_learn: bool = True, ai_enabled: bool = True):
        self.auto_learn = auto_learn
        self.ai_enabled = ai_enabled
        self.current_strategy = StrategyType.SMART
        self.strategy_history = []
        self.performance_tracker = {}
        
        # Initialize AI components
        self.ai_predictor = AIPredictor()
        self.adaptive_learner = AdaptiveLearner()
        
        # Strategy configurations
        self.strategy_configs = self._init_strategy_configs()
        
        # Auto-optimization thread
        self.optimization_thread = None
        self.running = False
        
        self._start_auto_optimization()
    
    def _init_strategy_configs(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Initialize detailed strategy configurations"""
        return {
            StrategyType.STEALTH: {
                'description': 'Maximum stealth with advanced evasion',
                'risk_level': 'low',
                'success_rate': 0.85,
                'resource_usage': 'medium',
                'request_delay': (2, 8),
                'user_agent_rotation': True,
                'proxy_rotation': True,
                'javascript_execution': False,
                'fingerprint_randomization': True,
                'fallback_strategy': StrategyType.SAFE
            },
            StrategyType.AGGRESSIVE: {
                'description': 'High-speed crawling with risk tolerance',
                'risk_level': 'high',
                'success_rate': 0.95,
                'resource_usage': 'low',
                'request_delay': (0.1, 0.5),
                'user_agent_rotation': False,
                'proxy_rotation': False,
                'javascript_execution': True,
                'fingerprint_randomization': False,
                'fallback_strategy': StrategyType.SMART
            },
            StrategyType.SMART: {
                'description': 'AI-adaptive with real-time optimization',
                'risk_level': 'medium',
                'success_rate': 0.90,
                'resource_usage': 'medium',
                'request_delay': (1, 3),
                'user_agent_rotation': True,
                'proxy_rotation': True,
                'javascript_execution': True,
                'fingerprint_randomization': True,
                'fallback_strategy': StrategyType.STEALTH
            },
            StrategyType.PREDATOR: {
                'description': 'Ultra-aggressive for low-protection targets',
                'risk_level': 'very_high',
                'success_rate': 0.98,
                'resource_usage': 'very_low',
                'request_delay': (0.05, 0.2),
                'user_agent_rotation': False,
                'proxy_rotation': False,
                'javascript_execution': True,
                'fingerprint_randomization': False,
                'fallback_strategy': StrategyType.AGGRESSIVE
            },
            StrategyType.GHOST: {
                'description': 'Maximum anonymity with Tor-like behavior',
                'risk_level': 'very_low',
                'success_rate': 0.70,
                'resource_usage': 'high',
                'request_delay': (5, 15),
                'user_agent_rotation': True,
                'proxy_rotation': True,
                'javascript_execution': False,
                'fingerprint_randomization': True,
                'fallback_strategy': StrategyType.STEALTH
            },
            StrategyType.TERMUX_OPTIMIZED: {
                'description': 'Mobile-optimized for Termux environment',
                'risk_level': 'low',
                'success_rate': 0.80,
                'resource_usage': 'very_low',
                'request_delay': (2, 4),
                'user_agent_rotation': True,
                'proxy_rotation': False,
                'javascript_execution': False,
                'fingerprint_randomization': True,
                'fallback_strategy': StrategyType.SAFE
            }
        }
    
    def _start_auto_optimization(self):
        """Start background auto-optimization thread"""
        if self.auto_learn:
            self.running = True
            self.optimization_thread = threading.Thread(
                target=self._optimization_worker, 
                daemon=True
            )
            self.optimization_thread.start()
    
    def _optimization_worker(self):
        """Background worker for continuous optimization"""
        while self.running:
            try:
                self._analyze_and_optimize()
                time.sleep(300)  # Optimize every 5 minutes
            except Exception as e:
                print(f"🔧 Optimization worker error: {e}")
                time.sleep(60)
    
    def _analyze_and_optimize(self):
        """Analyze performance and optimize strategies"""
        if len(self.strategy_history) < 10:
            return  # Not enough data
        
        # Analyze recent performance
        recent_history = self.strategy_history[-50:]
        success_rates = {}
        
        for entry in recent_history:
            strategy = entry['strategy']
            success = entry.get('success', False)
            
            if strategy not in success_rates:
                success_rates[strategy] = {'success': 0, 'total': 0}
            
            success_rates[strategy]['total'] += 1
            if success:
                success_rates[strategy]['success'] += 1
        
        # Update strategy configurations based on performance
        for strategy, data in success_rates.items():
            if data['total'] >= 5:  # Minimum samples
                new_success_rate = data['success'] / data['total']
                current_config = self.strategy_configs.get(strategy, {})
                current_config['success_rate'] = new_success_rate
    
    def select_strategy(self, 
                       target_analysis: Dict[str, Any],
                       environment: Dict[str, Any],
                       use_ai: bool = True) -> StrategyType:
        """
        Select optimal strategy with AI-powered decision making
        
        Args:
            target_analysis: Analysis of target website
            environment: Current environment context
            use_ai: Whether to use AI prediction
        
        Returns:
            Optimal StrategyType
        """
        try:
            # Environment-based priority
            if environment.get('is_termux', False):
                return StrategyType.TERMUX_OPTIMIZED
            
            if not use_ai or not self.ai_enabled:
                return self._rule_based_selection(target_analysis, environment)
            
            # AI-powered selection
            return self._ai_powered_selection(target_analysis, environment)
            
        except Exception as e:
            # Fallback to safe strategy on error
            print(f"🎯 Strategy selection error, using fallback: {e}")
            return StrategyType.SAFE
    
    def _rule_based_selection(self, target_analysis: Dict, environment: Dict) -> StrategyType:
        """Rule-based strategy selection as fallback"""
        protection_level = target_analysis.get('protection_level', 'unknown')
        content_size = target_analysis.get('content_size', 0)
        response_time = target_analysis.get('response_time', 0)
        
        if protection_level in ['high', 'very_high']:
            return StrategyType.STEALTH
        elif protection_level == 'medium':
            return StrategyType.SMART
        elif response_time > 10.0:
            return StrategyType.AGGRESSIVE
        elif content_size > 1000000:  # Large content
            return StrategyType.PREDATOR
        else:
            return StrategyType.SMART
    
    def _ai_powered_selection(self, target_analysis: Dict, environment: Dict) -> StrategyType:
        """AI-powered strategy selection with pattern recognition"""
        # Get risk patterns
        patterns = self.ai_predictor.pattern_detector(target_analysis)
        risk_score = self.ai_predictor.risk_assessor(patterns, environment)
        
        # Get performance predictions for each strategy
        strategy_scores = {}
        for strategy in StrategyType:
            if strategy in self.strategy_configs:
                base_score = self.strategy_configs[strategy].get('success_rate', 0.5)
                ai_score = self.ai_predictor.performance_predictor(strategy, risk_score)
                
                # Combine base success rate with AI prediction
                final_score = (base_score * 0.6) + (ai_score * 0.4)
                strategy_scores[strategy] = final_score
        
        if not strategy_scores:
            return StrategyType.SMART
        
        # Select strategy with highest score
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Record selection for learning
        selection_record = {
            'strategy': selected_strategy,
            'target_analysis': target_analysis,
            'environment': environment,
            'risk_score': risk_score,
            'strategy_scores': strategy_scores,
            'timestamp': datetime.now()
        }
        self.strategy_history.append(selection_record)
        
        return selected_strategy
    
    def record_strategy_performance(self, 
                                  strategy: StrategyType, 
                                  success: bool, 
                                  metrics: Dict[str, Any]):
        """
        Record strategy performance for machine learning
        
        Args:
            strategy: The strategy used
            success: Whether the operation was successful
            metrics: Performance metrics dictionary
        """
        performance_record = {
            'strategy': strategy,
            'success': success,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        
        # Update performance tracker
        strategy_key = str(strategy)
        if strategy_key not in self.performance_tracker:
            self.performance_tracker[strategy_key] = {
                'total_uses': 0,
                'successful_uses': 0,
                'total_response_time': 0.0,
                'last_used': datetime.now()
            }
        
        tracker = self.performance_tracker[strategy_key]
        tracker['total_uses'] += 1
        tracker['last_used'] = datetime.now()
        
        if success:
            tracker['successful_uses'] += 1
        
        if 'response_time' in metrics:
            tracker['total_response_time'] += metrics['response_time']
        
        # Update adaptive learner
        if self.auto_learn:
            self.adaptive_learner.record_performance(strategy, {
                'success': success,
                'response_time': metrics.get('response_time', 0),
                'detection_flagged': metrics.get('detection_flagged', False)
            })
    
    def get_strategy_config(self, strategy: StrategyType) -> Dict[str, Any]:
        """Get detailed configuration for a strategy"""
        config = self.strategy_configs.get(strategy, {}).copy()
        
        # Add performance data if available
        strategy_key = str(strategy)
        if strategy_key in self.performance_tracker:
            tracker = self.performance_tracker[strategy_key]
            if tracker['total_uses'] > 0:
                config['actual_success_rate'] = tracker['successful_uses'] / tracker['total_uses']
                config['avg_response_time'] = tracker['total_response_time'] / tracker['total_uses']
                config['total_uses'] = tracker['total_uses']
        
        return config
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        total_uses = sum(t['total_uses'] for t in self.performance_tracker.values())
        
        if total_uses == 0:
            return {'total_strategies_used': 0, 'overall_success_rate': 0.0}
        
        successful_uses = sum(t['successful_uses'] for t in self.performance_tracker.values())
        overall_success_rate = successful_uses / total_uses
        
        # Strategy performance ranking
        strategy_performance = []
        for strategy_key, tracker in self.performance_tracker.items():
            if tracker['total_uses'] >= 5:  # Minimum uses for ranking
                success_rate = tracker['successful_uses'] / tracker['total_uses']
                strategy_performance.append({
                    'strategy': strategy_key,
                    'success_rate': success_rate,
                    'total_uses': tracker['total_uses'],
                    'last_used': tracker['last_used']
                })
        
        # Sort by success rate
        strategy_performance.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return {
            'total_strategies_used': len(self.performance_tracker),
            'total_operations': total_uses,
            'overall_success_rate': overall_success_rate,
            'top_performing_strategies': strategy_performance[:5],
            'ai_enabled': self.ai_enabled,
            'auto_learn_enabled': self.auto_learn
        }
    
    def emergency_fallback(self, current_strategy: StrategyType, error_type: str) -> StrategyType:
        """
        Emergency fallback strategy selection
        
        Args:
            current_strategy: The strategy that failed
            error_type: Type of error encountered
        
        Returns:
            Fallback strategy
        """
        fallback_rules = {
            'detection_blocked': StrategyType.GHOST,
            'rate_limited': StrategyType.STEALTH,
            'network_error': StrategyType.SAFE,
            'timeout': StrategyType.AGGRESSIVE,
            'parsing_error': StrategyType.SMART
        }
        
        fallback_strategy = fallback_rules.get(error_type, StrategyType.SAFE)
        
        # Get configured fallback as secondary option
        config_fallback = self.strategy_configs.get(current_strategy, {}).get('fallback_strategy')
        
        return config_fallback or fallback_strategy
    
    def auto_fix_strategy(self, strategy: StrategyType, issues: List[str]) -> StrategyType:
        """
        Automatically fix strategy based on detected issues
        
        Args:
            strategy: Current strategy
            issues: List of detected issues
        
        Returns:
            Fixed strategy
        """
        fix_mapping = {
            'high_detection_risk': StrategyType.STEALTH,
            'slow_performance': StrategyType.AGGRESSIVE,
            'resource_exhaustion': StrategyType.TERMUX_OPTIMIZED,
            'javascript_issues': StrategyType.SAFE,
            'proxy_failures': StrategyType.SMART
        }
        
        for issue in issues:
            if issue in fix_mapping:
                return fix_mapping[issue]
        
        return strategy  # No fix needed
    
    def shutdown(self):
        """Clean shutdown of strategy selector"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)

# Global strategy selector instance
_global_selector = None

def get_strategy_selector() -> StrategySelector:
    """Get global strategy selector instance"""
    global _global_selector
    if _global_selector is None:
        _global_selector = StrategySelector()
    return _global_selector

def quick_strategy_select(target_url: str, use_ai: bool = True) -> StrategyType:
    """
    Quick strategy selection for simple use cases
    
    Args:
        target_url: Target URL for analysis
        use_ai: Whether to use AI prediction
    
    Returns:
        Selected StrategyType
    """
    selector = get_strategy_selector()
    
    # Basic target analysis
    target_analysis = {
        'url': target_url,
        'protection_level': 'unknown',
        'response_time': 0,
        'content_size': 0
    }
    
    # Environment detection
    environment = {
        'is_termux': 'TERMUX_VERSION' in __import__('os').environ,
        'network_type': 'mobile' if 'TERMUX_VERSION' in __import__('os').environ else 'desktop',
        'network_stability': 'stable'
    }
    
    return selector.select_strategy(target_analysis, environment, use_ai)

def get_strategy_analytics() -> Dict[str, Any]:
    """Get strategy performance analytics"""
    selector = get_strategy_selector()
    return selector.get_performance_analytics()

# Auto-cleanup on exit
import atexit
atexit.register(lambda: get_strategy_selector().shutdown() if _global_selector else None)
