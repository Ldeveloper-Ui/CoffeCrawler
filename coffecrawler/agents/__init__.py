"""
🤖 AI AGENTS MODULE - Advanced Bot Intelligence Systems
Next-generation AI agents with adaptive strategies and human emulation.
Dev is here Guys
"""

from .bot_agent import BotAgent
from .human_emulator import HumanEmulator
from .rotation_manager import RotationManager
from .strategy_selector import (
    StrategySelector, 
    StrategyType, 
    get_strategy_selector, 
    quick_strategy_select, 
    get_strategy_analytics,
    AIPredictor,
    AdaptiveLearner
)

__all__ = [
    # Main Agent Classes
    'BotAgent',
    'HumanEmulator', 
    'RotationManager',
    'StrategySelector',
    
    # Strategy Types
    'StrategyType',
    
    # AI Components
    'AIPredictor',
    'AdaptiveLearner',
    
    # Utility Functions
    'get_strategy_selector',
    'quick_strategy_select', 
    'get_strategy_analytics'
]

# Agent types and behaviors with enhanced strategies
AGENT_PROFILES = {
    'stealth': {
        'behavior': 'cautious',
        'speed': 'moderate',
        'detection_avoidance': 'high',
        'human_emulation': True,
        'recommended_strategy': StrategyType.STEALTH
    },
    'aggressive': {
        'behavior': 'bold',
        'speed': 'fast',
        'detection_avoidance': 'low',
        'extraction_power': 'max',
        'recommended_strategy': StrategyType.AGGRESSIVE
    },
    'adaptive': {
        'behavior': 'intelligent',
        'speed': 'variable',
        'detection_avoidance': 'adaptive',
        'self_learning': True,
        'recommended_strategy': StrategyType.SMART
    },
    'intelligent': {
        'behavior': 'ai_enhanced',
        'speed': 'optimized',
        'detection_avoidance': 'ai_powered',
        'predictive_analysis': True,
        'recommended_strategy': StrategyType.NEURAL
    },
    'predator': {
        'behavior': 'ultra_aggressive',
        'speed': 'maximum',
        'detection_avoidance': 'minimal',
        'resource_usage': 'low',
        'recommended_strategy': StrategyType.PREDATOR
    },
    'ghost': {
        'behavior': 'ultra_stealth',
        'speed': 'slow',
        'detection_avoidance': 'maximum',
        'anonymity': 'ultimate',
        'recommended_strategy': StrategyType.GHOST
    },
    'termux_optimized': {
        'behavior': 'mobile_optimized',
        'speed': 'balanced',
        'detection_avoidance': 'medium',
        'resource_efficient': True,
        'recommended_strategy': StrategyType.TERMUX_OPTIMIZED
    }
}

class AgentFactory:
    """Factory for creating different agent types with AI strategy integration"""

    @staticmethod
    def create_agent(agent_type, config=None):
        """Create agent instance based on type with AI strategy"""
        from .bot_agent import BotAgent

        agent_config = AGENT_PROFILES.get(agent_type, AGENT_PROFILES['adaptive'])
        if config:
            agent_config.update(config)

        # Auto-select strategy based on agent type
        if 'recommended_strategy' in agent_config:
            strategy = agent_config['recommended_strategy']
        else:
            strategy = quick_strategy_select("default")
            
        agent_config['default_strategy'] = strategy
        
        return BotAgent(agent_type, agent_config)

    @staticmethod
    def create_ai_agent(target_url=None, environment=None):
        """Create AI-optimized agent with automatic strategy selection"""
        from .bot_agent import BotAgent
        
        if target_url:
            strategy = quick_strategy_select(target_url, use_ai=True)
        else:
            strategy = StrategyType.SMART
            
        agent_config = {
            'behavior': 'ai_adaptive',
            'strategy': strategy,
            'auto_learn': True,
            'ai_enhanced': True
        }
        
        return BotAgent('intelligent', agent_config)

    @staticmethod
    def get_available_agents():
        """Get list of all available agent types"""
        return list(AGENT_PROFILES.keys())
    
    @staticmethod
    def get_agent_strategy(agent_type, target_analysis=None):
        """Get recommended strategy for agent type"""
        agent_profile = AGENT_PROFILES.get(agent_type, {})
        default_strategy = agent_profile.get('recommended_strategy', StrategyType.SMART)
        
        if target_analysis:
            # Use AI to refine strategy based on target
            return quick_strategy_select(
                target_analysis.get('url', 'default'), 
                use_ai=True
            )
        return default_strategy

# AI-Powered features with enhanced capabilities
AI_CAPABILITIES = {
    'pattern_recognition': True,
    'behavioral_analysis': True,
    'predictive_scraping': True,
    'auto_strategy_adjustment': True,
    'error_learning': True,
    'risk_assessment': True,
    'performance_optimization': True,
    'adaptive_learning': True,
    'real_time_analytics': True
}

# Strategy performance tracking
STRATEGY_PERFORMANCE = {
    'total_operations': 0,
    'successful_operations': 0,
    'strategy_effectiveness': {},
    'learning_cycles': 0
}

def get_global_strategy_selector() -> StrategySelector:
    """Get the global strategy selector instance"""
    return get_strategy_selector()

def analyze_agent_performance() -> dict:
    """Get comprehensive performance analytics for all agents"""
    selector = get_strategy_selector()
    strategy_analytics = selector.get_performance_analytics()
    
    return {
        'agent_profiles_count': len(AGENT_PROFILES),
        'ai_capabilities': AI_CAPABILITIES,
        'strategy_performance': strategy_analytics,
        'total_operations': STRATEGY_PERFORMANCE['total_operations'],
        'success_rate': (
            STRATEGY_PERFORMANCE['successful_operations'] / 
            STRATEGY_PERFORMANCE['total_operations'] 
            if STRATEGY_PERFORMANCE['total_operations'] > 0 else 0
        )
    }

def record_operation(success: bool, strategy_used: StrategyType = None):
    """Record operation for performance tracking"""
    STRATEGY_PERFORMANCE['total_operations'] += 1
    if success:
        STRATEGY_PERFORMANCE['successful_operations'] += 1
        
    if strategy_used:
        strategy_key = str(strategy_used)
        if strategy_key not in STRATEGY_PERFORMANCE['strategy_effectiveness']:
            STRATEGY_PERFORMANCE['strategy_effectiveness'][strategy_key] = {
                'uses': 0,
                'successes': 0
            }
        
        STRATEGY_PERFORMANCE['strategy_effectiveness'][strategy_key]['uses'] += 1
        if success:
            STRATEGY_PERFORMANCE['strategy_effectiveness'][strategy_key]['successes'] += 1

# Auto-initialize global strategy selector on import
_selector = get_strategy_selector()
STRATEGY_PERFORMANCE['learning_cycles'] = 1

print("🤖 Bot Agent loaded successfully - AI Systems Activated!")
print("🧠 Human Emulator loaded successfully - Behavioral AI Activated!") 
print("🔄 Advanced Rotation Manager loaded successfully - Active Defense Systems Online!")
print("🎯 Strategy Selector loaded successfully - Tactical AI Engaged!")
print(f"📊 {len(AGENT_PROFILES)} Agent Profiles Available")
print("🚀 AI-Powered Agent Factory Ready!")
