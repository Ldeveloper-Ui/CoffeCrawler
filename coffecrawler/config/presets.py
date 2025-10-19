"""
🎛️ Configuration Presets for CoffeCrawler
Dev is Here!
"""

STEALTH_MODE = {
    'user_agent_rotation': True,
    'request_delay': (2, 5),
    'max_retries': 3,
    'timeout': 30,
    'stealth_level': 'high',
    'javascript_enabled': False
}

AGGRESSIVE_MODE = {
    'user_agent_rotation': False, 
    'request_delay': (0.1, 0.5),
    'max_retries': 1,
    'timeout': 10,
    'stealth_level': 'low',
    'javascript_enabled': True
}

SMART_MODE = {
    'user_agent_rotation': True,
    'request_delay': (1, 3),
    'max_retries': 2,
    'timeout': 20,
    'stealth_level': 'medium',
    'javascript_enabled': True,
    'adaptive_delays': True
}

SAFE_MODE = {
    'user_agent_rotation': True,
    'request_delay': (3, 8),
    'max_retries': 5,
    'timeout': 45,
    'stealth_level': 'very_high',
    'javascript_enabled': False,
    'respect_robots_txt': True
}

TERMUX_OPTIMIZED = {
    'user_agent_rotation': True,
    'request_delay': (2, 4),
    'max_retries': 2,
    'timeout': 25,
    'stealth_level': 'medium',
    'javascript_enabled': False,
    'memory_efficient': True,
    'low_resource_mode': True,
    'battery_optimized': True
}
