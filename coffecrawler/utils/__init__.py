"""
UTILS MODULE - Powerful Utilities & Helpers
Smart tools for data extraction, debugging, and optimization.
Ldeveloper Have 0 Utils 😭
"""

from .data_extractor import DataExtractor
from .debug_fixer import DebugFixer
from .anti_detection import AntiDetection
from .cache_manager import CacheManager
from .performance_optimizer import PerformanceOptimizer
from .export_manager import ExportManager
from .mobile_emulator import MobileEmulator

__all__ = [
    'DataExtractor',
    'DebugFixer',
    'AntiDetection',
    'CacheManager',
    'PerformanceOptimizer',
    'ExportManager',
    'MobileEmulator'
]

# Utility configurations
UTILITY_CONFIG = {
    'auto_cache': True,
    'smart_retry': True,
    'memory_monitoring': True,
    'export_formats': ['json', 'csv', 'xml', 'sqlite'],
    'compression': True
}

class UtilityManager:
    """Central manager for all utility functions"""
    
    @staticmethod
    def enable_advanced_utils():
        """Enable all advanced utilities"""
        UTILITY_CONFIG.update({
            'neural_parsing': True,
            'ai_enhanced_extraction': True,
            'predictive_caching': True,
            'adaptive_compression': True
        })
        return "Advanced utilities activated! 🚀"
    
    @staticmethod
    def get_system_info():
        """Get detailed system information"""
        import psutil
        import platform
        
        return {
            'system': platform.system(),
            'memory_available': f"{psutil.virtual_memory().available // (1024**2)} MB",
            'cpu_usage': f"{psutil.cpu_percent()}%",
            'disk_usage': f"{psutil.disk_usage('/').percent}%",
            'optimization_level': 'high'
        }
    
    @staticmethod
    def quick_setup(profile='default'):
        """Quick setup utility profiles"""
        profiles = {
            'default': {
                'cache_size': '100MB',
                'retry_attempts': 3,
                'timeout': 30
            },
            'power': {
                'cache_size': '500MB',
                'retry_attempts': 5,
                'timeout': 60,
                'parallel_processing': True
            },
            'termux': {
                'cache_size': '50MB',
                'retry_attempts': 2,
                'timeout': 45,
                'low_memory_mode': True
            }
        }
        
        config = profiles.get(profile, profiles['default'])
        UTILITY_CONFIG.update(config)
        return f"Profile '{profile}' applied! ✅"

# Export convenience functions
def enable_fixer_mode(aggressiveness='medium'):
    """Enable the revolutionary auto-fix mode"""
    from .debug_fixer import DebugFixer
    fixer = DebugFixer()
    return fixer.activate(aggressiveness)

def export_data(data, format='json', filename=None):
    """Quick data export utility"""
    from .export_manager import ExportManager
    exporter = ExportManager()
    return exporter.export(data, format, filename)

def mobile_emulate(device='android_type_5'):
    """Enable mobile device emulation"""
    from .mobile_emulator import MobileEmulator
    emulator = MobileEmulator()
    return emulator.set_device(device)

print("🔧 Advanced utilities loaded - Ready for action!")
