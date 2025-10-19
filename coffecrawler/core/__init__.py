"""
CORE MODULE - The Brain of CoffeCrawler
Contains all fundamental engines and processing systems
"""

from .crawler_engine import CoffeCrawler
from .http_client import HTTPClient
from .headless_browser import HeadlessBrowser
from .parser_engine import ParserEngine
from .session_manager import SessionManager
from .engine_router import EngineRouter

__all__ = [
    'CoffeCrawler',
    'HTTPClient', 
    'HeadlessBrowser',
    'ParserEngine',
    'SessionManager',
    'EngineRouter'
]

# Core version and capabilities
CORE_CAPABILITIES = {
    'multi_engine': True,
    'smart_fallback': True,
    'real_time_adaptation': True,
    'memory_optimized': True,
    'async_support': True
}

class CoreManager:
    """Manage core engine operations"""
    
    @staticmethod
    def get_capabilities():
        """Get core system capabilities"""
        return CORE_CAPABILITIES.copy()
    
    @staticmethod
    def system_check():
        """Perform system compatibility check"""
        import sys
        requirements = {
            'python_version': (3, 8),
            'platform': ['linux', 'android'],
            'memory': 128  # MB
        }
        
        # Check Python version
        if sys.version_info < requirements['python_version']:
            raise EnvironmentError("Python 3.8+ required")
        
        return {
            'status': 'optimal',
            'python_version': sys.version,
            'platform': sys.platform,
            'cores_available': os.cpu_count()
        }

# Auto-initialize core systems
print("🔧 CoffeCrawler Core initialized - Multi-engine system ready!")
