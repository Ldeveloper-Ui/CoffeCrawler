"""
☕ COFFEECRAWLER - Next Generation Web Crawling Library
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Version: 1.0.1 | Author: Ldeveloper (Termux Ambitious Developer)
Description: Revolutionary AI-powered web crawling with modular architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import importlib
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# =============================================================================
# 🎨 ASCII COLOR SYSTEM - TERMUX COMPATIBLE
# =============================================================================

class Colors:
    """ANSI Color codes for Termux compatibility"""
    # Basic Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'
    
    # Backgrounds
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    BG_PURPLE = '\033[45m'
    
    # Reset
    RESET = '\033[0m'
    
    # Custom Colors
    ORANGE = '\033[38;5;214m'
    PINK = '\033[38;5;205m'
    GOLD = '\033[38;5;220m'
    ELECTRIC_BLUE = '\033[38;5;45m'
    NEON_GREEN = '\033[38;5;46m'

# Color shortcuts
C = Colors

# =============================================================================
# 🎪 ASCII ART & BANNER SYSTEM
# =============================================================================

class Banner:
    """Epic ASCII banner system"""
    
    @staticmethod
    def get_main_banner():
        """Main CoffeCrawler banner"""
        return f"""
{C.ELECTRIC_BLUE}{C.BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  {C.GOLD}▓▓▓▓▓▓▓▓▓▓  {C.PURPLE}▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓▓▓▓▓▓▓  {C.PURPLE}▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓          {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓          {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓▓▓▓▓▓▓  {C.PURPLE}▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓  {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓▓▓▓▓▓▓  {C.PURPLE}▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓  {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓          {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓          {C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║  {C.GOLD}▓▓▓▓      {C.PURPLE}▓▓▓▓    ▓▓▓▓{C.ELECTRIC_BLUE}  {C.NEON_GREEN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{C.ELECTRIC_BLUE}  {C.PINK}▓▓▓▓     ▓▓▓▓{C.ELECTRIC_BLUE}  ║
    ║                                                              ║
    ║    {C.CYAN}{C.BOLD}☕ COFFEECRAWLER v1.0.1 - Next Generation Web Crawling{C.ELECTRIC_BLUE}     ║
    ║    {C.YELLOW}🤖 AI-Powered | 🕷️ Smart Parsing | 🔧 Auto-Fix Mode{C.ELECTRIC_BLUE}           ║
    ║    {C.GREEN}🎯 Adaptive AI | 📱 Termux Optimized | 🚀 Blazing Fast{C.ELECTRIC_BLUE}         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{C.RESET}
"""

    @staticmethod
    def get_module_icon(module_name):
        """Get icon for each module"""
        icons = {
            'core': '🔧',
            'agents': '🤖', 
            'utils': '🛠️',
            'plugins': '🔌',
            'config': '⚙️',
            'exceptions': '🚨'
        }
        return icons.get(module_name, '📁')

    @staticmethod
    def get_status_color(status, text):
        """Get colored status text"""
        colors = {
            'loading': C.YELLOW,
            'success': C.GREEN,
            'error': C.RED,
            'warning': C.ORANGE,
            'info': C.CYAN,
            'active': C.NEON_GREEN,
            'inactive': C.RED
        }
        color = colors.get(status, C.WHITE)
        return f"{color}{text}{C.RESET}"

# =============================================================================
# 🏷️ PACKAGE METADATA
# =============================================================================

__version__ = "1.0.1"
__author__ = "Ldeveloper"
__email__ = "vlskthegamer@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024, CoffeCrawler Development Team"

# =============================================================================
# 🔧 ADVANCED LAZY IMPORT SYSTEM WITH PROGRESS TRACKING
# =============================================================================

class _AdvancedLazyLoader:
    """Advanced lazy loader dengan progress tracking dan color coding"""
    
    def __init__(self):
        self._cache = {}
        self._loading_stats = {
            'total': 0,
            'loaded': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Detailed module mapping dengan prioritas
        self._module_map = {
            # 🎯 CORE MODULES - High Priority
            'CoffeCrawler': ('coffecrawler.core.crawler_engine', 'CoffeCrawler', 'core'),
            'HTTPClient': ('coffecrawler.core.http_client', 'HTTPClient', 'core'),
            'HeadlessBrowser': ('coffecrawler.core.headless_browser', 'HeadlessBrowser', 'core'),
            'ParserEngine': ('coffecrawler.core.parser_engine', 'ParserEngine', 'core'),
            
            # 🤖 AI AGENTS - High Priority  
            'BotAgent': ('coffecrawler.agents.bot_agent', 'BotAgent', 'agents'),
            'HumanEmulator': ('coffecrawler.agents.human_emulator', 'HumanEmulator', 'agents'),
            'RotationManager': ('coffecrawler.agents.rotation_manager', 'RotationManager', 'agents'),
            'StrategySelector': ('coffecrawler.agents.strategy_selector', 'StrategySelector', 'agents'),
            'StrategyType': ('coffecrawler.agents.strategy_selector', 'StrategyType', 'agents'),
            
            # 🎯 STRATEGY SYSTEM - High Priority
            'get_strategy_selector': ('coffecrawler.agents.strategy_selector', 'get_strategy_selector', 'agents'),
            'quick_strategy_select': ('coffecrawler.agents.strategy_selector', 'quick_strategy_select', 'agents'),
            'get_strategy_analytics': ('coffecrawler.agents.strategy_selector', 'get_strategy_analytics', 'agents'),
            
            # 🛠️ UTILITIES - Medium Priority
            'DataExtractor': ('coffecrawler.utils.data_extractor', 'DataExtractor', 'utils'),
            'DebugFixer': ('coffecrawler.utils.debug_fixer', 'DebugFixer', 'utils'),
            'AntiDetection': ('coffecrawler.utils.anti_detection', 'AntiDetection', 'utils'),
            'CacheManager': ('coffecrawler.utils.cache_manager', 'CacheManager', 'utils'),
            
            # 🔧 HELPER FUNCTIONS - Medium Priority
            'enable_debug_mode': ('coffecrawler.utils.helpers', 'enable_debug_mode', 'utils'),
            'enable_fixer_mode': ('coffecrawler.utils.helpers', 'enable_fixer_mode', 'utils'),
            'set_mobile_emulation': ('coffecrawler.utils.helpers', 'set_mobile_emulation', 'utils'),
            'export_data': ('coffecrawler.utils.helpers', 'export_data', 'utils'),
            'quick_crawl': ('coffecrawler.utils.helpers', 'quick_crawl', 'utils'),
            
            # 🔌 BROWSER PLUGINS - Low Priority
            'Mozilla': ('coffecrawler.plugins.browser_plugins', 'Mozilla', 'plugins'),
            'Chrome': ('coffecrawler.plugins.browser_plugins', 'Chrome', 'plugins'),
            'AndroidBrowser': ('coffecrawler.plugins.browser_plugins', 'AndroidBrowser', 'plugins'),
            'Safari': ('coffecrawler.plugins.browser_plugins', 'Safari', 'plugins'),
            'Edge': ('coffecrawler.plugins.browser_plugins', 'Edge', 'plugins'),
            
            # ⚙️ CONFIG PRESETS - Low Priority
            'STEALTH_MODE': ('coffecrawler.config.presets', 'STEALTH_MODE', 'config'),
            'AGGRESSIVE_MODE': ('coffecrawler.config.presets', 'AGGRESSIVE_MODE', 'config'),
            'SMART_MODE': ('coffecrawler.config.presets', 'SMART_MODE', 'config'),
            'SAFE_MODE': ('coffecrawler.config.presets', 'SAFE_MODE', 'config'),
            'TERMUX_OPTIMIZED': ('coffecrawler.config.presets', 'TERMUX_OPTIMIZED', 'config'),
            
            # 🚨 EXCEPTIONS - High Priority
            'CoffeCrawlerError': ('coffecrawler.exceptions', 'CoffeCrawlerError', 'exceptions'),
            'CrawlerBlockedError': ('coffecrawler.exceptions', 'CrawlerBlockedError', 'exceptions'),
            'ParserError': ('coffecrawler.exceptions', 'ParserError', 'exceptions'),
            'NetworkError': ('coffecrawler.exceptions', 'NetworkError', 'exceptions'),
            'FixerModeActivated': ('coffecrawler.exceptions', 'FixerModeActivated', 'exceptions'),
            'AIError': ('coffecrawler.exceptions', 'AIError', 'exceptions'),
            'StrategyError': ('coffecrawler.exceptions', 'StrategyError', 'exceptions'),
            'ConfigurationError': ('coffecrawler.exceptions', 'ConfigurationError', 'exceptions'),
            'TermuxCompatibilityError': ('coffecrawler.exceptions', 'TermuxCompatibilityError', 'exceptions'),
            'RotationError': ('coffecrawler.exceptions', 'RotationError', 'exceptions'),
            'ProxyError': ('coffecrawler.exceptions', 'ProxyError', 'exceptions'),
            'IdentityError': ('coffecrawler.exceptions', 'IdentityError', 'exceptions'),
            'SecurityError': ('coffecrawler.exceptions', 'SecurityError', 'exceptions'),
            'EmulationError': ('coffecrawler.exceptions', 'EmulationError', 'exceptions'),
        }
        
        self._loading_stats['total'] = len(self._module_map)
        self._loading_stats['start_time'] = datetime.now()

    def __getattr__(self, name):
        if name in self._module_map:
            if name not in self._cache:
                module_path, attribute, category = self._module_map[name]
                icon = Banner.get_module_icon(category)
                
                try:
                    # Show loading animation
                    print(f"{C.YELLOW}🔄 Loading {icon} {category}.{name}...{C.RESET}", end='\r')
                    time.sleep(0.05)  # Small delay untuk effect
                    
                    module = importlib.import_module(module_path)
                    result = getattr(module, attribute)
                    
                    self._cache[name] = result
                    self._loading_stats['loaded'] += 1
                    
                    # Success message
                    print(f"{C.GREEN}✅ Loaded {icon} {C.BOLD}{name}{C.RESET}{C.GREEN} successfully!{C.RESET}")
                    
                except (ImportError, AttributeError) as e:
                    # Error handling
                    self._cache[name] = None
                    self._loading_stats['failed'] += 1
                    print(f"{C.RED}❌ Failed to load {icon} {name}: {str(e)[:50]}...{C.RESET}")
                    
            return self._cache[name]
        raise AttributeError(f"module 'coffecrawler' has no attribute '{name}'")

    def get_detailed_stats(self):
        """Get detailed loading statistics"""
        if not self._loading_stats['end_time']:
            self._loading_stats['end_time'] = datetime.now()
        
        load_time = (self._loading_stats['end_time'] - self._loading_stats['start_time']).total_seconds()
        
        return {
            'total_modules': self._loading_stats['total'],
            'loaded_successfully': self._loading_stats['loaded'],
            'failed_to_load': self._loading_stats['failed'],
            'load_time_seconds': round(load_time, 3),
            'success_rate': round((self._loading_stats['loaded'] / self._loading_stats['total']) * 100, 1)
        }
    
    def get_module_categories(self):
        """Get statistics by module category"""
        categories = {}
        for name, (_, _, category) in self._module_map.items():
            if category not in categories:
                categories[category] = {'total': 0, 'loaded': 0}
            categories[category]['total'] += 1
            if self._cache.get(name) is not None:
                categories[category]['loaded'] += 1
        return categories

# Initialize advanced loader
_advanced_lazy = _AdvancedLazyLoader()

# =============================================================================
# 🚀 PUBLIC API - SEMUA ATTRIBUTES DI-HANDLE OLEH ADVANCED LOADER
# =============================================================================

# 🎯 CORE MODULES
CoffeCrawler = _advanced_lazy.CoffeCrawler
HTTPClient = _advanced_lazy.HTTPClient
HeadlessBrowser = _advanced_lazy.HeadlessBrowser
ParserEngine = _advanced_lazy.ParserEngine

# 🤖 AI AGENTS
BotAgent = _advanced_lazy.BotAgent
HumanEmulator = _advanced_lazy.HumanEmulator
RotationManager = _advanced_lazy.RotationManager
StrategySelector = _advanced_lazy.StrategySelector
StrategyType = _advanced_lazy.StrategyType

# 🎯 STRATEGY SYSTEM
get_strategy_selector = _advanced_lazy.get_strategy_selector
quick_strategy_select = _advanced_lazy.quick_strategy_select
get_strategy_analytics = _advanced_lazy.get_strategy_analytics

# 🛠️ UTILITIES
DataExtractor = _advanced_lazy.DataExtractor
DebugFixer = _advanced_lazy.DebugFixer
AntiDetection = _advanced_lazy.AntiDetection
CacheManager = _advanced_lazy.CacheManager

# 🔧 HELPER FUNCTIONS
enable_debug_mode = _advanced_lazy.enable_debug_mode
enable_fixer_mode = _advanced_lazy.enable_fixer_mode
set_mobile_emulation = _advanced_lazy.set_mobile_emulation
export_data = _advanced_lazy.export_data
quick_crawl = _advanced_lazy.quick_crawl

# 🔌 BROWSER PLUGINS
Mozilla = _advanced_lazy.Mozilla
Chrome = _advanced_lazy.Chrome
AndroidBrowser = _advanced_lazy.AndroidBrowser
Safari = _advanced_lazy.Safari
Edge = _advanced_lazy.Edge

# ⚙️ CONFIG PRESETS
STEALTH_MODE = _advanced_lazy.STEALTH_MODE
AGGRESSIVE_MODE = _advanced_lazy.AGGRESSIVE_MODE
SMART_MODE = _advanced_lazy.SMART_MODE
SAFE_MODE = _advanced_lazy.SAFE_MODE
TERMUX_OPTIMIZED = _advanced_lazy.TERMUX_OPTIMIZED

# 🚨 EXCEPTIONS
CoffeCrawlerError = _advanced_lazy.CoffeCrawlerError
CrawlerBlockedError = _advanced_lazy.CrawlerBlockedError
ParserError = _advanced_lazy.ParserError
NetworkError = _advanced_lazy.NetworkError
FixerModeActivated = _advanced_lazy.FixerModeActivated
AIError = _advanced_lazy.AIError
StrategyError = _advanced_lazy.StrategyError
ConfigurationError = _advanced_lazy.ConfigurationError
TermuxCompatibilityError = _advanced_lazy.TermuxCompatibilityError
RotationError = _advanced_lazy.RotationError
ProxyError = _advanced_lazy.ProxyError
IdentityError = _advanced_lazy.IdentityError
SecurityError = _advanced_lazy.SecurityError
EmulationError = _advanced_lazy.EmulationError

# =============================================================================
# 🎯 PUBLIC API LIST
# =============================================================================

__all__ = [
    # Core Modules
    'CoffeCrawler', 'HTTPClient', 'HeadlessBrowser', 'ParserEngine',
    
    # AI Agents
    'BotAgent', 'HumanEmulator', 'RotationManager', 'StrategySelector', 'StrategyType',
    
    # Strategy System
    'get_strategy_selector', 'quick_strategy_select', 'get_strategy_analytics',
    
    # Utilities
    'DataExtractor', 'DebugFixer', 'AntiDetection', 'CacheManager',
    
    # Helper Functions
    'enable_debug_mode', 'enable_fixer_mode', 'set_mobile_emulation',
    'export_data', 'quick_crawl',
    
    # Browser Plugins
    'Mozilla', 'Chrome', 'AndroidBrowser', 'Safari', 'Edge',
    
    # Configuration Presets
    'STEALTH_MODE', 'AGGRESSIVE_MODE', 'SMART_MODE', 'SAFE_MODE', 'TERMUX_OPTIMIZED',
    
    # Exceptions
    'CoffeCrawlerError', 'CrawlerBlockedError', 'ParserError', 'NetworkError',
    'FixerModeActivated', 'AIError', 'StrategyError', 'ConfigurationError',
    'TermuxCompatibilityError', 'RotationError', 'ProxyError', 'IdentityError',
    'SecurityError', 'EmulationError',
    
    # Public Functions
    'get_version', 'get_loading_stats', 'system_dashboard', 'module_categories'
]

# =============================================================================
# ⚙️ PUBLIC FUNCTIONS
# =============================================================================

def get_version() -> str:
    """Get current CoffeCrawler version"""
    return __version__

def get_loading_stats() -> Dict[str, Any]:
    """Get detailed loading statistics"""
    return _advanced_lazy.get_detailed_stats()

def module_categories() -> Dict[str, Dict]:
    """Get module loading statistics by category"""
    return _advanced_lazy.get_module_categories()

def system_dashboard() -> Dict[str, Any]:
    """Get comprehensive system dashboard"""
    stats = get_loading_stats()
    categories = module_categories()
    
    # System info
    termux_detected = 'TERMUX_VERSION' in os.environ
    python_version = sys.version.split()[0]
    platform_info = sys.platform
    
    # Performance metrics
    performance = "🚀 OPTIMAL" if stats['success_rate'] > 80 else "⚠️  DEGRADED" if stats['success_rate'] > 50 else "🔴 CRITICAL"
    
    return {
        'version': __version__,
        'performance': performance,
        'load_time': f"{stats['load_time_seconds']}s",
        'success_rate': f"{stats['success_rate']}%",
        'modules_loaded': f"{stats['loaded_successfully']}/{stats['total_modules']}",
        'platform': platform_info,
        'python_version': python_version,
        'termux_detected': termux_detected,
        'categories': categories
    }

# =============================================================================
# 🎪 EPIC INITIALIZATION & DASHBOARD
# =============================================================================

def _display_epic_welcome():
    """Display epic welcome dashboard"""
    
    # Clear screen and show banner
    os.system('clear' if os.name == 'posix' else 'cls')
    print(Banner.get_main_banner())
    
    # Initial loading message
    print(f"{C.CYAN}{C.BOLD}🚀 Initializing CoffeCrawler System...{C.RESET}")
    print(f"{C.PURPLE}⏳ Loading modules...{C.RESET}")
    time.sleep(0.5)

def _display_system_dashboard():
    """Display comprehensive system dashboard"""
    dashboard = system_dashboard()
    categories = dashboard['categories']
    
    print(f"\n{C.WHITE}{C.BOLD}📊 SYSTEM DASHBOARD{C.RESET}")
    print(f"{C.CYAN}┌────────────────────────────────────────────────────────┐{C.RESET}")
    
    # Main stats
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Version:{C.RESET} {C.GREEN}{dashboard['version']}{C.RESET}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Performance:{C.RESET} {Banner.get_status_color('active' if dashboard['success_rate'] > '80' else 'warning', dashboard['performance'])}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Load Time:{C.RESET} {C.YELLOW}{dashboard['load_time']}{C.RESET}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Success Rate:{C.RESET} {C.GREEN}{dashboard['success_rate']}{C.RESET}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Platform:{C.RESET} {C.BLUE}{dashboard['platform']}{C.RESET}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}Termux:{C.RESET} {Banner.get_status_color('active' if dashboard['termux_detected'] else 'inactive', 'DETECTED' if dashboard['termux_detected'] else 'NOT DETECTED')}")
    
    print(f"{C.CYAN}├────────────────────────────────────────────────────────┤{C.RESET}")
    
    # Module categories
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}📁 MODULE CATEGORIES:{C.RESET}")
    for category, stats in categories.items():
        icon = Banner.get_module_icon(category)
        loaded = stats['loaded']
        total = stats['total']
        percentage = (loaded / total) * 100 if total > 0 else 0
        
        color = C.GREEN if percentage > 80 else C.YELLOW if percentage > 50 else C.RED
        status_icon = "✅" if percentage > 80 else "⚠️ " if percentage > 50 else "❌"
        
        print(f"{C.CYAN}│{C.RESET}   {status_icon} {icon} {C.BOLD}{category.upper():<12}{C.RESET} {color}{loaded:2d}/{total:2d} ({percentage:5.1f}%){C.RESET}")
    
    print(f"{C.CYAN}└────────────────────────────────────────────────────────┘{C.RESET}")

def _display_quick_start():
    """Display quick start guide"""
    print(f"\n{C.GREEN}{C.BOLD}🎯 QUICK START GUIDE{C.RESET}")
    print(f"{C.YELLOW}┌────────────────────────────────────────────────────────┐{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET} {C.CYAN}Basic Crawling:{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}from coffecrawler import CoffeCrawler{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}crawler = CoffeCrawler(){C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}data = crawler.crawl('https://example.com'){C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET} {C.CYAN}AI Strategy Selection:{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}from coffecrawler import get_strategy_selector{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}selector = get_strategy_selector(){C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}strategy = selector.select_strategy(analysis, env){C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET} {C.CYAN}Debug Info:{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}import coffecrawler{C.RESET}")
    print(f"{C.YELLOW}│{C.RESET}   {C.WHITE}print(coffecrawler.system_dashboard()){C.RESET}")
    print(f"{C.YELLOW}└────────────────────────────────────────────────────────┘{C.RESET}")

# =============================================================================
# 🚀 MAIN INITIALIZATION
# =============================================================================

def _initialize_epic_system():
    """Initialize the epic CoffeCrawler system"""
    
    # Display welcome banner
    _display_epic_welcome()
    
    # Modules are loaded lazily, so we trigger some core ones
    print(f"\n{C.PURPLE}🎯 Loading core modules...{C.RESET}")
    
    # Trigger loading of essential modules
    _ = CoffeCrawler
    _ = BotAgent
    _ = StrategySelector
    _ = get_strategy_selector
    
    # Finalize loading stats
    _advanced_lazy.get_detailed_stats()
    
    # Display dashboard
    _display_system_dashboard()
    
    # Display quick start
    _display_quick_start()
    
    print(f"\n{C.GREEN}{C.BOLD}🎉 COFFEECRAWLER SYSTEM READY! Happy crawling! 🚀{C.RESET}")

# Auto-initialize epic system
_initialize_epic_system()

# Cleanup with style
import atexit
def _epic_shutdown():
    stats = get_loading_stats()
    print(f"\n{C.PURPLE}☕ CoffeCrawler shutdown complete. {stats['loaded_successfully']} modules served. Thank you!{C.RESET}")

atexit.register(_epic_shutdown)
