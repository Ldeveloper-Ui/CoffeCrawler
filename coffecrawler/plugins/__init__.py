"""
PLUGINS MODULE - Extensible Browser & Functionality Plugins
Expand CoffeCrawler's capabilities with powerful plugins.
Plugins is Here!
"""

from .browser_plugins import (
    Mozilla,
    Chrome,
    AndroidBrowser,
    Safari,
    Edge,
    Opera,
    CustomBrowser
)

from .extraction_plugins import (
    EcommerceExtractor,
    SocialMediaExtractor,
    SearchEngineExtractor,
    NewsExtractor,
    ImageExtractor,
    VideoExtractor
)

from .ai_plugins import (
    ContentAnalyzer,
    SentimentDetector,
    PatternRecognizer,
    SmartClassifier
)

from .export_plugins import (
    ExcelExporter,
    PDFExporter,
    DatabaseExporter,
    CloudExporter
)

__all__ = [
    # Browser Plugins
    'Mozilla',
    'Chrome',
    'AndroidBrowser', 
    'Safari',
    'Edge',
    'Opera',
    'CustomBrowser',
    
    # Extraction Plugins
    'EcommerceExtractor',
    'SocialMediaExtractor',
    'SearchEngineExtractor',
    'NewsExtractor',
    'ImageExtractor', 
    'VideoExtractor',
    
    # AI Plugins
    'ContentAnalyzer',
    'SentimentDetector',
    'PatternRecognizer',
    'SmartClassifier',
    
    # Export Plugins
    'ExcelExporter',
    'PDFExporter',
    'DatabaseExporter',
    'CloudExporter'
]

# Plugin registry system
PLUGIN_REGISTRY = {
    'browsers': {
        'mozilla': Mozilla,
        'chrome': Chrome,
        'android': AndroidBrowser,
        'safari': Safari,
        'edge': Edge,
        'opera': Opera,
        'custom': CustomBrowser
    },
    'extractors': {
        'ecommerce': EcommerceExtractor,
        'social_media': SocialMediaExtractor,
        'search_engine': SearchEngineExtractor,
        'news': NewsExtractor,
        'images': ImageExtractor,
        'videos': VideoExtractor
    },
    'ai_tools': {
        'content_analysis': ContentAnalyzer,
        'sentiment': SentimentDetector,
        'pattern_recognition': PatternRecognizer,
        'classification': SmartClassifier
    }
}

class PluginManager:
    """Central plugin management system"""
    
    @staticmethod
    def list_plugins():
        """List all available plugins"""
        return {
            category: list(plugins.keys())
            for category, plugins in PLUGIN_REGISTRY.items()
        }
    
    @staticmethod
    def get_plugin(plugin_type, plugin_name):
        """Get specific plugin by type and name"""
        category = PLUGIN_REGISTRY.get(plugin_type, {})
        plugin_class = category.get(plugin_name)
        if plugin_class:
            return plugin_class()
        raise ValueError(f"Plugin {plugin_name} not found in {plugin_type}")
    
    @staticmethod
    def register_custom_plugin(plugin_type, plugin_name, plugin_class):
        """Register custom plugin - EXTENSIBILITY POWER!"""
        if plugin_type not in PLUGIN_REGISTRY:
            PLUGIN_REGISTRY[plugin_type] = {}
        PLUGIN_REGISTRY[plugin_type][plugin_name] = plugin_class
        return f"Plugin {plugin_name} registered successfully! ✅"

# Browser configuration presets
BROWSER_CONFIGS = {
    'mozilla_stealth': {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'accept_language': 'en-US,en;q=0.9',
        'viewport': '1920x1080'
    },
    'chrome_mobile': {
        'user_agent': 'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36',
        'viewport': '360x800',
        'device_scale_factor': 3.0
    },
    'termux_optimized': {
        'user_agent': 'Mozilla/5.0 (Linux; Android 13; Termux Build) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36',
        'low_resource': True,
        'memory_efficient': True,
        'battery_saver': True
    }
}

def get_browser_config(config_name):
    """Get predefined browser configuration"""
    return BROWSER_CONFIGS.get(config_name, BROWSER_CONFIGS['mozilla_stealth'])

def create_custom_browser(name, **config):
    """Create custom browser plugin on-the-fly"""
    from .browser_plugins import CustomBrowser
    return CustomBrowser(name, **config)

print("🔌 Plugin system loaded - Extensible architecture ready!")
