"""
🛠️ Utility helpers for CoffeCrawler
"""

def enable_debug_mode():
    """Enable debug mode for detailed logging"""
    print("🔧 Debug mode enabled")
    return {"debug": True, "verbose": True}

def enable_fixer_mode():
    """Enable auto-fix mode for automatic issue resolution"""
    print("🛠️ Fixer mode activated")
    return {"auto_fix": True, "self_healing": True}

def set_mobile_emulation():
    """Set mobile device emulation"""
    print("📱 Mobile emulation enabled")
    return {"mobile_view": True, "touch_emulation": True}

def export_data(data, format_type='json'):
    """Quick data export utility"""
    print(f"💾 Exporting data as {format_type}")
    return {"exported": True, "format": format_type}

def quick_crawl(url):
    """Quick crawl utility function"""
    print(f"🚀 Quick crawling: {url}")
    return {"status": "started", "url": url}
