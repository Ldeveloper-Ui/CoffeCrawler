"""
🚀 HEADLESS BROWSER - Advanced Headless Browser Engine for CoffeCrawler
Revolutionary browser automation with stealth, AI-powered behavior, and multi-engine support
"""

import asyncio
import random
import time
import base64
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
import json
import warnings

from ..exceptions import (
    CoffeCrawlerError,
    BrowserError, 
    JSError,
    CrawlerBlockedError
)
from ..utils.performance_optimizer import PerformanceOptimizer


@dataclass
class BrowserSession:
    """Advanced browser session container"""
    id: str
    page: Any
    context: Any
    browser: Any
    start_time: float
    user_agent: str
    viewport: Dict[str, int]
    stealth_level: str


class HeadlessBrowser:
    """
    🚀 ADVANCED HEADLESS BROWSER ENGINE - Revolutionary Browser Automation
    
    Features:
    - Multi-browser support (Playwright, Selenium, native)
    - AI-powered stealth and evasion
    - Human behavior emulation
    - Mobile device simulation
    - Automatic detection bypass
    - Screenshot and debugging capabilities
    - Resource optimization for Termux
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.sessions = {}
        self.active_pages = {}
        self.performance_monitor = BrowserPerformanceMonitor()
        
        # Browser configuration
        self.browser_type = self._determine_browser_type()
        self.stealth_mode = crawler.agent_type in ['stealth', 'adaptive']
        self.human_emulation = crawler.agent_type in ['stealth', 'adaptive', 'intelligent']
        self.mobile_emulation = crawler.mobile_emulation
        
        # Advanced features
        self.enable_javascript = crawler.enable_javascript
        self.resource_optimization = True  # Critical for Termux
        self.auto_bypass = True  # Automatic anti-bot bypass
        
        # Initialize browser engine
        self._initialize_browser_engine()
        
        if crawler.debug_mode:
            print(f"🚀 Headless Browser initialized - Engine: {self.browser_type.upper()}")
            print(f"   Stealth: {self.stealth_mode} | Human Emulation: {self.human_emulation}")
    
    def _determine_browser_type(self) -> str:
        """Determine the best available browser engine"""
        # Priority: Playwright > Selenium > native
        try:
            import playwright
            return 'playwright'
        except ImportError:
            try:
                import selenium
                return 'selenium'
            except ImportError:
                return 'native'
    
    def _initialize_browser_engine(self):
        """Initialize the selected browser engine"""
        try:
            if self.browser_type == 'playwright':
                self._init_playwright()
            elif self.browser_type == 'selenium':
                self._init_selenium()
            else:
                self._init_native()
                
        except Exception as e:
            raise BrowserError(f"Browser engine initialization failed: {e}")
    
    def _init_playwright(self):
        """Initialize Playwright engine with advanced features"""
        import playwright
        from playwright.async_api import async_playwright
        
        self.playwright = async_playwright()
        self.browser = None
        self.context_pool = []
        
        print("🔧 Initializing Playwright with advanced features...")
    
    def _init_selenium(self):
        """Initialize Selenium engine"""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        
        # Advanced configuration for stealth
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Mobile emulation
        if self.mobile_emulation:
            mobile_emulation = {
                "deviceMetrics": {"width": 390, "height": 844, "pixelRatio": 3.0},
                "userAgent": self._get_mobile_user_agent()
            }
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Termux optimization
        if self.resource_optimization:
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-images')
            options.add_argument('--blink-settings=imagesEnabled=false')
        
        self.driver = webdriver.Chrome(options=options)
        
        # Apply stealth techniques
        if self.stealth_mode:
            self._apply_selenium_stealth()
    
    def _init_native(self):
        """Initialize native browser engine (fallback)"""
        print("⚠️  Using native browser engine - limited features")
        # Basic implementation using requests + BeautifulSoup for simple JS rendering
    
    async def fetch(self, url: str, strategy: Dict, **kwargs) -> Dict[str, Any]:
        """
        🎯 MAIN FETCH METHOD - Advanced Browser-Based Page Retrieval
        
        Args:
            url: Target URL
            strategy: Crawling strategy
            **kwargs: Additional parameters
        
        Returns:
            Dict with page content and metadata
        """
        start_time = time.time()
        session_id = self._generate_session_id()
        
        try:
            if self.crawler.debug_mode:
                print(f"   🌐 Browser navigating to: {url}")
            
            # Create browser session
            session = await self._create_browser_session(strategy)
            self.sessions[session_id] = session
            
            # Apply advanced stealth if enabled
            if self.stealth_mode:
                await self._apply_advanced_stealth(session.page)
            
            # Navigate to URL
            await self._navigate_to_url(session.page, url, strategy)
            
            # Human behavior simulation
            if self.human_emulation:
                await self._simulate_human_behavior(session.page, strategy)
            
            # Wait for page readiness
            await self._wait_for_page_ready(session.page, strategy)
            
            # Get page content
            page_content = await self._extract_page_content(session.page)
            
            # Capture screenshots if debug mode
            screenshot = None
            if self.crawler.debug_mode:
                screenshot = await self._capture_screenshot(session.page)
            
            # Performance monitoring
            load_time = time.time() - start_time
            self.performance_monitor.record_page_load(load_time, True)
            
            return {
                'success': True,
                'content': page_content,
                'url': url,
                'load_time': load_time,
                'session_id': session_id,
                'screenshot': screenshot,
                'metadata': {
                    'browser_engine': self.browser_type,
                    'viewport': session.viewport,
                    'user_agent': session.user_agent,
                    'stealth_level': session.stealth_level
                }
            }
            
        except Exception as e:
            load_time = time.time() - start_time
            self.performance_monitor.record_page_load(load_time, False)
            
            if self.crawler.debug_mode:
                print(f"   ❌ Browser fetch failed: {e}")
            
            raise BrowserError(f"Headless browser operation failed: {e}") from e
        
        finally:
            # Cleanup session
            await self._cleanup_session(session_id)
    
    async def _create_browser_session(self, strategy: Dict) -> BrowserSession:
        """Create advanced browser session with optimal configuration"""
        if self.browser_type == 'playwright':
            return await self._create_playwright_session(strategy)
        elif self.browser_type == 'selenium':
            return await self._create_selenium_session(strategy)
        else:
            return await self._create_native_session(strategy)
    
    async def _create_playwright_session(self, strategy: Dict) -> BrowserSession:
        """Create Playwright browser session"""
        playwright = await self.playwright.start()
        
        # Browser type selection
        browser_config = self._get_browser_config(strategy)
        browser = await getattr(playwright, browser_config['type']).launch(
            headless=True,
            args=browser_config['args']
        )
        
        # Context configuration
        context_options = {
            'viewport': browser_config['viewport'],
            'user_agent': browser_config['user_agent'],
            'java_script_enabled': self.enable_javascript,
            'ignore_https_errors': True,
        }
        
        # Mobile emulation
        if self.mobile_emulation:
            context_options.update({
                'has_touch': True,
                'is_mobile': True,
                'device_scale_factor': 3.0
            })
        
        context = await browser.new_context(**context_options)
        
        # Create page
        page = await context.new_page()
        
        # Apply evasion techniques
        if self.stealth_mode:
            await self._apply_playwright_stealth(page)
        
        return BrowserSession(
            id=self._generate_session_id(),
            page=page,
            context=context,
            browser=browser,
            start_time=time.time(),
            user_agent=browser_config['user_agent'],
            viewport=browser_config['viewport'],
            stealth_level='advanced' if self.stealth_mode else 'basic'
        )
    
    async def _create_selenium_session(self, strategy: Dict) -> BrowserSession:
        """Create Selenium browser session"""
        # Selenium session is already initialized in __init__
        return BrowserSession(
            id=self._generate_session_id(),
            page=self.driver,
            context=None,
            browser=self.driver,
            start_time=time.time(),
            user_agent=self.driver.execute_script("return navigator.userAgent"),
            viewport={'width': 1920, 'height': 1080},
            stealth_level='basic'
        )
    
    async def _create_native_session(self, strategy: Dict) -> BrowserSession:
        """Create native browser session (fallback)"""
        # Simple session for basic HTML rendering
        return BrowserSession(
            id=self._generate_session_id(),
            page=None,
            context=None,
            browser=None,
            start_time=time.time(),
            user_agent=self._get_user_agent(),
            viewport={'width': 1920, 'height': 1080},
            stealth_level='none'
        )
    
    async def _apply_advanced_stealth(self, page):
        """Apply advanced stealth and evasion techniques"""
        if self.browser_type == 'playwright':
            await self._apply_playwright_stealth(page)
        elif self.browser_type == 'selenium':
            self._apply_selenium_stealth()
    
    async def _apply_playwright_stealth(self, page):
        """Apply Playwright-specific stealth techniques"""
        # Remove automation detection
        await page.add_init_script("""
            // Override webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Override chrome runtime
            Object.defineProperty(window, 'chrome', {
                get: () => ({
                    runtime: {},
                }),
            });
            
            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        # Additional evasion scripts
        evasion_scripts = [
            "delete Object.getPrototypeOf(navigator).webdriver",
            "window.navigator.chrome = { runtime: {} }",
            "Object.defineProperty(navigator, 'platform', { get: () => 'Win32' })"
        ]
        
        for script in evasion_scripts:
            try:
                await page.evaluate(script)
            except:
                pass
    
    def _apply_selenium_stealth(self):
        """Apply Selenium-specific stealth techniques"""
        # Remove automation flags
        self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
    
    async def _navigate_to_url(self, page, url: str, strategy: Dict):
        """Navigate to URL with intelligent waiting"""
        navigation_timeout = strategy.get('timeout', self.crawler.timeout) * 1000
        
        if self.browser_type == 'playwright':
            await page.goto(
                url, 
                timeout=navigation_timeout,
                wait_until='networkidle'  # Wait for network to be idle
            )
        elif self.browser_type == 'selenium':
            self.driver.get(url)
            self.driver.implicitly_wait(navigation_timeout / 1000)
    
    async def _simulate_human_behavior(self, page, strategy: Dict):
        """Simulate human-like browsing behavior"""
        if self.browser_type != 'playwright':
            return  # Only supported in Playwright
        
        behavior_config = {
            'aggressive': {'scrolls': 2, 'movements': 3, 'delay': 0.1},
            'stealth': {'scrolls': 5, 'movements': 8, 'delay': 0.5},
            'adaptive': {'scrolls': 3, 'movements': 5, 'delay': 0.3}
        }
        
        config = behavior_config.get(strategy.get('method', 'adaptive'))
        
        # Random mouse movements
        for _ in range(config['movements']):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.1, config['delay']))
        
        # Random scrolling
        for _ in range(config['scrolls']):
            scroll_amount = random.randint(200, 800)
            await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            await asyncio.sleep(random.uniform(0.2, config['delay']))
    
    async def _wait_for_page_ready(self, page, strategy: Dict):
        """Wait for page to be fully ready"""
        if self.browser_type == 'playwright':
            # Wait for specific elements or conditions
            try:
                await page.wait_for_function(
                    'document.readyState === "complete"',
                    timeout=5000
                )
            except:
                pass  # Continue even if not fully ready
            
            # Additional wait for dynamic content
            if strategy.get('wait_for_dynamic', False):
                await asyncio.sleep(2)
    
    async def _extract_page_content(self, page) -> str:
        """Extract page content with advanced processing"""
        if self.browser_type == 'playwright':
            content = await page.content()
            
            # Execute JavaScript to get dynamically loaded content
            try:
                dynamic_content = await page.evaluate("""
                    () => {
                        // Extract additional metadata
                        const metadata = {
                            title: document.title,
                            url: window.location.href,
                            timestamp: Date.now(),
                            contentLength: document.documentElement.outerHTML.length
                        };
                        
                        // Try to get text content for analysis
                        const bodyText = document.body ? document.body.innerText : '';
                        
                        return {
                            metadata: metadata,
                            bodyText: bodyText.substring(0, 5000) // Limit size
                        };
                    }
                """)
                
                # Add metadata to content
                content += f"\n<!-- DYNAMIC_METADATA: {json.dumps(dynamic_content)} -->"
                
            except Exception as e:
                if self.crawler.debug_mode:
                    print(f"   ⚠️ Dynamic content extraction failed: {e}")
            
            return content
        
        elif self.browser_type == 'selenium':
            return self.driver.page_source
        
        else:
            return ""  # Native implementation would use requests
    
    async def _capture_screenshot(self, page) -> Optional[str]:
        """Capture screenshot for debugging"""
        try:
            if self.browser_type == 'playwright':
                screenshot_bytes = await page.screenshot(type='jpeg', quality=80)
                return base64.b64encode(screenshot_bytes).decode('utf-8')
            elif self.browser_type == 'selenium':
                screenshot_bytes = self.driver.get_screenshot_as_png()
                return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Screenshot capture failed: {e}")
            return None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import hashlib
        import time
        base_string = f"{time.time()}_{random.random()}"
        return hashlib.md5(base_string.encode()).hexdigest()[:8]
    
    def _get_browser_config(self, strategy: Dict) -> Dict:
        """Get browser configuration based on strategy"""
        base_config = {
            'type': 'chromium',  # Default to Chromium
            'args': [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-features=VizDisplayCompositor',
            ],
            'viewport': {'width': 1920, 'height': 1080},
            'user_agent': self._get_user_agent()
        }
        
        # Mobile emulation
        if self.mobile_emulation:
            base_config['viewport'] = {'width': 390, 'height': 844}
            base_config['user_agent'] = self._get_mobile_user_agent()
        
        # Stealth enhancements
        if self.stealth_mode:
            base_config['args'].extend([
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
            ])
        
        # Termux optimizations
        if self.resource_optimization:
            base_config['args'].extend([
                '--disable-gpu',
                '--disable-extensions',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
            ])
        
        return base_config
    
    def _get_user_agent(self) -> str:
        """Get desktop user agent"""
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        )
    
    def _get_mobile_user_agent(self) -> str:
        """Get mobile user agent"""
        return (
            "Mozilla/5.0 (Linux; Android 10; SM-G973F) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Mobile Safari/537.36"
        )
    
    async def _cleanup_session(self, session_id: str):
        """Cleanup browser session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            try:
                if self.browser_type == 'playwright':
                    await session.context.close()
                    await session.browser.close()
                elif self.browser_type == 'selenium':
                    # Selenium session is reused, don't close here
                    pass
            except Exception as e:
                if self.crawler.debug_mode:
                    print(f"   ⚠️ Session cleanup failed: {e}")
            
            del self.sessions[session_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get browser performance statistics"""
        return self.performance_monitor.get_stats()
    
    async def close(self):
        """Close all browser resources"""
        # Close all active sessions
        for session_id in list(self.sessions.keys()):
            await self._cleanup_session(session_id)
        
        # Close main browser instance
        if hasattr(self, 'browser') and self.browser:
            await self.browser.close()
        
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        
        if hasattr(self, 'driver'):
            self.driver.quit()


class BrowserPerformanceMonitor:
    """Advanced browser performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            'total_page_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'average_load_time': 0,
            'total_load_time': 0,
            'resource_usage': [],
            'browser_crashes': 0
        }
    
    def record_page_load(self, load_time: float, success: bool):
        """Record page load performance"""
        self.metrics['total_page_loads'] += 1
        self.metrics['total_load_time'] += load_time
        
        if success:
            self.metrics['successful_loads'] += 1
        else:
            self.metrics['failed_loads'] += 1
        
        # Update average
        self.metrics['average_load_time'] = (
            self.metrics['total_load_time'] / self.metrics['total_page_loads']
        )
    
    def record_resource_usage(self, memory_usage: int, cpu_usage: float):
        """Record resource usage"""
        self.metrics['resource_usage'].append({
            'timestamp': time.time(),
            'memory_mb': memory_usage,
            'cpu_percent': cpu_usage
        })
        
        # Keep only last 100 records
        if len(self.metrics['resource_usage']) > 100:
            self.metrics['resource_usage'].pop(0)
    
    def record_browser_crash(self):
        """Record browser crash"""
        self.metrics['browser_crashes'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.metrics.copy()
        
        # Calculate success rate
        if stats['total_page_loads'] > 0:
            stats['success_rate'] = (
                stats['successful_loads'] / stats['total_page_loads'] * 100
            )
        else:
            stats['success_rate'] = 0
        
        return stats


# Factory function for easy creation
def create_headless_browser(crawler, browser_type: str = 'auto'):
    """Factory function to create headless browser instance"""
    browser = HeadlessBrowser(crawler)
    
    # Override browser type if specified
    if browser_type != 'auto':
        browser.browser_type = browser_type
    
    return browser


print("✅ Headless Browser Engine loaded successfully!")
