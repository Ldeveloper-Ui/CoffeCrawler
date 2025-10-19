"""
☕ CRAWLER ENGINE - The Brain of CoffeCrawler
Revolutionary multi-engine crawling system with AI-powered decision making
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from urllib.parse import urlparse
import warnings

from ..agents.bot_agent import BotAgent
from ..agents.human_emulator import HumanEmulator
from ..utils.data_extractor import DataExtractor
from ..utils.debug_fixer import DebugFixer
from ..utils.performance_optimizer import PerformanceOptimizer
from ..exceptions import (
    CoffeCrawlerError,
    CrawlerBlockedError,
    ParserError,
    NetworkError
)


@dataclass
class CrawlResult:
    """Advanced result container with metadata"""
    success: bool
    data: Any
    url: str
    engine_used: str
    response_time: float
    status_code: int = 200
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    fixer_applied: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []


class CoffeCrawler:
    """
    ☕ MAIN CRAWLER ENGINE - Revolutionary Web Crawling System
    
    Features:
    - Multi-engine architecture (HTTP + Headless)
    - AI-powered decision making
    - Smart fallback system
    - Auto-fix and debugging
    - Mobile optimization
    - Real-time adaptation
    """
    
    def __init__(self, 
                 mode: str = 'smart',
                 agent_type: str = 'stealth', 
                 agent_module: str = 'Mozilla',
                 module_version: str = 'default',
                 **kwargs):
        """
        Initialize CoffeCrawler with advanced configuration
        
        Args:
            mode: Operation mode ('smart', 'stealth', 'aggressive', 'safe', 'hybrid')
            agent_type: Agent behavior ('stealth', 'aggressive', 'adaptive', 'intelligent')
            agent_module: Browser module ('Mozilla', 'Chrome', 'AndroidBrowser')
            module_version: Specific module version
            **kwargs: Advanced configuration options
        """
        # Core configuration
        self.mode = mode.lower()
        self.agent_type = agent_type.lower()
        self.agent_module = agent_module
        self.module_version = module_version
        
        # Advanced settings
        self.debug_mode = kwargs.get('debug_mode', False)
        self.auto_fix = kwargs.get('auto_fix', True)
        self.auto_retry = kwargs.get('auto_retry', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout = kwargs.get('timeout', 30)
        self.enable_javascript = kwargs.get('enable_javascript', True)
        self.mobile_emulation = kwargs.get('mobile_emulation', True)
        
        # Performance optimization
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.compression_enabled = kwargs.get('compression_enabled', True)
        self.parallel_processing = kwargs.get('parallel_processing', True)
        
        # Initialize components
        self._initialize_components()
        self._performance_monitor = PerformanceMonitor()
        
        # Session management
        self.session_id = self._generate_session_id()
        self.request_count = 0
        self.total_response_time = 0
        
        print(f"☕ CoffeCrawler v1.0.0 Initialized")
        print(f"   Mode: {self.mode.upper()} | Agent: {self.agent_type.upper()} | Module: {self.agent_module}")
        if self.debug_mode:
            print(f"   🔧 Debug Mode: ENABLED | 🛠️ Auto-Fix: {self.auto_fix}")
    
    def _initialize_components(self):
        """Initialize all core components"""
        try:
            # Core engines
            from .http_client import HTTPClient
            from .headless_browser import HeadlessBrowser
            from .parser_engine import ParserEngine
            
            self.http_client = HTTPClient(self)
            self.headless_browser = HeadlessBrowser(self)
            self.parser_engine = ParserEngine(self)
            
            # AI and utility components
            self.bot_agent = BotAgent(self)
            self.human_emulator = HumanEmulator(self)
            self.data_extractor = DataExtractor(self)
            self.debug_fixer = DebugFixer(self)
            self.performance_optimizer = PerformanceOptimizer(self)
            
            # Session and state management
            self.engine_router = EngineRouter(self)
            self.session_manager = SessionManager(self)
            
        except Exception as e:
            raise CoffeCrawlerError(f"Component initialization failed: {e}")
    
    def _generate_session_id(self):
        """Generate unique session ID"""
        import hashlib
        import time
        base_string = f"{time.time()}_{self.mode}_{self.agent_type}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]
    
    def data(self, 
             url: str, 
             extract_rules: Union[List[str], Dict[str, str], str],
             bot_agent: str = 'adaptive',
             **kwargs) -> CrawlResult:
        """
        🎯 MAIN DATA EXTRACTION METHOD - Revolutionary One-Call Solution
        
        Args:
            url: Target URL to crawl
            extract_rules: Rules for data extraction
            bot_agent: Bot behavior mode
            **kwargs: Additional parameters
        
        Returns:
            CrawlResult: Structured result with data and metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        if self.debug_mode:
            print(f"🔍 CRAWL START: {url}")
            print(f"   Rules: {extract_rules} | Bot: {bot_agent}")
        
        try:
            # 1. AI-Powered Strategy Selection
            strategy = self.bot_agent.choose_strategy(url, extract_rules, bot_agent)
            
            # 2. Human Emulation (if stealth mode)
            if self.agent_type == 'stealth':
                self.human_emulator.simulate_behavior(strategy)
            
            # 3. Engine Execution with Smart Fallback
            raw_data = self._execute_with_fallback(url, strategy, kwargs)
            
            # 4. Smart Data Extraction
            extracted_data = self.data_extractor.process(raw_data, extract_rules, strategy)
            
            # 5. Performance Monitoring
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            # 6. Return Structured Result
            return CrawlResult(
                success=True,
                data=extracted_data,
                url=url,
                engine_used=strategy['engine'],
                response_time=response_time,
                status_code=getattr(raw_data, 'status_code', 200),
                metadata={
                    'strategy': strategy,
                    'session_id': self.session_id,
                    'request_number': self.request_count,
                    'content_type': getattr(raw_data, 'content_type', 'unknown'),
                    'content_length': len(str(raw_data)) if raw_data else 0
                }
            )
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ CRAWL ERROR: {type(e).__name__}: {e}")
            
            # Auto-Fix Mode Activation
            if self.auto_fix:
                fixed_result = self.debug_fixer.auto_recover(e, url, extract_rules)
                if fixed_result:
                    fixed_result.fixer_applied = True
                    return fixed_result
            
            # Return error result
            response_time = time.time() - start_time
            return CrawlResult(
                success=False,
                data=None,
                url=url,
                engine_used='unknown',
                response_time=response_time,
                status_code=500,
                errors=[f"{type(e).__name__}: {str(e)}"]
            )
    
    def _execute_with_fallback(self, url: str, strategy: Dict, kwargs: Dict) -> Any:
        """
        🚀 SMART EXECUTION WITH AUTOMATIC FALLBACK
        
        Implements multi-engine execution with intelligent fallback
        """
        engines_to_try = self.engine_router.get_engine_sequence(strategy)
        
        for i, engine_config in enumerate(engines_to_try):
            engine_name = engine_config['engine']
            engine_method = engine_config['method']
            
            try:
                if self.debug_mode:
                    print(f"   🚀 Trying {engine_name.upper()} engine...")
                
                # Execute with current engine
                if engine_name == 'http':
                    result = self.http_client.fetch(url, strategy, **kwargs)
                elif engine_name == 'headless':
                    result = self.headless_browser.fetch(url, strategy, **kwargs)
                else:
                    continue
                
                # Validate result
                if self._validate_result(result, strategy):
                    if self.debug_mode:
                        print(f"   ✅ {engine_name.upper()} engine SUCCESS!")
                    return result
                else:
                    raise NetworkError(f"Invalid response from {engine_name}")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"   ❌ {engine_name.upper()} engine failed: {e}")
                
                # Last engine in sequence - raise exception
                if i == len(engines_to_try) - 1:
                    raise
                # Otherwise continue to next engine
                continue
        
        raise NetworkError("All engines failed")
    
    def _validate_result(self, result: Any, strategy: Dict) -> bool:
        """Validate if the result meets quality criteria"""
        if result is None:
            return False
        
        # Check for blocking indicators
        if hasattr(result, 'text'):
            text = result.text.lower()
            blocking_indicators = [
                'access denied', 'captcha', 'cloudflare', 
                'bot detected', 'forbidden'
            ]
            if any(indicator in text for indicator in blocking_indicators):
                raise CrawlerBlockedError("Website blocked the request")
        
        return True
    
    def batch_crawl(self, 
                   urls: List[str], 
                   extract_rules: Union[List[str], Dict[str, str]],
                   bot_agent: str = 'adaptive',
                   max_concurrent: int = 5) -> List[CrawlResult]:
        """
        🔥 BATCH CRAWLING - Process multiple URLs efficiently
        
        Args:
            urls: List of URLs to crawl
            extract_rules: Extraction rules
            bot_agent: Bot behavior
            max_concurrent: Maximum concurrent requests
        
        Returns:
            List of CrawlResult objects
        """
        if self.debug_mode:
            print(f"🔥 BATCH CRAWL: {len(urls)} URLs | Concurrent: {max_concurrent}")
        
        results = []
        
        if self.parallel_processing and len(urls) > 1:
            # Parallel execution
            results = self._parallel_batch_crawl(urls, extract_rules, bot_agent, max_concurrent)
        else:
            # Sequential execution
            for url in urls:
                result = self.data(url, extract_rules, bot_agent)
                results.append(result)
        
        return results
    
    def _parallel_batch_crawl(self, urls, extract_rules, bot_agent, max_concurrent):
        """Execute batch crawling in parallel"""
        import concurrent.futures
        
        def crawl_single(url):
            return self.data(url, extract_rules, bot_agent)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_url = {executor.submit(crawl_single, url): url for url in urls}
            results = []
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = CrawlResult(
                        success=False,
                        data=None,
                        url=url,
                        engine_used='unknown',
                        response_time=0,
                        errors=[str(e)]
                    )
                    results.append(error_result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_response_time = (self.total_response_time / self.request_count 
                           if self.request_count > 0 else 0)
        
        return {
            'session_id': self.session_id,
            'total_requests': self.request_count,
            'average_response_time': round(avg_response_time, 2),
            'current_mode': self.mode,
            'agent_type': self.agent_type,
            'memory_usage': self._get_memory_usage(),
            'cache_hit_rate': getattr(self.http_client.cache_manager, 'hit_rate', 0)
        }
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        return f"{process.memory_info().rss // 1024 // 1024} MB"
    
    def enable_debug_mode(self, level: str = 'verbose'):
        """Enable advanced debugging"""
        self.debug_mode = True
        self.debug_level = level
        print(f"🔧 Debug Mode ENABLED (Level: {level})")
    
    def disable_debug_mode(self):
        """Disable debugging"""
        self.debug_mode = False
        print("🔧 Debug Mode DISABLED")
    
    def change_mode(self, new_mode: str, new_agent: str = None):
        """Dynamically change operation mode"""
        valid_modes = ['smart', 'stealth', 'aggressive', 'safe', 'hybrid']
        if new_mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        
        self.mode = new_mode
        if new_agent:
            self.agent_type = new_agent
        
        print(f"🔄 Mode changed: {new_mode.upper()} | Agent: {self.agent_type.upper()}")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.close()
    
    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'http_client'):
            self.http_client.close()
        if hasattr(self, 'headless_browser'):
            self.headless_browser.close()
        
        print("☕ CoffeCrawler session ended gracefully")


# Supporting Classes
class EngineRouter:
    """Intelligent engine routing system"""
    
    def __init__(self, crawler):
        self.crawler = crawler
    
    def get_engine_sequence(self, strategy: Dict) -> List[Dict]:
        """Get optimal engine sequence based on strategy"""
        base_sequence = []
        
        mode = self.crawler.mode
        agent_type = self.crawler.agent_type
        
        if mode == 'smart':
            base_sequence = [
                {'engine': 'http', 'method': 'fast'},
                {'engine': 'headless', 'method': 'fallback'}
            ]
        elif mode == 'stealth':
            base_sequence = [
                {'engine': 'headless', 'method': 'stealth'},
                {'engine': 'http', 'method': 'fallback'}
            ]
        elif mode == 'aggressive':
            base_sequence = [
                {'engine': 'http', 'method': 'aggressive'},
                {'engine': 'http', 'method': 'retry'},
                {'engine': 'headless', 'method': 'last_resort'}
            ]
        elif mode == 'safe':
            base_sequence = [
                {'engine': 'headless', 'method': 'safe'},
                {'engine': 'http', 'method': 'limited'}
            ]
        elif mode == 'hybrid':
            base_sequence = [
                {'engine': 'http', 'method': 'primary'},
                {'engine': 'headless', 'method': 'secondary'}
            ]
        
        return base_sequence


class SessionManager:
    """Advanced session management"""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.sessions = {}
    
    def get_session(self, engine: str):
        """Get or create session for specific engine"""
        if engine not in self.sessions:
            if engine == 'http':
                self.sessions[engine] = self._create_http_session()
            elif engine == 'headless':
                self.sessions[engine] = self._create_headless_session()
        
        return self.sessions[engine]
    
    def _create_http_session(self):
        """Create HTTP session with optimal configuration"""
        import requests
        session = requests.Session()
        
        # Configure session based on crawler settings
        session.headers.update({
            'User-Agent': self._get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def _create_headless_session(self):
        """Create headless browser session"""
        # Implementation depends on headless browser backend
        return None
    
    def _get_user_agent(self):
        """Get appropriate user agent based on configuration"""
        if self.crawler.mobile_emulation:
            return ("Mozilla/5.0 (Linux; Android 10; SM-G973F) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/119.0.0.0 Mobile Safari/537.36")
        else:
            return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/119.0.0.0 Safari/537.36")


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_failed': 0,
            'average_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.metrics['requests_total'] += 1
        if not success:
            self.metrics['requests_failed'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['requests_total']
        self.metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def record_cache(self, hit: bool):
        """Record cache performance"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()


# Quick utility function
def create_crawler(mode='smart', **kwargs):
    """Quick factory function for creating crawlers"""
    return CoffeCrawler(mode=mode, **kwargs)


print("✅ CoffeCrawler Engine loaded successfully!")
