"""
🚀 HTTP CLIENT - Lightning-Fast HTTP Engine for CoffeCrawler
Advanced HTTP client with caching, retry mechanisms, and smart optimization
"""

import requests
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
import gzip
import brotli
from functools import wraps
import threading

from ..exceptions import NetworkError, CrawlerBlockedError
from ..utils.cache_manager import CacheManager
from ..utils.performance_optimizer import PerformanceOptimizer


class HTTPClient:
    """
    🚀 ADVANCED HTTP CLIENT - Core HTTP Engine for CoffeCrawler
    
    Features:
    - Smart caching with multiple backends
    - Automatic retry with exponential backoff
    - Connection pooling and keep-alive
    - Compression handling
    - Rate limiting
    - Request deduplication
    - Mobile optimization
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.session = None
        self.cache_manager = CacheManager(crawler)
        self.performance_optimizer = PerformanceOptimizer(crawler)
        self._lock = threading.RLock()
        
        # Request tracking
        self.request_history = []
        self.consecutive_failures = 0
        
        # Initialize client
        self._initialize_session()
        
        if crawler.debug_mode:
            print("🚀 HTTP Client initialized with advanced features")
    
    def _initialize_session(self):
        """Initialize HTTP session with optimal configuration"""
        self.session = requests.Session()
        
        # Configure session based on crawler mode
        self._configure_session()
        
        # Setup adapters for connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def _configure_session(self):
        """Configure session based on crawler settings"""
        headers = self._get_base_headers()
        self.session.headers.update(headers)
        
        # Timeout configuration
        self.timeout = self.crawler.timeout
        
        # SSL verification (can be disabled for debugging)
        self.session.verify = not self.crawler.debug_mode
        
        # Redirect handling
        self.session.max_redirects = 5
        
        # Compression
        self.session.headers['Accept-Encoding'] = 'gzip, deflate, br'
    
    def _get_base_headers(self) -> Dict[str, str]:
        """Get base headers based on configuration"""
        base_headers = {
            'User-Agent': self._get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Add mobile headers if mobile emulation enabled
        if self.crawler.mobile_emulation:
            base_headers.update({
                'X-Requested-With': 'com.android.browser',
                'X-Mobile': 'true'
            })
        
        return base_headers
    
    def _get_user_agent(self) -> str:
        """Get appropriate user agent string"""
        agent_module = self.crawler.agent_module.lower()
        mobile = self.crawler.mobile_emulation
        
        user_agents = {
            'mozilla': {
                'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
                'mobile': 'Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/119.0 Firefox/119.0'
            },
            'chrome': {
                'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'mobile': 'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36'
            },
            'androidbrowser': {
                'mobile': 'Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36'
            }
        }
        
        # Get the appropriate user agent
        module_agents = user_agents.get(agent_module, user_agents['chrome'])
        return module_agents['mobile' if mobile else 'desktop']
    
    def fetch(self, 
              url: str, 
              strategy: Dict[str, Any],
              **kwargs) -> requests.Response:
        """
        🎯 MAIN FETCH METHOD - Smart HTTP Request Execution
        
        Args:
            url: Target URL
            strategy: Crawling strategy
            **kwargs: Additional parameters
        
        Returns:
            requests.Response: HTTP response object
        """
        start_time = time.time()
        
        try:
            # 1. Check cache first
            cache_key = self._generate_cache_key(url, strategy)
            cached_response = self.cache_manager.get(cache_key)
            
            if cached_response and not kwargs.get('force_fresh', False):
                if self.crawler.debug_mode:
                    print(f"   💾 Cache HIT: {url}")
                return self._create_response_from_cache(cached_response, url)
            
            if self.crawler.debug_mode:
                print(f"   🔄 Cache MISS: {url}")
            
            # 2. Prepare request
            request_args = self._prepare_request_args(strategy, kwargs)
            
            # 3. Execute with retry mechanism
            response = self._execute_with_retry(url, request_args, strategy)
            
            # 4. Validate response
            self._validate_response(response, url)
            
            # 5. Cache successful response
            if response.status_code == 200:
                self.cache_manager.set(cache_key, {
                    'content': response.text,
                    'headers': dict(response.headers),
                    'status_code': response.status_code,
                    'url': response.url,
                    'timestamp': time.time()
                })
            
            # 6. Record success
            self._record_request(True, time.time() - start_time)
            self.consecutive_failures = 0
            
            return response
            
        except Exception as e:
            # Record failure
            self._record_request(False, time.time() - start_time)
            self.consecutive_failures += 1
            
            if self.crawler.debug_mode:
                print(f"   ❌ HTTP Request failed: {e}")
            
            raise NetworkError(f"HTTP request failed: {e}") from e
    
    def _prepare_request_args(self, strategy: Dict, kwargs: Dict) -> Dict:
        """Prepare request arguments based on strategy"""
        args = {
            'timeout': self.timeout,
            'allow_redirects': True,
            'stream': False  # For memory efficiency
        }
        
        # Apply strategy-specific settings
        if strategy.get('engine') == 'http':
            if strategy.get('method') == 'aggressive':
                args['timeout'] = 10  # Shorter timeout for aggressive mode
            elif strategy.get('method') == 'stealth':
                args['timeout'] = 30  # Longer timeout for stealth
        
        # Apply kwargs overrides
        args.update(kwargs)
        
        return args
    
    def _execute_with_retry(self, 
                           url: str, 
                           request_args: Dict, 
                           strategy: Dict) -> requests.Response:
        """Execute request with intelligent retry mechanism"""
        max_retries = strategy.get('retry_attempts', self.crawler.max_retries)
        retry_delays = [1, 2, 4, 8]  # Exponential backoff
        
        for attempt in range(max_retries + 1):
            try:
                # Add random delay for retries (except first attempt)
                if attempt > 0:
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    if self.crawler.debug_mode:
                        print(f"   🔄 Retry {attempt}/{max_retries} after {delay}s...")
                    time.sleep(delay)
                
                # Execute request
                response = self.session.get(url, **request_args)
                return response
                
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    raise NetworkError(f"Timeout after {max_retries} retries")
                continue
                
            except requests.exceptions.ConnectionError:
                if attempt == max_retries:
                    raise NetworkError("Connection failed after retries")
                continue
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise NetworkError(f"Request failed: {e}")
                continue
    
    def _validate_response(self, response: requests.Response, url: str):
        """Validate HTTP response for common issues"""
        
        # Check status code
        if response.status_code >= 400:
            if response.status_code == 403:
                raise CrawlerBlockedError(f"Access forbidden: {url}")
            elif response.status_code == 429:
                raise CrawlerBlockedError(f"Rate limited: {url}")
            elif response.status_code == 503:
                raise NetworkError(f"Service unavailable: {url}")
            else:
                raise NetworkError(f"HTTP {response.status_code}: {url}")
        
        # Check for blocking indicators in content
        content_lower = response.text.lower()
        blocking_indicators = [
            'access denied', 'captcha', 'cloudflare',
            'bot detected', 'automated traffic', '403 forbidden'
        ]
        
        for indicator in blocking_indicators:
            if indicator in content_lower:
                raise CrawlerBlockedError(f"Blocking detected: {indicator}")
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/json' not in content_type:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Unexpected content type: {content_type}")
    
    def _generate_cache_key(self, url: str, strategy: Dict) -> str:
        """Generate unique cache key for request"""
        key_data = {
            'url': url,
            'user_agent': self.session.headers.get('User-Agent', ''),
            'mode': self.crawler.mode,
            'strategy': strategy.get('method', 'default')
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _create_response_from_cache(self, cached_data: Dict, url: str) -> requests.Response:
        """Create Response object from cached data"""
        response = requests.Response()
        response.status_code = cached_data['status_code']
        response.headers = cached_data['headers']
        response._content = cached_data['content'].encode('utf-8')
        response.url = url
        response.encoding = 'utf-8'
        
        return response
    
    def _record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.request_history.append({
            'timestamp': time.time(),
            'success': success,
            'response_time': response_time
        })
        
        # Keep only last 100 requests in history
        if len(self.request_history) > 100:
            self.request_history.pop(0)
    
    def batch_fetch(self, 
                   urls: List[str], 
                   strategy: Dict,
                   max_concurrent: int = 5) -> List[requests.Response]:
        """
        🔥 BATCH FETCH - Process multiple URLs concurrently
        
        Args:
            urls: List of URLs to fetch
            strategy: Crawling strategy
            max_concurrent: Maximum concurrent requests
        
        Returns:
            List of responses
        """
        from concurrent.futures import ThreadPoolExecutor
        
        def fetch_single(url):
            return self.fetch(url, strategy)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(fetch_single, url) for url in urls]
            results = []
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error response
                    error_response = requests.Response()
                    error_response.status_code = 500
                    error_response.error = str(e)
                    results.append(error_response)
            
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP client statistics"""
        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.request_history if r['success'])
        avg_response_time = (
            sum(r['response_time'] for r in self.request_history) / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'average_response_time': round(avg_response_time, 2),
            'consecutive_failures': self.consecutive_failures,
            'cache_hit_rate': self.cache_manager.get_hit_rate(),
            'current_session': self.session is not None
        }
    
    def clear_cache(self):
        """Clear HTTP cache"""
        self.cache_manager.clear()
        if self.crawler.debug_mode:
            print("💾 HTTP cache cleared")
    
    def close(self):
        """Close HTTP session and cleanup"""
        if self.session:
            self.session.close()
            self.session = None
        
        if self.cache_manager:
            self.cache_manager.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Advanced Caching System
class AdvancedCacheManager(CacheManager):
    """Enhanced cache manager with compression and TTL"""
    
    def __init__(self, crawler):
        super().__init__(crawler)
        self.compression_enabled = crawler.compression_enabled
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached response with decompression"""
        cached = super().get(key)
        
        if cached and self.compression_enabled:
            # Decompress if needed
            if cached.get('compressed'):
                cached['content'] = self._decompress(cached['content'])
        
        return cached
    
    def set(self, key: str, data: Dict, ttl: int = None):
        """Set cache with optional compression"""
        if ttl is None:
            ttl = self.default_ttl
        
        if self.compression_enabled and len(data['content']) > 1024:  # Compress large content
            data['content'] = self._compress(data['content'])
            data['compressed'] = True
        
        data['expires_at'] = time.time() + ttl
        super().set(key, data)
    
    def _compress(self, text: str) -> str:
        """Compress text using gzip"""
        import gzip
        return gzip.compress(text.encode('utf-8'))
    
    def _decompress(self, compressed_data: bytes) -> str:
        """Decompress gzip data"""
        import gzip
        return gzip.decompress(compressed_data).decode('utf-8')


# Performance Decorators
def timed_request(func):
    """Decorator to time HTTP requests"""
    @wraps(func)
    def wrapper(self, url, *args, **kwargs):
        start_time = time.time()
        result = func(self, url, *args, **kwargs)
        response_time = time.time() - start_time
        
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.record_request(True, response_time)
        
        return result
    return wrapper


def retry_on_failure(max_retries=3):
    """Decorator for automatic retry on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator


print("✅ HTTP Client loaded successfully!")
