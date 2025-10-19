"""
🔄 ROTATION MANAGER - Advanced Identity & Resource Rotation for CoffeCrawler
Revolutionary rotation system with intelligent IP, user agent, fingerprint management, and active defense mechanisms
"""

import random
import time
import hashlib
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque, Counter
import threading
import base64
import os
import ipaddress
from datetime import datetime, timedelta
import secrets
import string
from cryptography.fernet import Fernet
import socket
import urllib3
from fake_useragent import UserAgent
import numpy as np
from scipy import stats

from ..exceptions import RotationError, ProxyError, IdentityError, SecurityError


@dataclass
class RotationIdentity:
    """Advanced rotation identity container with active capabilities"""
    id: str
    user_agent: str
    ip_address: str
    proxy_config: Dict[str, Any]
    browser_fingerprint: Dict[str, Any]
    headers: Dict[str, str]
    cookies: Dict[str, str]
    ssl_cipher: str
    tls_version: str
    http2_support: bool
    created_at: float
    last_used: float
    usage_count: int = 0
    success_rate: float = 1.0
    ban_status: bool = False
    risk_level: float = 0.0
    geographic_data: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_flags: Set[str] = field(default_factory=set)


@dataclass
class ProxyServer:
    """Advanced proxy server with active monitoring"""
    id: str
    host: str
    port: int
    protocol: str
    username: Optional[str] = None
    password: Optional[str] = None
    country: str = "Unknown"
    city: str = "Unknown"
    provider: str = "Unknown"
    speed: float = 0.0  # ms
    reliability: float = 1.0
    anonymity_level: str = "transparent"  # transparent, anonymous, elite
    last_checked: float = field(default_factory=time.time)
    ban_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    active: bool = True
    premium: bool = False
    bandwidth_used: float = 0.0
    bandwidth_limit: Optional[float] = None
    security_score: float = 0.0


@dataclass
class UserAgentProfile:
    """Advanced user agent profile with behavioral patterns"""
    user_agent: str
    browser: str
    version: str
    platform: str
    device_type: str
    language: str
    accept_header: str
    screen_resolution: str
    hardware_concurrency: int
    device_memory: int
    timezone: str
    plugins: List[str]
    fonts: List[str]
    canvas_fingerprint: str
    webgl_fingerprint: str
    audio_fingerprint: str


@dataclass
class RotationStrategy:
    """Intelligent rotation strategy configuration"""
    name: str
    description: str
    rotation_trigger: str  # request_count, time_interval, ban_detected, performance
    rotation_interval: int
    max_usage_per_identity: int
    risk_tolerance: float
    performance_threshold: float
    geographic_diversity: bool
    platform_diversity: bool
    browser_diversity: bool


class RotationManager:
    """
    🔄 ADVANCED ROTATION MANAGER - Revolutionary Identity & Resource Management
    
    Features:
    - Intelligent IP rotation with multi-protocol proxy pools
    - Dynamic user agent generation with realistic fingerprints
    - Active ban detection and evasion systems
    - Geographic and platform diversity simulation
    - Real-time performance optimization
    - Cryptographic identity protection
    - Multi-threaded proxy validation
    - Machine learning-based risk assessment
    - Bandwidth management and throttling
    - Automatic recovery and fallback systems
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.manager_id = self._generate_manager_id()
        
        # Core identity management
        self.active_identities: Dict[str, RotationIdentity] = {}
        self.identity_pool = deque()
        self.identity_history = deque(maxlen=5000)
        self.identity_factory = IdentityFactory()
        
        # Advanced proxy management
        self.proxy_pool: List[ProxyServer] = []
        self.proxy_blacklist: Set[str] = set()
        self.proxy_performance = defaultdict(list)
        self.proxy_validator = ProxyValidator()
        self.proxy_rotator = ProxyRotator()
        
        # User agent management
        self.user_agent_pool: List[UserAgentProfile] = []
        self.user_agent_rotator = UserAgentRotator()
        self.fingerprint_generator = FingerprintGenerator()
        
        # Rotation strategies
        self.rotation_strategies: Dict[str, RotationStrategy] = {}
        self.active_strategy: Optional[RotationStrategy] = None
        
        # Security systems
        self.ban_detector = BanDetector()
        self.risk_assessor = RiskAssessor()
        self.security_monitor = SecurityMonitor()
        
        # Performance systems
        self.performance_optimizer = PerformanceOptimizer()
        self.load_balancer = LoadBalancer()
        
        # Configuration
        self.rotation_enabled = True
        self.auto_recovery_enabled = True
        self.performance_optimization_enabled = True
        self.security_monitoring_enabled = True
        
        # Statistics and monitoring
        self.rotation_stats = RotationStats()
        self.performance_monitor = PerformanceMonitor()
        self.security_auditor = SecurityAuditor()
        
        # Background tasks
        self.background_tasks = []
        self.maintenance_thread = None
        self.is_running = True
        
        # Initialize all systems
        self._initialize_all_systems()
        
        # Start background maintenance
        self._start_background_maintenance()
        
        if crawler.debug_mode:
            print(f"🔄 Advanced Rotation Manager {self.manager_id} initialized")
            print(f"   Active Systems: {len(self.proxy_pool)} proxies, {len(self.user_agent_pool)} user agents")
            print(f"   Security: {self.security_monitoring_enabled} | Optimization: {self.performance_optimization_enabled}")
    
    def _generate_manager_id(self) -> str:
        """Generate cryptographically secure manager ID"""
        random_bytes = secrets.token_bytes(16)
        return hashlib.sha256(random_bytes).hexdigest()[:12]
    
    def _initialize_all_systems(self):
        """Initialize all rotation systems"""
        try:
            # Initialize rotation strategies
            self._initialize_rotation_strategies()
            
            # Initialize proxy systems
            self._initialize_proxy_systems()
            
            # Initialize user agent systems
            self._initialize_user_agent_systems()
            
            # Initialize identity systems
            self._initialize_identity_systems()
            
            # Initialize security systems
            self._initialize_security_systems()
            
            # Initialize performance systems
            self._initialize_performance_systems()
            
            if self.crawler.debug_mode:
                print("   ✅ All rotation systems initialized successfully")
                
        except Exception as e:
            raise RotationError(f"System initialization failed: {e}") from e
    
    def _initialize_rotation_strategies(self):
        """Initialize intelligent rotation strategies"""
        self.rotation_strategies = {
            'stealth': RotationStrategy(
                name='stealth',
                description='Maximum stealth with frequent rotations',
                rotation_trigger='request_count',
                rotation_interval=5,
                max_usage_per_identity=10,
                risk_tolerance=0.1,
                performance_threshold=0.8,
                geographic_diversity=True,
                platform_diversity=True,
                browser_diversity=True
            ),
            'aggressive': RotationStrategy(
                name='aggressive',
                description='Maximum performance with minimal rotation',
                rotation_trigger='ban_detected',
                rotation_interval=100,
                max_usage_per_identity=100,
                risk_tolerance=0.8,
                performance_threshold=0.6,
                geographic_diversity=False,
                platform_diversity=False,
                browser_diversity=False
            ),
            'intelligent': RotationStrategy(
                name='intelligent',
                description='AI-powered adaptive rotation',
                rotation_trigger='performance',
                rotation_interval=25,
                max_usage_per_identity=50,
                risk_tolerance=0.3,
                performance_threshold=0.9,
                geographic_diversity=True,
                platform_diversity=True,
                browser_diversity=True
            ),
            'balanced': RotationStrategy(
                name='balanced',
                description='Balanced performance and security',
                rotation_trigger='time_interval',
                rotation_interval=30,
                max_usage_per_identity=30,
                risk_tolerance=0.5,
                performance_threshold=0.7,
                geographic_diversity=True,
                platform_diversity=False,
                browser_diversity=True
            )
        }
        
        # Set default strategy based on crawler mode
        crawler_mode = getattr(self.crawler, 'agent_type', 'adaptive')
        strategy_name = 'intelligent' if crawler_mode in ['adaptive', 'intelligent'] else crawler_mode
        self.active_strategy = self.rotation_strategies.get(strategy_name, self.rotation_strategies['balanced'])
    
    def _initialize_proxy_systems(self):
        """Initialize advanced proxy management systems"""
        # Load proxies from multiple sources
        proxy_sources = [
            self._load_proxies_from_file(),
            self._load_proxies_from_env(),
            self._generate_residential_proxies(),
            self._load_premium_proxies()
        ]
        
        for proxy_list in proxy_sources:
            self.proxy_pool.extend(proxy_list)
        
        # Initialize proxy validator
        self.proxy_validator.initialize(self.proxy_pool)
        
        # Validate proxies in background
        self._start_proxy_validation()
        
        if self.crawler.debug_mode:
            print(f"   🔌 Loaded {len(self.proxy_pool)} proxies from {len(proxy_sources)} sources")
    
    def _initialize_user_agent_systems(self):
        """Initialize advanced user agent systems"""
        # Load user agents from multiple sources
        ua_sources = [
            self._load_user_agents_from_file(),
            self._generate_dynamic_user_agents(),
            self._load_mobile_user_agents(),
            self._load_custom_user_agents()
        ]
        
        for ua_list in ua_sources:
            self.user_agent_pool.extend(ua_list)
        
        # Initialize fingerprint generator
        self.fingerprint_generator.initialize()
        
        if self.crawler.debug_mode:
            print(f"   🕵️ Loaded {len(self.user_agent_pool)} user agent profiles")
    
    def _initialize_identity_systems(self):
        """Initialize advanced identity management systems"""
        # Create initial identity pool
        initial_identities = self._generate_initial_identities(50)
        self.identity_pool.extend(initial_identities)
        
        # Initialize identity factory
        self.identity_factory.initialize(
            proxy_pool=self.proxy_pool,
            user_agent_pool=self.user_agent_pool,
            fingerprint_generator=self.fingerprint_generator
        )
    
    def _initialize_security_systems(self):
        """Initialize advanced security systems"""
        self.ban_detector.initialize()
        self.risk_assessor.initialize()
        self.security_monitor.initialize()
    
    def _initialize_performance_systems(self):
        """Initialize advanced performance systems"""
        self.performance_optimizer.initialize()
        self.load_balancer.initialize(self.proxy_pool)
    
    def _load_proxies_from_file(self) -> List[ProxyServer]:
        """Load proxies from configuration files"""
        proxies = []
        config_paths = [
            'proxies.json',
            'config/proxies.json',
            '/etc/coffecrawler/proxies.json'
        ]
        
        for path in config_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        proxy_data = json.load(f)
                    
                    for proxy_info in proxy_data:
                        proxy = ProxyServer(
                            id=self._generate_proxy_id(proxy_info),
                            host=proxy_info.get('host', ''),
                            port=proxy_info.get('port', 8080),
                            protocol=proxy_info.get('protocol', 'http'),
                            username=proxy_info.get('username'),
                            password=proxy_info.get('password'),
                            country=proxy_info.get('country', 'Unknown'),
                            city=proxy_info.get('city', 'Unknown'),
                            provider=proxy_info.get('provider', 'Unknown'),
                            anonymity_level=proxy_info.get('anonymity', 'transparent'),
                            premium=proxy_info.get('premium', False)
                        )
                        proxies.append(proxy)
                        
            except Exception as e:
                if self.crawler.debug_mode:
                    print(f"   ⚠️ Failed to load proxies from {path}: {e}")
        
        return proxies
    
    def _load_proxies_from_env(self) -> List[ProxyServer]:
        """Load proxies from environment variables"""
        proxies = []
        
        # Check for proxy environment variables
        env_proxies = [
            os.getenv('HTTP_PROXY'),
            os.getenv('HTTPS_PROXY'),
            os.getenv('SOCKS_PROXY'),
            os.getenv('COFFEECRAWLER_PROXIES')
        ]
        
        for env_proxy in env_proxies:
            if env_proxy:
                try:
                    parsed = urlparse(env_proxy)
                    proxy = ProxyServer(
                        id=self._generate_proxy_id({'host': parsed.hostname, 'port': parsed.port}),
                        host=parsed.hostname,
                        port=parsed.port or 8080,
                        protocol=parsed.scheme,
                        username=parsed.username,
                        password=parsed.password
                    )
                    proxies.append(proxy)
                except Exception as e:
                    if self.crawler.debug_mode:
                        print(f"   ⚠️ Failed to parse env proxy {env_proxy}: {e}")
        
        return proxies
    
    def _generate_residential_proxies(self) -> List[ProxyServer]:
        """Generate residential-like proxy configurations"""
        residential_proxies = []
        
        # Simulate residential IP ranges (in production, this would use real residential proxies)
        residential_ranges = [
            "192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"  # Private ranges for simulation
        ]
        
        for ip_range in residential_ranges:
            network = ipaddress.ip_network(ip_range, strict=False)
            for _ in range(5):  # Generate 5 proxies per range
                ip = str(network[random.randint(1, network.num_addresses - 2)])
                proxy = ProxyServer(
                    id=self._generate_proxy_id({'host': ip}),
                    host=ip,
                    port=random.randint(8000, 9000),
                    protocol=random.choice(['http', 'socks5']),
                    country="Residential",
                    city="Simulated",
                    provider="Residential Network",
                    anonymity_level="elite",
                    premium=True
                )
                residential_proxies.append(proxy)
        
        return residential_proxies
    
    def _load_premium_proxies(self) -> List[ProxyServer]:
        """Load premium proxy configurations"""
        # Placeholder for premium proxy integration
        # In production, this would integrate with premium proxy services
        return []
    
    def _load_user_agents_from_file(self) -> List[UserAgentProfile]:
        """Load user agents from configuration files"""
        user_agents = []
        
        try:
            # Use fake-useragent library as base
            ua_generator = UserAgent()
            
            # Generate diverse user agent profiles
            for _ in range(50):
                profile = self.user_agent_rotator.generate_realistic_profile(ua_generator)
                user_agents.append(profile)
                
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Failed to load user agents: {e}")
            
            # Fallback to basic user agents
            user_agents.extend(self._generate_basic_user_agents())
        
        return user_agents
    
    def _generate_dynamic_user_agents(self) -> List[UserAgentProfile]:
        """Generate dynamic user agent profiles"""
        profiles = []
        
        browsers = ['chrome', 'firefox', 'safari', 'edge']
        platforms = ['windows', 'mac', 'linux', 'android', 'ios']
        
        for _ in range(25):
            browser = random.choice(browsers)
            platform = random.choice(platforms)
            
            profile = self.user_agent_rotator.generate_custom_profile(browser, platform)
            profiles.append(profile)
        
        return profiles
    
    def _load_mobile_user_agents(self) -> List[UserAgentProfile]:
        """Load mobile-specific user agent profiles"""
        mobile_profiles = []
        
        mobile_devices = [
            ('android', 'mobile'),
            ('ios', 'mobile'),
            ('android', 'tablet'),
            ('ios', 'tablet')
        ]
        
        for os_type, device_type in mobile_devices:
            for _ in range(10):
                profile = self.user_agent_rotator.generate_mobile_profile(os_type, device_type)
                mobile_profiles.append(profile)
        
        return mobile_profiles
    
    def _load_custom_user_agents(self) -> List[UserAgentProfile]:
        """Load custom user agent profiles"""
        custom_profiles = []
        
        # Add some custom enterprise browser profiles
        enterprise_browsers = [
            ('chrome', 'enterprise', 'windows'),
            ('firefox', 'enterprise', 'windows'),
            ('safari', 'enterprise', 'mac'),
            ('edge', 'enterprise', 'windows')
        ]
        
        for browser, environment, platform in enterprise_browsers:
            profile = self.user_agent_rotator.generate_enterprise_profile(browser, environment, platform)
            custom_profiles.append(profile)
        
        return custom_profiles
    
    def _generate_basic_user_agents(self) -> List[UserAgentProfile]:
        """Generate basic user agent profiles as fallback"""
        basic_profiles = []
        
        basic_configs = [
            {
                'browser': 'chrome',
                'version': '119.0.0.0',
                'platform': 'windows',
                'device_type': 'desktop'
            },
            {
                'browser': 'firefox',
                'version': '119.0',
                'platform': 'windows', 
                'device_type': 'desktop'
            },
            {
                'browser': 'safari',
                'version': '17.0',
                'platform': 'mac',
                'device_type': 'desktop'
            },
            {
                'browser': 'edge',
                'version': '119.0.0.0',
                'platform': 'windows',
                'device_type': 'desktop'
            }
        ]
        
        for config in basic_configs:
            profile = UserAgentProfile(
                user_agent=self.user_agent_rotator.construct_user_agent(config),
                browser=config['browser'],
                version=config['version'],
                platform=config['platform'],
                device_type=config['device_type'],
                language='en-US',
                accept_header='text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                screen_resolution='1920x1080',
                hardware_concurrency=8,
                device_memory=8,
                timezone='America/New_York',
                plugins=['PDF Viewer', 'Chrome PDF Viewer', 'Chromium PDF Viewer'],
                fonts=['Arial', 'Times New Roman', 'Courier New'],
                canvas_fingerprint='basic_canvas',
                webgl_fingerprint='basic_webgl',
                audio_fingerprint='basic_audio'
            )
            basic_profiles.append(profile)
        
        return basic_profiles
    
    def _generate_initial_identities(self, count: int) -> List[RotationIdentity]:
        """Generate initial rotation identities"""
        identities = []
        
        for _ in range(count):
            identity = self.identity_factory.create_identity()
            identities.append(identity)
        
        return identities
    
    def _generate_proxy_id(self, proxy_info: Dict) -> str:
        """Generate unique proxy ID"""
        proxy_string = f"{proxy_info.get('host', '')}:{proxy_info.get('port', 0)}"
        return hashlib.md5(proxy_string.encode()).hexdigest()[:8]
    
    def _start_background_maintenance(self):
        """Start background maintenance tasks"""
        def maintenance_worker():
            while self.is_running:
                try:
                    self._perform_maintenance_tasks()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    if self.crawler.debug_mode:
                        print(f"   ⚠️ Maintenance task error: {e}")
        
        self.maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        self.maintenance_thread.start()
        
        if self.crawler.debug_mode:
            print("   🔧 Background maintenance started")
    
    def _start_proxy_validation(self):
        """Start background proxy validation"""
        def validation_worker():
            while self.is_running:
                try:
                    self.proxy_validator.validate_proxy_pool(self.proxy_pool)
                    time.sleep(300)  # Validate every 5 minutes
                except Exception as e:
                    if self.crawler.debug_mode:
                        print(f"   ⚠️ Proxy validation error: {e}")
        
        validation_thread = threading.Thread(target=validation_worker, daemon=True)
        validation_thread.start()
    
    def _perform_maintenance_tasks(self):
        """Perform regular maintenance tasks"""
        try:
            # Clean up old identities
            self._cleanup_old_identities()
            
            # Update performance metrics
            self.performance_optimizer.update_metrics()
            
            # Check proxy health
            self._check_proxy_health()
            
            # Rotate identities if needed
            self._perform_strategic_rotation()
            
            # Run security audit
            if self.security_monitoring_enabled:
                self.security_auditor.run_audit()
                
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Maintenance task failed: {e}")
    
    def _cleanup_old_identities(self):
        """Clean up old and low-performance identities"""
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        # Remove identities older than max_age
        self.identity_pool = deque([
            identity for identity in self.identity_pool
            if current_time - identity.created_at < max_age
        ])
        
        # Remove identities with low success rate
        self.identity_pool = deque([
            identity for identity in self.identity_pool
            if identity.success_rate > 0.3
        ])
    
    def _check_proxy_health(self):
        """Check and update proxy health status"""
        for proxy in self.proxy_pool:
            if not proxy.active:
                continue
                
            # Check if proxy should be deactivated
            if proxy.failure_count > 10 or proxy.reliability < 0.1:
                proxy.active = False
                self.proxy_blacklist.add(proxy.id)
    
    def _perform_strategic_rotation(self):
        """Perform strategic identity rotation based on current strategy"""
        if not self.active_strategy or not self.rotation_enabled:
            return
        
        # Check if rotation is needed based on strategy
        rotation_needed = self._evaluate_rotation_need()
        
        if rotation_needed:
            new_identities = self._generate_strategic_identities(10)
            self.identity_pool.extend(new_identities)
    
    def _evaluate_rotation_need(self) -> bool:
        """Evaluate if rotation is needed based on current strategy"""
        if not self.active_strategy:
            return False
        
        strategy = self.active_strategy
        
        # Check different rotation triggers
        if strategy.rotation_trigger == 'request_count':
            total_requests = self.rotation_stats.total_requests
            return total_requests % strategy.rotation_interval == 0
        
        elif strategy.rotation_trigger == 'time_interval':
            current_time = time.time()
            last_rotation = getattr(self, '_last_rotation_time', 0)
            return current_time - last_rotation > strategy.rotation_interval
        
        elif strategy.rotation_trigger == 'ban_detected':
            recent_bans = self.ban_detector.recent_ban_count
            return recent_bans > 0
        
        elif strategy.rotation_trigger == 'performance':
            avg_performance = self.performance_monitor.average_performance
            return avg_performance < strategy.performance_threshold
        
        return False
    
    def _generate_strategic_identities(self, count: int) -> List[RotationIdentity]:
        """Generate identities based on current strategy"""
        identities = []
        
        strategy = self.active_strategy
        
        for _ in range(count):
            identity = self.identity_factory.create_strategic_identity(strategy)
            identities.append(identity)
        
        return identities
    
    def get_identity(self, url: str, strategy: Dict) -> RotationIdentity:
        """
        🎯 GET INTELLIGENT IDENTITY - Advanced Identity Selection
        
        Args:
            url: Target URL
            strategy: Current crawling strategy
        
        Returns:
            RotationIdentity: Optimized identity for the request
        """
        start_time = time.time()
        
        try:
            # 1. Risk assessment for target URL
            risk_level = self.risk_assessor.assess_url_risk(url)
            
            # 2. Performance optimization
            performance_needs = self.performance_optimizer.assess_performance_needs(strategy)
            
            # 3. Security considerations
            security_requirements = self.security_monitor.get_security_requirements(url)
            
            # 4. Intelligent identity selection
            identity = self._select_optimal_identity(url, risk_level, performance_needs, security_requirements)
            
            # 5. Update identity usage
            identity.last_used = time.time()
            identity.usage_count += 1
            
            # 6. Record statistics
            self.rotation_stats.record_identity_usage(identity, url)
            
            # 7. Check if rotation is needed
            if self._should_rotate_identity(identity):
                self._rotate_identity(identity)
            
            if self.crawler.debug_mode:
                print(f"   🆔 Identity selected: {identity.id}")
                print(f"   📊 Usage: {identity.usage_count} | Success: {identity.success_rate:.2f}")
            
            return identity
            
        except Exception as e:
            raise RotationError(f"Identity selection failed: {e}") from e
    
    def _select_optimal_identity(self, url: str, risk_level: float, 
                               performance_needs: Dict, security_requirements: Dict) -> RotationIdentity:
        """Select optimal identity based on multiple factors"""
        
        # Filter available identities
        candidate_identities = [
            identity for identity in self.identity_pool
            if self._is_identity_suitable(identity, risk_level, performance_needs, security_requirements)
        ]
        
        if not candidate_identities:
            # Create new identity if no suitable ones found
            return self.identity_factory.create_identity()
        
        # Score identities based on multiple factors
        scored_identities = []
        for identity in candidate_identities:
            score = self._calculate_identity_score(identity, url, risk_level, performance_needs)
            scored_identities.append((identity, score))
        
        # Select best identity
        scored_identities.sort(key=lambda x: x[1], reverse=True)
        best_identity, best_score = scored_identities[0]
        
        return best_identity
    
    def _is_identity_suitable(self, identity: RotationIdentity, risk_level: float,
                            performance_needs: Dict, security_requirements: Dict) -> bool:
        """Check if identity is suitable for current requirements"""
        
        # Check ban status
        if identity.ban_status:
            return False
        
        # Check risk compatibility
        if identity.risk_level > risk_level + 0.2:
            return False
        
        # Check usage limits
        if identity.usage_count >= self.active_strategy.max_usage_per_identity:
            return False
        
        # Check performance requirements
        if (identity.performance_metrics.get('response_time', 0) > 
            performance_needs.get('max_response_time', 10.0)):
            return False
        
        # Check security requirements
        security_flags = security_requirements.get('required_flags', set())
        if not security_flags.issubset(identity.security_flags):
            return False
        
        return True
    
    def _calculate_identity_score(self, identity: RotationIdentity, url: str, 
                                risk_level: float, performance_needs: Dict) -> float:
        """Calculate comprehensive identity score"""
        score = 0.0
        
        # Success rate factor (40%)
        success_factor = identity.success_rate * 0.4
        
        # Performance factor (30%)
        performance_metrics = identity.performance_metrics
        response_time = performance_metrics.get('response_time', 5.0)
        performance_factor = max(0, 1.0 - (response_time / 10.0)) * 0.3
        
        # Risk compatibility factor (20%)
        risk_compatibility = max(0, 1.0 - abs(identity.risk_level - risk_level)) * 0.2
        
        # Freshness factor (10%)
        time_since_use = time.time() - identity.last_used
        freshness_factor = min(1.0, time_since_use / 3600) * 0.1
        
        score = success_factor + performance_factor + risk_compatibility + freshness_factor
        
        return score
    
    def _should_rotate_identity(self, identity: RotationIdentity) -> bool:
        """Determine if identity should be rotated"""
        if not self.rotation_enabled:
            return False
        
        strategy = self.active_strategy
        
        # Check usage count
        if identity.usage_count >= strategy.max_usage_per_identity:
            return True
        
        # Check success rate
        if identity.success_rate < strategy.performance_threshold:
            return True
        
        # Check ban status
        if identity.ban_status:
            return True
        
        # Check risk level
        if identity.risk_level > strategy.risk_tolerance:
            return True
        
        return False
    
    def _rotate_identity(self, identity: RotationIdentity):
        """Rotate to a new identity"""
        # Mark old identity for retirement
        identity.ban_status = True
        
        # Create new identity
        new_identity = self.identity_factory.create_identity()
        self.identity_pool.append(new_identity)
        
        if self.crawler.debug_mode:
            print(f"   🔄 Identity rotated: {identity.id} -> {new_identity.id}")
    
    def record_identity_performance(self, identity: RotationIdentity, success: bool, 
                                 response_time: float, ban_detected: bool = False):
        """
        📊 RECORD IDENTITY PERFORMANCE - Update identity metrics
        
        Args:
            identity: The identity used
            success: Whether the request was successful
            response_time: Request response time
            ban_detected: Whether a ban was detected
        """
        try:
            # Update success rate
            if success:
                identity.success_rate = (identity.success_rate * identity.usage_count + 1) / (identity.usage_count + 1)
            else:
                identity.success_rate = (identity.success_rate * identity.usage_count) / (identity.usage_count + 1)
            
            # Update performance metrics
            identity.performance_metrics['response_time'] = response_time
            identity.performance_metrics['last_response_time'] = response_time
            
            # Update ban status
            if ban_detected:
                identity.ban_status = True
                self.ban_detector.record_ban(identity)
            
            # Update risk assessment
            identity.risk_level = self.risk_assessor.calculate_identity_risk(identity)
            
            # Update rotation statistics
            self.rotation_stats.record_performance(identity, success, response_time, ban_detected)
            
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ Performance recording failed: {e}")
    
    def update_strategy(self, strategy_name: str):
        """
        🎯 UPDATE ROTATION STRATEGY - Change active rotation strategy
        
        Args:
            strategy_name: Name of the strategy to activate
        """
        if strategy_name in self.rotation_strategies:
            self.active_strategy = self.rotation_strategies[strategy_name]
            
            if self.crawler.debug_mode:
                print(f"   🔄 Rotation strategy changed to: {strategy_name}")
        else:
            raise RotationError(f"Unknown strategy: {strategy_name}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive rotation performance statistics"""
        return {
            'rotation_stats': self.rotation_stats.get_stats(),
            'performance_stats': self.performance_monitor.get_stats(),
            'security_stats': self.security_auditor.get_stats(),
            'identity_pool_size': len(self.identity_pool),
            'proxy_pool_size': len([p for p in self.proxy_pool if p.active]),
            'active_strategy': self.active_strategy.name if self.active_strategy else 'None'
        }
    
    def emergency_rotation(self):
        """🆘 EMERGENCY ROTATION - Immediate full identity rotation"""
        if self.crawler.debug_mode:
            print("   🚨 EMERGENCY ROTATION ACTIVATED!")
        
        # Clear current identity pool
        self.identity_pool.clear()
        
        # Generate new identities
        new_identities = self._generate_strategic_identities(20)
        self.identity_pool.extend(new_identities)
        
        # Reset statistics
        self.rotation_stats.reset_emergency()
        
        # Clear proxy blacklist
        self.proxy_blacklist.clear()
        
        if self.crawler.debug_mode:
            print("   ✅ Emergency rotation completed")
    
    def add_custom_proxy(self, proxy_config: Dict[str, Any]):
        """
        ➕ ADD CUSTOM PROXY - Add custom proxy to the pool
        
        Args:
            proxy_config: Proxy configuration dictionary
        """
        try:
            proxy = ProxyServer(
                id=self._generate_proxy_id(proxy_config),
                host=proxy_config['host'],
                port=proxy_config['port'],
                protocol=proxy_config.get('protocol', 'http'),
                username=proxy_config.get('username'),
                password=proxy_config.get('password'),
                country=proxy_config.get('country', 'Unknown'),
                city=proxy_config.get('city', 'Unknown'),
                provider=proxy_config.get('provider', 'Custom'),
                anonymity_level=proxy_config.get('anonymity', 'transparent'),
                premium=proxy_config.get('premium', False)
            )
            
            self.proxy_pool.append(proxy)
            
            if self.crawler.debug_mode:
                print(f"   ✅ Custom proxy added: {proxy.host}:{proxy.port}")
                
        except Exception as e:
            raise ProxyError(f"Failed to add custom proxy: {e}") from e
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current rotation configuration"""
        return {
            'manager_id': self.manager_id,
            'active_strategy': self.active_strategy.name if self.active_strategy else None,
            'rotation_enabled': self.rotation_enabled,
            'identity_pool_size': len(self.identity_pool),
            'proxy_pool_size': len(self.proxy_pool),
            'performance_stats': self.get_performance_stats(),
            'timestamp': time.time()
        }
    
    def shutdown(self):
        """🛑 SHUTDOWN ROTATION MANAGER - Clean shutdown"""
        self.is_running = False
        
        # Wait for maintenance thread to finish
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)
        
        # Clear all pools
        self.identity_pool.clear()
        self.proxy_pool.clear()
        self.user_agent_pool.clear()
        
        if self.crawler.debug_mode:
            print("   🛑 Rotation Manager shut down successfully")


# Advanced Supporting Classes

class IdentityFactory:
    """Advanced identity factory with cryptographic security"""
    
    def __init__(self):
        self.identity_counter = 0
        self.crypto_key = Fernet.generate_key()
        self.crypto_suite = Fernet(self.crypto_key)
    
    def initialize(self, proxy_pool: List[ProxyServer], user_agent_pool: List[UserAgentProfile],
                  fingerprint_generator: 'FingerprintGenerator'):
        """Initialize identity factory with resources"""
        self.proxy_pool = proxy_pool
        self.user_agent_pool = user_agent_pool
        self.fingerprint_generator = fingerprint_generator
    
    def create_identity(self) -> RotationIdentity:
        """Create a new rotation identity"""
        identity_id = self._generate_identity_id()
        
        # Select random proxy
        active_proxies = [p for p in self.proxy_pool if p.active]
        proxy = random.choice(active_proxies) if active_proxies else None
        
        # Select random user agent profile
        user_agent_profile = random.choice(self.user_agent_pool) if self.user_agent_pool else None
        
        # Generate browser fingerprint
        fingerprint = self.fingerprint_generator.generate_fingerprint()
        
        # Create identity
        identity = RotationIdentity(
            id=identity_id,
            user_agent=user_agent_profile.user_agent if user_agent_profile else '',
            ip_address=proxy.host if proxy else '127.0.0.1',
            proxy_config=self._create_proxy_config(proxy) if proxy else {},
            browser_fingerprint=fingerprint,
            headers=self._generate_headers(user_agent_profile) if user_agent_profile else {},
            cookies={},
            ssl_cipher=random.choice(['TLS_AES_128_GCM_SHA256', 'TLS_CHACHA20_POLY1305_SHA256']),
            tls_version=random.choice(['TLSv1.2', 'TLSv1.3']),
            http2_support=random.choice([True, False]),
            created_at=time.time(),
            last_used=time.time(),
            geographic_data=self._generate_geographic_data(proxy) if proxy else {},
            security_flags={'encrypted', 'verified'}
        )
        
        return identity
    
    def create_strategic_identity(self, strategy: RotationStrategy) -> RotationIdentity:
        """Create identity based on specific strategy"""
        identity = self.create_identity()
        
        # Apply strategy-specific modifications
        if strategy.geographic_diversity:
            identity.geographic_data = self._ensure_geographic_diversity()
        
        if strategy.platform_diversity:
            identity.browser_fingerprint = self._ensure_platform_diversity(identity.browser_fingerprint)
        
        return identity
    
    def _generate_identity_id(self) -> str:
        """Generate cryptographically secure identity ID"""
        self.identity_counter += 1
        random_data = secrets.token_bytes(16)
        counter_data = self.identity_counter.to_bytes(8, 'big')
        combined = random_data + counter_data
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def _create_proxy_config(self, proxy: ProxyServer) -> Dict[str, Any]:
        """Create proxy configuration dictionary"""
        config = {
            'host': proxy.host,
            'port': proxy.port,
            'protocol': proxy.protocol,
            'username': proxy.username,
            'password': proxy.password
        }
        
        # Encrypt sensitive data
        if proxy.username and proxy.password:
            sensitive_data = f"{proxy.username}:{proxy.password}"
            encrypted = self.crypto_suite.encrypt(sensitive_data.encode())
            config['auth_encrypted'] = base64.b64encode(encrypted).decode()
            config['username'] = None
            config['password'] = None
        
        return config
    
    def _generate_headers(self, user_agent_profile: UserAgentProfile) -> Dict[str, str]:
        """Generate realistic HTTP headers"""
        headers = {
            'User-Agent': user_agent_profile.user_agent,
            'Accept': user_agent_profile.accept_header,
            'Accept-Language': user_agent_profile.language,
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Add platform-specific headers
        if 'windows' in user_agent_profile.platform.lower():
            headers['Sec-CH-UA-Platform'] = '"Windows"'
        elif 'mac' in user_agent_profile.platform.lower():
            headers['Sec-CH-UA-Platform'] = '"macOS"'
        
        return headers
    
    def _generate_geographic_data(self, proxy: ProxyServer) -> Dict[str, str]:
        """Generate geographic data based on proxy"""
        return {
            'country': proxy.country,
            'city': proxy.city,
            'timezone': self._get_timezone_for_country(proxy.country),
            'locale': self._get_locale_for_country(proxy.country)
        }
    
    def _get_timezone_for_country(self, country: str) -> str:
        """Get timezone for country"""
        timezone_map = {
            'US': 'America/New_York',
            'GB': 'Europe/London',
            'DE': 'Europe/Berlin',
            'JP': 'Asia/Tokyo',
            'IN': 'Asia/Kolkata',
            'BR': 'America/Sao_Paulo'
        }
        return timezone_map.get(country, 'UTC')
    
    def _get_locale_for_country(self, country: str) -> str:
        """Get locale for country"""
        locale_map = {
            'US': 'en-US',
            'GB': 'en-GB',
            'DE': 'de-DE',
            'FR': 'fr-FR',
            'JP': 'ja-JP',
            'CN': 'zh-CN'
        }
        return locale_map.get(country, 'en-US')
    
    def _ensure_geographic_diversity(self) -> Dict[str, str]:
        """Ensure geographic diversity in identity"""
        countries = ['US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU', 'BR', 'IN', 'SG']
        country = random.choice(countries)
        
        return {
            'country': country,
            'city': 'Unknown',
            'timezone': self._get_timezone_for_country(country),
            'locale': self._get_locale_for_country(country)
        }
    
    def _ensure_platform_diversity(self, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure platform diversity in browser fingerprint"""
        platforms = ['windows', 'mac', 'linux', 'android', 'ios']
        new_platform = random.choice(platforms)
        
        fingerprint['platform'] = new_platform
        fingerprint['user_agent'] = fingerprint['user_agent'].replace(
            fingerprint.get('platform', 'windows'), new_platform
        )
        
        return fingerprint


class ProxyValidator:
    """Advanced proxy validation with multi-protocol support"""
    
    def __init__(self):
        self.validation_results = {}
        self.test_urls = [
            'http://httpbin.org/ip',
            'https://httpbin.org/ip',
            'http://api.ipify.org',
            'https://api.ipify.org'
        ]
    
    def initialize(self, proxy_pool: List[ProxyServer]):
        """Initialize validator with proxy pool"""
        self.proxy_pool = proxy_pool
    
    def validate_proxy_pool(self, proxy_pool: List[ProxyServer]):
        """Validate entire proxy pool"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for proxy in proxy_pool:
                if proxy.active:
                    future = executor.submit(self.validate_proxy, proxy)
                    futures.append((proxy, future))
            
            for proxy, future in futures:
                try:
                    result = future.result(timeout=10)
                    self._update_proxy_status(proxy, result)
                except Exception as e:
                    proxy.failure_count += 1
                    proxy.reliability = max(0, proxy.reliability - 0.1)
    
    def validate_proxy(self, proxy: ProxyServer) -> Dict[str, Any]:
        """Validate single proxy"""
        try:
            start_time = time.time()
            
            # Test different protocols
            if proxy.protocol in ['http', 'https']:
                result = self._test_http_proxy(proxy)
            elif proxy.protocol.startswith('socks'):
                result = self._test_socks_proxy(proxy)
            else:
                result = {'success': False, 'error': 'Unsupported protocol'}
            
            # Calculate speed
            result['response_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'response_time': 0}
    
    def _test_http_proxy(self, proxy: ProxyServer) -> Dict[str, Any]:
        """Test HTTP/HTTPS proxy"""
        proxies = {
            'http': f"{proxy.protocol}://{proxy.host}:{proxy.port}",
            'https': f"{proxy.protocol}://{proxy.host}:{proxy.port}"
        }
        
        # Add authentication if provided
        if proxy.username and proxy.password:
            proxies['http'] = f"{proxy.protocol}://{proxy.username}:{proxy.password}@{proxy.host}:{proxy.port}"
            proxies['https'] = f"{proxy.protocol}://{proxy.username}:{proxy.password}@{proxy.host}:{proxy.port}"
        
        try:
            response = requests.get(
                'http://httpbin.org/ip',
                proxies=proxies,
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'external_ip': response.json().get('origin', 'Unknown'),
                    'anonymity': self._check_anonymity(response)
                }
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_socks_proxy(self, proxy: ProxyServer) -> Dict[str, Any]:
        """Test SOCKS proxy"""
        try:
            import socks
            import socket
            
            # Set up SOCKS proxy
            socks.set_default_proxy(
                getattr(socks, f"SOCKS{proxy.protocol[-1]}"),
                proxy.host,
                proxy.port,
                username=proxy.username,
                password=proxy.password
            )
            socket.socket = socks.socksocket
            
            # Test connection
            response = requests.get(
                'http://httpbin.org/ip',
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'external_ip': response.json().get('origin', 'Unknown'),
                    'anonymity': 'elite'  # SOCKS proxies are generally more anonymous
                }
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            # Reset socket to default
            socks.set_default_proxy()
            socket.socket = socket._socketobject
    
    def _check_anonymity(self, response) -> str:
        """Check proxy anonymity level"""
        headers = response.headers
        
        # Check for proxy headers
        proxy_headers = ['via', 'x-forwarded-for', 'x-real-ip', 'x-proxy-id']
        if any(header in headers for header in proxy_headers):
            return 'transparent'
        
        # Check if origin IP is revealed
        origin = response.json().get('origin', '')
        if ',' in origin:  # Multiple IPs indicate proxy chain
            return 'anonymous'
        
        return 'elite'
    
    def _update_proxy_status(self, proxy: ProxyServer, result: Dict[str, Any]):
        """Update proxy status based on validation result"""
        if result['success']:
            proxy.success_count += 1
            proxy.reliability = min(1.0, proxy.reliability + 0.05)
            proxy.speed = result.get('response_time', proxy.speed)
            proxy.anonymity_level = result.get('anonymity', proxy.anonymity_level)
            proxy.last_checked = time.time()
        else:
            proxy.failure_count += 1
            proxy.reliability = max(0, proxy.reliability - 0.1)
            
            # Deactivate if too many failures
            if proxy.failure_count > 5:
                proxy.active = False


class UserAgentRotator:
    """Advanced user agent rotation and generation"""
    
    def __init__(self):
        self.ua_generator = UserAgent()
        self.platform_templates = self._load_platform_templates()
    
    def generate_realistic_profile(self, ua_generator: UserAgent) -> UserAgentProfile:
        """Generate realistic user agent profile"""
        user_agent = ua_generator.random
        browser, version, platform = self._parse_user_agent(user_agent)
        
        return UserAgentProfile(
            user_agent=user_agent,
            browser=browser,
            version=version,
            platform=platform,
            device_type=self._determine_device_type(platform),
            language='en-US',
            accept_header='text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            screen_resolution=self._generate_screen_resolution(platform),
            hardware_concurrency=random.randint(2, 16),
            device_memory=random.choice([4, 8, 16, 32]),
            timezone=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo', 'UTC']),
            plugins=self._generate_plugins(browser),
            fonts=self._generate_fonts(platform),
            canvas_fingerprint=secrets.token_hex(16),
            webgl_fingerprint=secrets.token_hex(16),
            audio_fingerprint=secrets.token_hex(16)
        )
    
    def generate_custom_profile(self, browser: str, platform: str) -> UserAgentProfile:
        """Generate custom user agent profile"""
        template = self.platform_templates.get(f"{browser}_{platform}", {})
        
        user_agent = template.get('user_agent', '')
        if not user_agent:
            user_agent = self._construct_user_agent({'browser': browser, 'platform': platform})
        
        return UserAgentProfile(
            user_agent=user_agent,
            browser=browser,
            version=template.get('version', '1.0'),
            platform=platform,
            device_type=self._determine_device_type(platform),
            language=template.get('language', 'en-US'),
            accept_header=template.get('accept_header', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
            screen_resolution=self._generate_screen_resolution(platform),
            hardware_concurrency=random.randint(2, 16),
            device_memory=random.choice([4, 8, 16, 32]),
            timezone=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo', 'UTC']),
            plugins=self._generate_plugins(browser),
            fonts=self._generate_fonts(platform),
            canvas_fingerprint=secrets.token_hex(16),
            webgl_fingerprint=secrets.token_hex(16),
            audio_fingerprint=secrets.token_hex(16)
        )
    
    def generate_mobile_profile(self, os_type: str, device_type: str) -> UserAgentProfile:
        """Generate mobile user agent profile"""
        mobile_templates = {
            'android_mobile': {
                'user_agent': 'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36',
                'screen_resolution': '1080x2340',
                'hardware_concurrency': 8,
                'device_memory': 6
            },
            'ios_mobile': {
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
                'screen_resolution': '1170x2532',
                'hardware_concurrency': 6,
                'device_memory': 4
            }
        }
        
        template_key = f"{os_type}_{device_type}"
        template = mobile_templates.get(template_key, {})
        
        return UserAgentProfile(
            user_agent=template.get('user_agent', ''),
            browser='chrome' if os_type == 'android' else 'safari',
            version='119.0.0.0' if os_type == 'android' else '14.0',
            platform=os_type,
            device_type=device_type,
            language='en-US',
            accept_header='text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            screen_resolution=template.get('screen_resolution', '1080x1920'),
            hardware_concurrency=template.get('hardware_concurrency', 4),
            device_memory=template.get('device_memory', 4),
            timezone=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo', 'UTC']),
            plugins=[],
            fonts=['Roboto', 'Arial'] if os_type == 'android' else ['San Francisco', 'Helvetica Neue'],
            canvas_fingerprint=secrets.token_hex(16),
            webgl_fingerprint=secrets.token_hex(16),
            audio_fingerprint=secrets.token_hex(16)
        )
    
    def generate_enterprise_profile(self, browser: str, environment: str, platform: str) -> UserAgentProfile:
        """Generate enterprise user agent profile"""
        enterprise_templates = {
            'chrome_enterprise_windows': {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
                'plugins': ['Chrome PDF Viewer', 'Chrome PDF Plugin', 'Native Client']
            },
            'firefox_enterprise_windows': {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
                'plugins': ['PDF Viewer', 'Google Talk Plugin']
            }
        }
        
        template_key = f"{browser}_{environment}_{platform}"
        template = enterprise_templates.get(template_key, {})
        
        return self.generate_custom_profile(browser, platform)
    
    def construct_user_agent(self, config: Dict) -> str:
        """Construct user agent string from configuration"""
        browser = config.get('browser', 'chrome')
        platform = config.get('platform', 'windows')
        device_type = config.get('device_type', 'desktop')
        
        if device_type == 'mobile':
            if browser == 'chrome':
                return f'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36'
            else:  # safari
                return f'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
        else:
            if browser == 'chrome':
                return f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            elif browser == 'firefox':
                return f'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
            elif browser == 'safari':
                return f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            else:  # edge
                return f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
    
    def _load_platform_templates(self) -> Dict[str, Any]:
        """Load platform-specific templates"""
        return {
            'chrome_windows': {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'version': '119.0.0.0',
                'language': 'en-US',
                'accept_header': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
            },
            'firefox_windows': {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
                'version': '119.0',
                'language': 'en-US',
                'accept_header': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
            },
            'safari_mac': {
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                'version': '17.0',
                'language': 'en-US',
                'accept_header': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
        }
    
    def _parse_user_agent(self, user_agent: str) -> Tuple[str, str, str]:
        """Parse user agent string into components"""
        # Simple parsing - in production would use more sophisticated parsing
        if 'Chrome' in user_agent:
            browser = 'chrome'
            version = '119.0.0.0'
        elif 'Firefox' in user_agent:
            browser = 'firefox'
            version = '119.0'
        elif 'Safari' in user_agent:
            browser = 'safari'
            version = '17.0'
        else:
            browser = 'chrome'
            version = '119.0.0.0'
        
        if 'Windows' in user_agent:
            platform = 'windows'
        elif 'Mac' in user_agent:
            platform = 'mac'
        elif 'Linux' in user_agent:
            platform = 'linux'
        elif 'Android' in user_agent:
            platform = 'android'
        elif 'iPhone' in user_agent or 'iPad' in user_agent:
            platform = 'ios'
        else:
            platform = 'windows'
        
        return browser, version, platform
    
    def _determine_device_type(self, platform: str) -> str:
        """Determine device type from platform"""
        if platform in ['android', 'ios']:
            return 'mobile'
        elif platform in ['windows', 'mac', 'linux']:
            return 'desktop'
        else:
            return 'desktop'
    
    def _generate_screen_resolution(self, platform: str) -> str:
        """Generate realistic screen resolution"""
        if platform in ['android', 'ios']:
            resolutions = ['1080x1920', '1440x2560', '1170x2532', '1284x2778']
        else:
            resolutions = ['1920x1080', '2560x1440', '3840x2160', '1366x768', '1536x864']
        
        return random.choice(resolutions)
    
    def _generate_plugins(self, browser: str) -> List[str]:
        """Generate browser plugins"""
        base_plugins = ['PDF Viewer']
        
        if browser == 'chrome':
            base_plugins.extend(['Chrome PDF Viewer', 'Chrome PDF Plugin', 'Native Client'])
        elif browser == 'firefox':
            base_plugins.extend(['PDF Viewer', 'Google Talk Plugin'])
        elif browser == 'safari':
            base_plugins.extend(['WebKit built-in PDF', 'QuickTime Plugin'])
        
        return base_plugins
    
    def _generate_fonts(self, platform: str) -> List[str]:
        """Generate system fonts"""
        if platform == 'windows':
            return ['Arial', 'Times New Roman', 'Courier New', 'Verdana', 'Tahoma']
        elif platform == 'mac':
            return ['Helvetica', 'Helvetica Neue', 'Times', 'Courier', 'Geneva']
        elif platform == 'linux':
            return ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'Times New Roman']
        else:
            return ['Arial', 'Helvetica', 'Times New Roman']


class FingerprintGenerator:
    """Advanced browser fingerprint generator"""
    
    def __init__(self):
        self.fingerprint_cache = {}
    
    def initialize(self):
        """Initialize fingerprint generator"""
        # Pre-generate common fingerprints
        for _ in range(100):
            fingerprint = self.generate_fingerprint()
            self.fingerprint_cache[fingerprint['canvas']] = fingerprint
    
    def generate_fingerprint(self) -> Dict[str, Any]:
        """Generate comprehensive browser fingerprint"""
        return {
            'canvas': self._generate_canvas_fingerprint(),
            'webgl': self._generate_webgl_fingerprint(),
            'audio': self._generate_audio_fingerprint(),
            'timezone': random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo', 'UTC']),
            'screen_resolution': self._generate_screen_fingerprint(),
            'available_resolutions': self._generate_available_resolutions(),
            'hardware_concurrency': random.randint(2, 16),
            'device_memory': random.choice([4, 8, 16, 32]),
            'platform': random.choice(['Win32', 'MacIntel', 'Linux x86_64']),
            'language': random.choice(['en-US', 'en-GB', 'de-DE', 'fr-FR', 'ja-JP']),
            'languages': self._generate_languages(),
            'cookie_enabled': True,
            'do_not_track': random.choice([None, '1', '0']),
            'plugins': self._generate_plugin_fingerprint(),
            'fonts': self._generate_font_fingerprint(),
            'webgl_vendor': random.choice(['Google Inc.', 'Intel Inc.', 'NVIDIA Corporation', 'Apple Inc.']),
            'webgl_renderer': random.choice(['ANGLE', 'Intel Iris OpenGL Engine', 'NVIDIA GeForce RTX']),
            'touch_support': self._generate_touch_support(),
            'device_pixel_ratio': random.choice([1, 1.5, 2, 3]),
            'color_depth': random.choice([24, 30, 32]),
            'pixel_depth': random.choice([24, 30, 32]),
            'hardware_acceleration': random.choice([True, False]),
            'battery_api': random.choice([True, False]),
            'connection_type': random.choice(['4g', 'wifi', 'ethernet', '3g']),
            'effective_connection_type': random.choice(['4g', '3g', '2g']),
            'downlink_max': random.uniform(1.0, 100.0)
        }
    
    def _generate_canvas_fingerprint(self) -> str:
        """Generate canvas fingerprint"""
        # Simulate canvas fingerprinting
        components = [
            secrets.token_hex(8),
            str(random.randint(1000, 9999)),
            secrets.token_hex(4)
        ]
        return ':'.join(components)
    
    def _generate_webgl_fingerprint(self) -> str:
        """Generate WebGL fingerprint"""
        # Simulate WebGL fingerprinting
        vendors = ['Google', 'Intel', 'NVIDIA', 'AMD', 'Apple']
        renderers = ['ANGLE', 'Intel Iris', 'NVIDIA GeForce', 'AMD Radeon', 'Apple M1']
        
        vendor = random.choice(vendors)
        renderer = random.choice(renderers)
        
        return f"{vendor}_{renderer}_{secrets.token_hex(4)}"
    
    def _generate_audio_fingerprint(self) -> str:
        """Generate audio fingerprint"""
        # Simulate audio context fingerprinting
        return f"audio_{secrets.token_hex(8)}"
    
    def _generate_screen_fingerprint(self) -> str:
        """Generate screen fingerprint"""
        resolutions = ['1920x1080', '2560x1440', '3840x2160', '1366x768', '1536x864']
        return random.choice(resolutions)
    
    def _generate_available_resolutions(self) -> List[str]:
        """Generate available screen resolutions"""
        base_resolutions = ['1920x1080', '2560x1440', '3840x2160', '1366x768']
        return random.sample(base_resolutions, random.randint(2, 4))
    
    def _generate_languages(self) -> List[str]:
        """Generate browser languages"""
        language_sets = [
            ['en-US', 'en'],
            ['en-GB', 'en'],
            ['de-DE', 'de', 'en-US'],
            ['fr-FR', 'fr', 'en-US'],
            ['ja-JP', 'ja', 'en-US']
        ]
        return random.choice(language_sets)
    
    def _generate_plugin_fingerprint(self) -> List[str]:
        """Generate plugin fingerprint"""
        plugins = [
            'PDF Viewer',
            'Chrome PDF Viewer', 
            'Chrome PDF Plugin',
            'Native Client',
            'Widevine Content Decryption Module',
            'Shockwave Flash'
        ]
        return random.sample(plugins, random.randint(2, 4))
    
    def _generate_font_fingerprint(self) -> List[str]:
        """Generate font fingerprint"""
        font_sets = {
            'windows': ['Arial', 'Times New Roman', 'Courier New', 'Verdana', 'Tahoma'],
            'mac': ['Helvetica', 'Helvetica Neue', 'Times', 'Courier', 'Geneva'],
            'linux': ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'Times New Roman']
        }
        
        platform = random.choice(['windows', 'mac', 'linux'])
        return font_sets[platform]
    
    def _generate_touch_support(self) -> Dict[str, Any]:
        """Generate touch support fingerprint"""
        return {
            'max_touch_points': random.randint(0, 10),
            'touch_start': random.choice([True, False]),
            'touch_event': random.choice([True, False]),
            'touch_force': random.choice([True, False])
        }


class BanDetector:
    """Advanced ban detection system"""
    
    def __init__(self):
        self.ban_patterns = self._load_ban_patterns()
        self.recent_bans = deque(maxlen=100)
        self.ban_threshold = 3
    
    def initialize(self):
        """Initialize ban detector"""
        # Pre-compile ban patterns
        for pattern in self.ban_patterns:
            pattern['compiled'] = re.compile(pattern['pattern'], re.IGNORECASE)
    
    def _load_ban_patterns(self) -> List[Dict[str, Any]]:
        """Load ban detection patterns"""
        return [
            {'pattern': r'access denied', 'weight': 0.8, 'type': 'blocked'},
            {'pattern': r'captcha', 'weight': 0.9, 'type': 'captcha'},
            {'pattern': r'cloudflare', 'weight': 0.7, 'type': 'waf'},
            {'pattern': r'your ip has been blocked', 'weight': 1.0, 'type': 'ip_block'},
            {'pattern': r'rate limit exceeded', 'weight': 0.8, 'type': 'rate_limit'},
            {'pattern': r'bot detected', 'weight': 0.9, 'type': 'bot_detection'},
            {'pattern': r'forbidden', 'weight': 0.6, 'type': 'forbidden'},
            {'pattern': r'unauthorized', 'weight': 0.5, 'type': 'unauthorized'},
            {'pattern': r'suspicious activity', 'weight': 0.8, 'type': 'suspicious'},
            {'pattern': r'please verify you are human', 'weight': 0.9, 'type': 'human_verification'}
        ]
    
    def detect_ban(self, response_text: str, status_code: int, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect ban patterns in response"""
        ban_score = 0.0
        detected_patterns = []
        ban_type = 'none'
        
        # Check status code
        if status_code in [403, 429, 503]:
            ban_score += 0.5
            ban_type = 'http_' + str(status_code)
        
        # Check response text for ban patterns
        for pattern in self.ban_patterns:
            if pattern['compiled'].search(response_text):
                ban_score += pattern['weight']
                detected_patterns.append(pattern['type'])
                ban_type = pattern['type'] if pattern['weight'] > ban_score else ban_type
        
        # Check headers for ban indicators
        ban_headers = ['cf-chl-bypass', 'x-ratelimit-remaining', 'x-bot-score']
        for header in ban_headers:
            if header in headers:
                ban_score += 0.3
                detected_patterns.append(f'header_{header}')
        
        return {
            'banned': ban_score > 0.7,
            'score': ban_score,
            'type': ban_type,
            'patterns': detected_patterns,
            'confidence': min(1.0, ban_score)
        }
    
    def record_ban(self, identity: RotationIdentity):
        """Record ban for identity"""
        ban_record = {
            'identity_id': identity.id,
            'timestamp': time.time(),
            'ip_address': identity.ip_address,
            'user_agent': identity.user_agent
        }
        
        self.recent_bans.append(ban_record)
    
    @property
    def recent_ban_count(self) -> int:
        """Get count of recent bans"""
        current_time = time.time()
        recent_threshold = 300  # 5 minutes
        
        return len([
            ban for ban in self.recent_bans
            if current_time - ban['timestamp'] < recent_threshold
        ])


class RiskAssessor:
    """Advanced risk assessment system"""
    
    def __init__(self):
        self.risk_factors = {}
        self.risk_history = deque(maxlen=1000)
    
    def initialize(self):
        """Initialize risk assessor"""
        self.risk_factors = {
            'high_security_domain': 0.8,
            'recent_ban': 0.7,
            'suspicious_pattern': 0.6,
            'high_frequency': 0.5,
            'unusual_behavior': 0.4,
            'new_identity': 0.3,
            'low_success_rate': 0.6
        }
    
    def assess_url_risk(self, url: str) -> float:
        """Assess risk level for target URL"""
        risk_score = 0.0
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check for high-security domains
        high_security_keywords = ['bank', 'paypal', 'government', 'secure', 'login']
        if any(keyword in domain for keyword in high_security_keywords):
            risk_score += self.risk_factors['high_security_domain']
        
        # Check for known risky TLDs
        risky_tlds = ['.gov', '.mil', '.bank', '.insurance']
        if any(domain.endswith(tld) for tld in risky_tlds):
            risk_score += 0.3
        
        # Check URL complexity (complex URLs might be API endpoints)
        if len(parsed_url.path) > 50 or 'api' in parsed_url.path:
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def calculate_identity_risk(self, identity: RotationIdentity) -> float:
        """Calculate risk level for identity"""
        risk_score = 0.0
        
        # Check success rate
        if identity.success_rate < 0.5:
            risk_score += self.risk_factors['low_success_rate']
        
        # Check usage count (new identities are riskier)
        if identity.usage_count < 5:
            risk_score += self.risk_factors['new_identity']
        
        # Check ban status
        if identity.ban_status:
            risk_score += self.risk_factors['recent_ban']
        
        # Check performance (slow responses might indicate issues)
        avg_response_time = identity.performance_metrics.get('response_time', 0)
        if avg_response_time > 10.0:
            risk_score += 0.2
        
        return min(1.0, risk_score)


class SecurityMonitor:
    """Advanced security monitoring system"""
    
    def __init__(self):
        self.security_events = deque(maxlen=1000)
        self.threat_level = 'low'
    
    def initialize(self):
        """Initialize security monitor"""
        self.security_events.clear()
        self.threat_level = 'low'
    
    def get_security_requirements(self, url: str) -> Dict[str, Any]:
        """Get security requirements for URL"""
        requirements = {
            'required_flags': set(),
            'min_tls_version': 'TLSv1.2',
            'encryption_required': True,
            'certificate_validation': True
        }
        
        parsed_url = urlparse(url)
        
        # HTTPS sites require stricter security
        if parsed_url.scheme == 'https':
            requirements['required_flags'].update(['encrypted', 'secure_connection'])
            requirements['min_tls_version'] = 'TLSv1.2'
        
        # High-security domains have additional requirements
        high_security_domains = ['.bank', '.gov', '.mil']
        if any(parsed_url.netloc.endswith(domain) for domain in high_security_domains):
            requirements['required_flags'].update(['certificate_pinned', 'strict_transport'])
            requirements['min_tls_version'] = 'TLSv1.3'
        
        return requirements
    
    def record_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Record security event"""
        event = {
            'type': event_type,
            'severity': severity,
            'details': details,
            'timestamp': time.time(),
            'threat_level': self.threat_level
        }
        
        self.security_events.append(event)
        
        # Update threat level based on recent events
        self._update_threat_level()
    
    def _update_threat_level(self):
        """Update overall threat level"""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event['timestamp'] < 300  # 5 minutes
        ]
        
        high_severity_count = sum(1 for event in recent_events if event['severity'] == 'high')
        medium_severity_count = sum(1 for event in recent_events if event['severity'] == 'medium')
        
        if high_severity_count >= 3:
            self.threat_level = 'critical'
        elif high_severity_count >= 1 or medium_severity_count >= 5:
            self.threat_level = 'high'
        elif medium_severity_count >= 2:
            self.threat_level = 'medium'
        else:
            self.threat_level = 'low'


class PerformanceOptimizer:
    """Advanced performance optimization system"""
    
    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.optimization_rules = {}
    
    def initialize(self):
        """Initialize performance optimizer"""
        self.optimization_rules = {
            'response_time_threshold': 5.0,
            'success_rate_threshold': 0.8,
            'concurrent_requests_max': 10,
            'bandwidth_threshold': 1024 * 1024,  # 1 MB/s
        }
    
    def assess_performance_needs(self, strategy: Dict) -> Dict[str, Any]:
        """Assess performance needs based on strategy"""
        strategy_type = strategy.get('method', 'adaptive')
        
        needs = {
            'max_response_time': 10.0,
            'min_success_rate': 0.7,
            'concurrent_requests': 5,
            'bandwidth_priority': 'medium'
        }
        
        if strategy_type == 'aggressive':
            needs.update({
                'max_response_time': 3.0,
                'min_success_rate': 0.6,
                'concurrent_requests': 10,
                'bandwidth_priority': 'high'
            })
        elif strategy_type == 'stealth':
            needs.update({
                'max_response_time': 15.0,
                'min_success_rate': 0.9,
                'concurrent_requests': 2,
                'bandwidth_priority': 'low'
            })
        
        return needs
    
    def update_metrics(self):
        """Update performance metrics"""
        # Clean up old metrics
        current_time = time.time()
        for key in list(self.performance_metrics.keys()):
            self.performance_metrics[key] = [
                metric for metric in self.performance_metrics[key]
                if current_time - metric['timestamp'] < 3600  # Keep 1 hour
            ]
    
    @property
    def average_performance(self) -> float:
        """Get average performance score"""
        if not self.performance_metrics:
            return 0.8  # Default
        
        all_metrics = []
        for metrics in self.performance_metrics.values():
            all_metrics.extend([m.get('score', 0.5) for m in metrics])
        
        if not all_metrics:
            return 0.8
        
        return sum(all_metrics) / len(all_metrics)


class LoadBalancer:
    """Advanced load balancing system"""
    
    def __init__(self):
        self.proxy_load = {}
        self.request_distribution = {}
    
    def initialize(self, proxy_pool: List[ProxyServer]):
        """Initialize load balancer"""
        for proxy in proxy_pool:
            self.proxy_load[proxy.id] = {
                'active_requests': 0,
                'total_requests': 0,
                'last_used': 0,
                'performance_score': 1.0
            }
    
    def select_best_proxy(self, performance_needs: Dict) -> Optional[ProxyServer]:
        """Select best proxy based on load and performance"""
        suitable_proxies = []
        
        for proxy_id, load_info in self.proxy_load.items():
            # Check if proxy can handle more load
            if load_info['active_requests'] < 5:  # Max concurrent requests per proxy
                suitable_proxies.append((proxy_id, load_info['performance_score']))
        
        if not suitable_proxies:
            return None
        
        # Select proxy with best performance score
        suitable_proxies.sort(key=lambda x: x[1], reverse=True)
        best_proxy_id = suitable_proxies[0][0]
        
        # Find proxy object
        for proxy in self.proxy_pool:
            if proxy.id == best_proxy_id:
                return proxy
        
        return None
    
    def record_proxy_usage(self, proxy: ProxyServer, success: bool, response_time: float):
        """Record proxy usage for load balancing"""
        if proxy.id not in self.proxy_load:
            self.proxy_load[proxy.id] = {
                'active_requests': 0,
                'total_requests': 0,
                'last_used': time.time(),
                'performance_score': 1.0
            }
        
        load_info = self.proxy_load[proxy.id]
        load_info['total_requests'] += 1
        load_info['last_used'] = time.time()
        
        # Update performance score
        if success:
            performance_boost = max(0, 1.0 - (response_time / 10.0))
            load_info['performance_score'] = min(1.0, load_info['performance_score'] + 0.05 * performance_boost)
        else:
            load_info['performance_score'] = max(0.1, load_info['performance_score'] - 0.1)


class RotationStats:
    """Advanced rotation statistics tracking"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.ban_detections = 0
        self.identity_rotations = 0
        self.emergency_rotations = 0
        self.start_time = time.time()
        self.request_history = deque(maxlen=1000)
    
    def record_identity_usage(self, identity: RotationIdentity, url: str):
        """Record identity usage"""
        self.total_requests += 1
        
        record = {
            'timestamp': time.time(),
            'identity_id': identity.id,
            'url': url,
            'success': None,  # Will be updated later
            'response_time': None
        }
        
        self.request_history.append(record)
    
    def record_performance(self, identity: RotationIdentity, success: bool, response_time: float, ban_detected: bool):
        """Record request performance"""
        if self.request_history:
            last_record = self.request_history[-1]
            if last_record['identity_id'] == identity.id:
                last_record['success'] = success
                last_record['response_time'] = response_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if ban_detected:
            self.ban_detections += 1
    
    def record_rotation(self):
        """Record identity rotation"""
        self.identity_rotations += 1
    
    def record_emergency_rotation(self):
        """Record emergency rotation"""
        self.emergency_rotations += 1
    
    def reset_emergency(self):
        """Reset emergency statistics"""
        self.emergency_rotations = 0
        self.ban_detections = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0
        
        # Calculate recent performance (last 100 requests)
        recent_requests = list(self.request_history)[-100:]
        recent_success = sum(1 for r in recent_requests if r.get('success', False))
        recent_success_rate = (recent_success / len(recent_requests)) if recent_requests else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'ban_detections': self.ban_detections,
            'identity_rotations': self.identity_rotations,
            'emergency_rotations': self.emergency_rotations,
            'success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'runtime_seconds': runtime,
            'requests_per_second': self.total_requests / runtime if runtime > 0 else 0
        }


class PerformanceMonitor:
    """Advanced performance monitoring"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alert_thresholds = {
            'response_time': 10.0,
            'success_rate': 0.7,
            'ban_rate': 0.1
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for metric_name, metric_values in self.metrics.items():
            if metric_values:
                values = [m['value'] for m in metric_values]
                stats[f'{metric_name}_avg'] = np.mean(values)
                stats[f'{metric_name}_max'] = max(values)
                stats[f'{metric_name}_min'] = min(values)
                stats[f'{metric_name}_std'] = np.std(values)
        
        return stats
    
    @property
    def average_performance(self) -> float:
        """Get average performance score"""
        if 'performance' not in self.metrics or not self.metrics['performance']:
            return 0.8
        
        scores = [m['value'] for m in self.metrics['performance']]
        return sum(scores) / len(scores)


class SecurityAuditor:
    """Advanced security auditing"""
    
    def __init__(self):
        self.audit_log = deque(maxlen=1000)
        self.security_score = 1.0
    
    def run_audit(self):
        """Run security audit"""
        audit_result = {
            'timestamp': time.time(),
            'score': self.security_score,
            'issues': [],
            'recommendations': []
        }
        
        # Check for security issues
        if self.security_score < 0.7:
            audit_result['issues'].append('Low security score detected')
            audit_result['recommendations'].append('Consider increasing rotation frequency')
        
        if self.security_score < 0.5:
            audit_result['issues'].append('Critical security risk detected')
            audit_result['recommendations'].append('Perform emergency rotation immediately')
        
        self.audit_log.append(audit_result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            'security_score': self.security_score,
            'recent_audits': len(self.audit_log),
            'last_audit': self.audit_log[-1] if self.audit_log else None
        }


# Factory function
def create_rotation_manager(crawler, strategy: str = 'intelligent') -> RotationManager:
    """Factory function to create rotation manager"""
    manager = RotationManager(crawler)
    
    if strategy in manager.rotation_strategies:
        manager.update_strategy(strategy)
    
    return manager


print("🔄 Advanced Rotation Manager loaded successfully - Active Defense Systems Online!")
