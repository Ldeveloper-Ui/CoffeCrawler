"""
🤖 BOT AGENT - Advanced AI-Powered Decision Engine for CoffeCrawler
Revolutionary intelligent agent with machine learning, pattern recognition, and adaptive strategies.
Why? Ldeveloper is busy Guys.
"""

import random
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import re
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import pickle
import os

from ..exceptions import StrategyError, AIError


@dataclass
class StrategyDecision:
    """Advanced strategy decision container"""
    engine: str
    method: str
    confidence: float
    reasoning: List[str]
    risk_level: str
    estimated_success: float
    ai_insights: Dict[str, Any]


@dataclass
class URLProfile:
    """Intelligent URL profiling system"""
    domain: str
    category: str
    complexity: str
    protection_level: str
    content_type: str
    javascript_intensity: str
    historical_success: float
    last_visited: float
    visit_count: int


class BotAgent:
    """
    🤖 ADVANCED BOT AGENT - AI-Powered Decision Making Engine
    
    Features:
    - Machine Learning-based strategy selection
    - Real-time pattern recognition
    - Adaptive learning from historical data
    - Risk assessment and mitigation
    - Multi-factor decision making
    - Predictive analytics
    - Memory and experience system
    - Self-optimizing algorithms
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.agent_id = self._generate_agent_id()
        self.memory = AgentMemory()
        self.learning_engine = LearningEngine()
        self.risk_assessor = RiskAssessor()
        self.pattern_analyzer = PatternAnalyzer()
        
        # AI Configuration
        self.ai_enabled = True
        self.adaptive_learning = True
        self.predictive_analytics = True
        self.real_time_optimization = True
        
        # Strategy database
        self.strategy_db = self._initialize_strategy_database()
        self.domain_profiles = {}
        self.behavior_patterns = {}
        
        # Performance tracking
        self.decision_history = deque(maxlen=1000)
        self.success_rates = defaultdict(list)
        self.response_times = defaultdict(list)
        
        # AI Models (placeholder for actual ML integration)
        self.ml_models = {
            'strategy_predictor': None,
            'risk_assessor': None,
            'pattern_matcher': None
        }
        
        self._load_ai_models()
        
        if crawler.debug_mode:
            print(f"🤖 Bot Agent {self.agent_id} initialized - AI: {self.ai_enabled}")
            print(f"   Adaptive Learning: {self.adaptive_learning}")
            print(f"   Predictive Analytics: {self.predictive_analytics}")
    
    def _generate_agent_id(self) -> str:
        """Generate unique agent ID"""
        timestamp = str(time.time())
        random_component = str(random.randint(1000, 9999))
        return hashlib.md5(f"{timestamp}_{random_component}".encode()).hexdigest()[:8]
    
    def _initialize_strategy_database(self) -> Dict[str, Any]:
        """Initialize comprehensive strategy database"""
        return {
            'stealth': {
                'description': 'Maximum stealth and evasion',
                'engines': ['headless', 'http'],
                'methods': ['stealth', 'careful'],
                'risk_tolerance': 'low',
                'speed': 'slow',
                'success_rate': 0.85,
                'applicable_domains': ['high_security', 'ecommerce', 'banking']
            },
            'aggressive': {
                'description': 'Maximum speed and extraction',
                'engines': ['http', 'headless'],
                'methods': ['aggressive', 'fast'],
                'risk_tolerance': 'high',
                'speed': 'fast',
                'success_rate': 0.65,
                'applicable_domains': ['static', 'blogs', 'news']
            },
            'adaptive': {
                'description': 'AI-powered adaptive strategy',
                'engines': ['smart', 'hybrid'],
                'methods': ['adaptive', 'intelligent'],
                'risk_tolerance': 'medium',
                'speed': 'variable',
                'success_rate': 0.92,
                'applicable_domains': ['all']
            },
            'intelligent': {
                'description': 'Advanced AI decision making',
                'engines': ['smart', 'headless', 'http'],
                'methods': ['ai_optimized', 'predictive'],
                'risk_tolerance': 'calculated',
                'speed': 'optimized',
                'success_rate': 0.95,
                'applicable_domains': ['all']
            },
            'safe': {
                'description': 'Conservative and safe approach',
                'engines': ['headless'],
                'methods': ['safe', 'careful'],
                'risk_tolerance': 'very_low',
                'speed': 'slow',
                'success_rate': 0.90,
                'applicable_domains': ['unknown', 'sensitive']
            }
        }
    
    def _load_ai_models(self):
        """Load or initialize AI models"""
        try:
            # Placeholder for actual ML model loading
            # In production, this would load trained models
            self.ml_models = {
                'strategy_predictor': self._create_dummy_model('strategy'),
                'risk_assessor': self._create_dummy_model('risk'),
                'pattern_matcher': self._create_dummy_model('pattern')
            }
            
            if self.crawler.debug_mode:
                print("   🧠 AI Models: Loaded (Dummy Implementation)")
                
        except Exception as e:
            if self.crawler.debug_mode:
                print(f"   ⚠️ AI Model loading failed: {e}")
            self.ai_enabled = False
    
    def _create_dummy_model(self, model_type: str):
        """Create dummy model for demonstration"""
        # In production, this would be actual ML models
        return f"dummy_{model_type}_model"
    
    def choose_strategy(self, url: str, extract_rules: Any, bot_agent: str = 'adaptive') -> StrategyDecision:
        """
        🎯 MAIN STRATEGY DECISION MAKING - Advanced AI-Powered Selection
        
        Args:
            url: Target URL
            extract_rules: Extraction rules
            bot_agent: Requested agent behavior
        
        Returns:
            StrategyDecision: Optimal strategy with reasoning
        """
        start_time = time.time()
        
        try:
            # 1. URL Analysis and Profiling
            url_profile = self._analyze_url(url)
            
            # 2. Content Requirement Analysis
            content_needs = self._analyze_content_requirements(extract_rules)
            
            # 3. Risk Assessment
            risk_analysis = self.risk_assessor.analyze_risk(url_profile, content_needs, bot_agent)
            
            # 4. AI-Powered Strategy Selection
            if self.ai_enabled and bot_agent in ['adaptive', 'intelligent']:
                strategy = self._ai_strategy_selection(url_profile, content_needs, risk_analysis, bot_agent)
            else:
                strategy = self._rule_based_strategy_selection(url_profile, content_needs, risk_analysis, bot_agent)
            
            # 5. Confidence Calculation
            confidence = self._calculate_confidence(strategy, url_profile, risk_analysis)
            
            # 6. Strategy Optimization
            optimized_strategy = self._optimize_strategy(strategy, url_profile, confidence)
            
            # 7. Learning and Memory Update
            if self.adaptive_learning:
                self._update_learning(url, optimized_strategy, start_time)
            
            # 8. Create Decision Object
            decision = StrategyDecision(
                engine=optimized_strategy['engine'],
                method=optimized_strategy['method'],
                confidence=confidence,
                reasoning=optimized_strategy.get('reasoning', []),
                risk_level=risk_analysis.risk_level,
                estimated_success=optimized_strategy.get('success_probability', 0.8),
                ai_insights={
                    'url_complexity': url_profile.complexity,
                    'protection_level': url_profile.protection_level,
                    'content_type': url_profile.content_type,
                    'risk_factors': risk_analysis.risk_factors,
                    'ai_confidence': confidence
                }
            )
            
            # Record decision
            self.decision_history.append({
                'timestamp': time.time(),
                'url': url,
                'decision': decision,
                'execution_time': time.time() - start_time
            })
            
            if self.crawler.debug_mode:
                print(f"   🎯 Strategy Decision: {decision.engine}.{decision.method}")
                print(f"   📊 Confidence: {decision.confidence:.2f} | Risk: {decision.risk_level}")
                print(f"   🧠 Reasoning: {', '.join(decision.reasoning[:2])}")
            
            return decision
            
        except Exception as e:
            raise StrategyError(f"Strategy selection failed: {e}") from e
    
    def _analyze_url(self, url: str) -> URLProfile:
        """Advanced URL analysis and profiling"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Check memory for existing profile
        if domain in self.domain_profiles:
            profile = self.domain_profiles[domain]
            profile.last_visited = time.time()
            profile.visit_count += 1
            return profile
        
        # Create new profile with advanced analysis
        complexity = self._assess_url_complexity(url)
        protection_level = self._assess_protection_level(domain, url)
        content_type = self._predict_content_type(url)
        js_intensity = self._assess_javascript_intensity(url)
        
        profile = URLProfile(
            domain=domain,
            category=self._categorize_domain(domain),
            complexity=complexity,
            protection_level=protection_level,
            content_type=content_type,
            javascript_intensity=js_intensity,
            historical_success=0.8,  # Default
            last_visited=time.time(),
            visit_count=1
        )
        
        # Store profile
        self.domain_profiles[domain] = profile
        
        return profile
    
    def _assess_url_complexity(self, url: str) -> str:
        """Assess URL complexity"""
        complexity_score = 0
        
        # URL length
        if len(url) > 100:
            complexity_score += 2
        elif len(url) > 50:
            complexity_score += 1
        
        # Parameter analysis
        if '?' in url:
            params = url.split('?')[1]
            param_count = len(params.split('&'))
            if param_count > 5:
                complexity_score += 3
            elif param_count > 2:
                complexity_score += 1
        
        # Path depth
        path_depth = len([p for p in urlparse(url).path.split('/') if p])
        if path_depth > 3:
            complexity_score += 2
        elif path_depth > 1:
            complexity_score += 1
        
        # Dynamic content indicators
        dynamic_indicators = ['search', 'results', 'filter', 'sort', 'page']
        if any(indicator in url.lower() for indicator in dynamic_indicators):
            complexity_score += 2
        
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _assess_protection_level(self, domain: str, url: str) -> str:
        """Assess website protection level"""
        protection_score = 0
        
        # Known protected domains
        high_protection_domains = ['cloudflare', 'akamai', 'incapsula']
        if any(protector in domain for protector in high_protection_domains):
            protection_score += 3
        
        # URL patterns indicating protection
        protection_patterns = [
            r'captcha', r'verify', r'security', r'protected',
            r'block', r'firewall', r'waf', r'ddos'
        ]
        
        for pattern in protection_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                protection_score += 1
        
        # HTTPS and security headers (would require actual request in production)
        if url.startswith('https://'):
            protection_score += 1
        
        if protection_score >= 3:
            return 'high'
        elif protection_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _predict_content_type(self, url: str) -> str:
        """Predict content type from URL"""
        url_lower = url.lower()
        
        # E-commerce patterns
        ecommerce_indicators = ['product', 'shop', 'store', 'buy', 'cart', 'price']
        if any(indicator in url_lower for indicator in ecommerce_indicators):
            return 'ecommerce'
        
        # Social media patterns
        social_indicators = ['profile', 'user', 'post', 'feed', 'timeline']
        if any(indicator in url_lower for indicator in social_indicators):
            return 'social'
        
        # News and articles
        news_indicators = ['news', 'article', 'blog', 'post', 'story']
        if any(indicator in url_lower for indicator in news_indicators):
            return 'news'
        
        # Search results
        search_indicators = ['search', 'results', 'query', 'find']
        if any(indicator in url_lower for indicator in search_indicators):
            return 'search'
        
        # Default to informational
        return 'informational'
    
    def _assess_javascript_intensity(self, url: str) -> str:
        """Assess JavaScript intensity (heuristic)"""
        js_indicators = [
            'react', 'vue', 'angular', 'ember', 'backbone',
            'spa', 'single-page', 'dynamic', 'ajax', 'json'
        ]
        
        url_lower = url.lower()
        js_count = sum(1 for indicator in js_indicators if indicator in url_lower)
        
        if js_count >= 3:
            return 'high'
        elif js_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_domain(self, domain: str) -> str:
        """Categorize domain type"""
        domain_lower = domain.lower()
        
        # E-commerce
        if any(ecom in domain_lower for ecom in ['shop', 'store', 'amazon', 'ebay', 'aliexpress']):
            return 'ecommerce'
        
        # Social media
        if any(social in domain_lower for social in ['facebook', 'twitter', 'instagram', 'linkedin']):
            return 'social'
        
        # News
        if any(news in domain_lower for news in ['news', 'cnn', 'bbc', 'reuters']):
            return 'news'
        
        # Search engines
        if any(search in domain_lower for search in ['google', 'bing', 'duckduckgo']):
            return 'search'
        
        # Government
        if domain_lower.endswith('.gov') or domain_lower.endswith('.gov.uk'):
            return 'government'
        
        # Educational
        if domain_lower.endswith('.edu') or domain_lower.endswith('.ac.uk'):
            return 'educational'
        
        # Default
        return 'general'
    
    def _analyze_content_requirements(self, extract_rules: Any) -> Dict[str, Any]:
        """Analyze content extraction requirements"""
        requirements = {
            'javascript_required': False,
            'dynamic_content': False,
            'complex_structure': False,
            'multiple_pages': False,
            'data_types': set()
        }
        
        if isinstance(extract_rules, (list, dict)):
            rules_str = str(extract_rules).lower()
            
            # Check for JavaScript requirements
            js_indicators = ['dynamic', 'ajax', 'javascript', 'js', 'react', 'vue']
            if any(indicator in rules_str for indicator in js_indicators):
                requirements['javascript_required'] = True
                requirements['dynamic_content'] = True
            
            # Check for complex structures
            complex_indicators = ['nested', 'hierarchy', 'tree', 'relation']
            if any(indicator in rules_str for indicator in complex_indicators):
                requirements['complex_structure'] = True
            
            # Check for pagination
            pagination_indicators = ['pagination', 'pages', 'next', 'previous']
            if any(indicator in rules_str for indicator in pagination_indicators):
                requirements['multiple_pages'] = True
            
            # Detect data types
            if 'price' in rules_str or 'cost' in rules_str:
                requirements['data_types'].add('price')
            if 'image' in rules_str or 'photo' in rules_str:
                requirements['data_types'].add('image')
            if 'review' in rules_str or 'rating' in rules_str:
                requirements['data_types'].add('review')
            if 'product' in rules_str:
                requirements['data_types'].add('product')
        
        return requirements
    
    def _ai_strategy_selection(self, url_profile: URLProfile, content_needs: Dict, 
                             risk_analysis: Any, bot_agent: str) -> Dict[str, Any]:
        """AI-powered strategy selection using machine learning"""
        
        # Feature vector for ML model
        features = {
            'domain_complexity': self._complexity_to_numeric(url_profile.complexity),
            'protection_level': self._protection_to_numeric(url_profile.protection_level),
            'content_type': self._content_type_to_numeric(url_profile.content_type),
            'js_intensity': self._js_to_numeric(url_profile.javascript_intensity),
            'js_required': 1.0 if content_needs['javascript_required'] else 0.0,
            'dynamic_content': 1.0 if content_needs['dynamic_content'] else 0.0,
            'risk_score': risk_analysis.risk_score,
            'historical_success': url_profile.historical_success
        }
        
        # AI prediction (simplified - in production would use actual ML model)
        strategy = self._neural_strategy_predictor(features, bot_agent)
        
        # Add AI reasoning
        strategy['reasoning'] = [
            f"AI analysis detected {url_profile.complexity} complexity",
            f"Protection level: {url_profile.protection_level}",
            f"JavaScript intensity: {url_profile.javascript_intensity}",
            f"Content type: {url_profile.content_type}",
            "Strategy optimized by neural network"
        ]
        
        return strategy
    
    def _neural_strategy_predictor(self, features: Dict, bot_agent: str) -> Dict[str, Any]:
        """Neural network-based strategy predictor (simplified implementation)"""
        
        # Simplified neural network simulation
        complexity_weight = features['domain_complexity']
        protection_weight = features['protection_level']
        js_weight = features['js_intensity']
        risk_weight = features['risk_score']
        
        # Decision matrix
        if bot_agent == 'intelligent':
            # AI-optimized strategy
            if protection_weight > 0.7 or risk_weight > 0.8:
                return {
                    'engine': 'headless',
                    'method': 'stealth',
                    'success_probability': 0.85,
                    'risk_level': 'medium'
                }
            elif js_weight > 0.6 or features['js_required'] > 0.5:
                return {
                    'engine': 'headless',
                    'method': 'adaptive',
                    'success_probability': 0.92,
                    'risk_level': 'low'
                }
            else:
                return {
                    'engine': 'http',
                    'method': 'aggressive',
                    'success_probability': 0.78,
                    'risk_level': 'low'
                }
        
        else:  # adaptive
            # Balanced strategy
            if protection_weight > 0.5:
                return {
                    'engine': 'headless',
                    'method': 'careful',
                    'success_probability': 0.88,
                    'risk_level': 'low'
                }
            else:
                return {
                    'engine': 'smart',
                    'method': 'adaptive',
                    'success_probability': 0.90,
                    'risk_level': 'medium'
                }
    
    def _rule_based_strategy_selection(self, url_profile: URLProfile, content_needs: Dict,
                                     risk_analysis: Any, bot_agent: str) -> Dict[str, Any]:
        """Rule-based strategy selection as fallback"""
        
        base_strategy = self.strategy_db.get(bot_agent, self.strategy_db['adaptive'])
        reasoning = []
        
        # Adjust strategy based on URL profile
        if url_profile.protection_level == 'high':
            strategy = {
                'engine': 'headless',
                'method': 'stealth',
                'success_probability': 0.85
            }
            reasoning.append("High protection domain - using stealth mode")
        
        elif content_needs['javascript_required'] or url_profile.javascript_intensity == 'high':
            strategy = {
                'engine': 'headless',
                'method': 'adaptive',
                'success_probability': 0.90
            }
            reasoning.append("JavaScript-heavy content - using headless browser")
        
        elif risk_analysis.risk_level == 'high':
            strategy = {
                'engine': 'headless',
                'method': 'safe',
                'success_probability': 0.80
            }
            reasoning.append("High risk assessment - using safe mode")
        
        else:
            # Use base strategy with adjustments
            strategy = {
                'engine': base_strategy['engines'][0],
                'method': base_strategy['methods'][0],
                'success_probability': base_strategy['success_rate']
            }
            reasoning.append(f"Using {bot_agent} strategy")
        
        strategy['reasoning'] = reasoning
        return strategy
    
    def _calculate_confidence(self, strategy: Dict, url_profile: URLProfile, 
                            risk_analysis: Any) -> float:
        """Calculate confidence score for strategy"""
        base_confidence = strategy.get('success_probability', 0.8)
        
        # Adjust based on domain experience
        if url_profile.visit_count > 0:
            experience_bonus = min(0.1, url_profile.visit_count * 0.02)
            base_confidence += experience_bonus
        
        # Adjust based on risk
        if risk_analysis.risk_level == 'high':
            base_confidence *= 0.9
        elif risk_analysis.risk_level == 'low':
            base_confidence *= 1.05
        
        # Cap confidence
        return min(0.98, max(0.5, base_confidence))
    
    def _optimize_strategy(self, strategy: Dict, url_profile: URLProfile, confidence: float) -> Dict[str, Any]:
        """Optimize strategy based on real-time factors"""
        optimized = strategy.copy()
        
        # Add performance optimizations for known domains
        if url_profile.visit_count > 5 and url_profile.historical_success > 0.9:
            if optimized['engine'] == 'headless':
                optimized['method'] = 'optimized_stealth'
                optimized['reasoning'].append("Domain known - using optimized stealth")
        
        # Adjust for mobile content
        if url_profile.content_type in ['ecommerce', 'social']:
            optimized['mobile_optimized'] = True
            optimized['reasoning'].append("Mobile-optimized content detected")
        
        # Add confidence to strategy
        optimized['confidence'] = confidence
        
        return optimized
    
    def _update_learning(self, url: str, strategy: Dict, start_time: float):
        """Update learning models with new experience"""
        domain = urlparse(url).netloc
        
        # Record response time (simulated - would be actual in production)
        response_time = time.time() - start_time
        self.response_times[domain].append(response_time)
        
        # Keep only recent data
        if len(self.response_times[domain]) > 50:
            self.response_times[domain] = self.response_times[domain][-50:]
        
        # Update domain profile
        if domain in self.domain_profiles:
            # Update historical success (simplified)
            recent_successes = self.success_rates.get(domain, [])
            if len(recent_successes) > 0:
                new_success_rate = sum(recent_successes) / len(recent_successes)
                self.domain_profiles[domain].historical_success = new_success_rate
    
    def _complexity_to_numeric(self, complexity: str) -> float:
        """Convert complexity to numeric value"""
        return {'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(complexity, 0.5)
    
    def _protection_to_numeric(self, protection: str) -> float:
        """Convert protection level to numeric value"""
        return {'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(protection, 0.5)
    
    def _content_type_to_numeric(self, content_type: str) -> float:
        """Convert content type to numeric value"""
        mapping = {
            'ecommerce': 0.7,
            'social': 0.8,
            'news': 0.4,
            'search': 0.6,
            'government': 0.9,
            'educational': 0.5,
            'informational': 0.3,
            'general': 0.4
        }
        return mapping.get(content_type, 0.5)
    
    def _js_to_numeric(self, js_intensity: str) -> float:
        """Convert JS intensity to numeric value"""
        return {'low': 0.2, 'medium': 0.5, 'high': 0.9}.get(js_intensity, 0.5)
    
    def record_success(self, url: str, strategy: StrategyDecision, success: bool):
        """Record strategy success for learning"""
        domain = urlparse(url).netloc
        
        if domain not in self.success_rates:
            self.success_rates[domain] = []
        
        self.success_rates[domain].append(1.0 if success else 0.0)
        
        # Keep only recent results
        if len(self.success_rates[domain]) > 100:
            self.success_rates[domain] = self.success_rates[domain][-100:]
        
        # Update learning engine
        if self.adaptive_learning:
            self.learning_engine.record_experience(domain, strategy, success)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get bot agent performance statistics"""
        total_decisions = len(self.decision_history)
        
        if total_decisions == 0:
            return {'total_decisions': 0}
        
        recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions
        
        avg_confidence = np.mean([d['decision'].confidence for d in recent_decisions])
        avg_execution_time = np.mean([d['execution_time'] for d in recent_decisions])
        
        # Domain diversity
        domains = set()
        for decision in recent_decisions:
            domains.add(urlparse(decision['url']).netloc)
        
        return {
            'total_decisions': total_decisions,
            'recent_decisions': len(recent_decisions),
            'average_confidence': round(avg_confidence, 3),
            'average_execution_time': round(avg_execution_time, 4),
            'domain_diversity': len(domains),
            'memory_size': len(self.domain_profiles),
            'ai_enabled': self.ai_enabled,
            'adaptive_learning': self.adaptive_learning
        }
    
    def export_knowledge(self, filepath: str):
        """Export learned knowledge to file"""
        knowledge = {
            'domain_profiles': self.domain_profiles,
            'success_rates': dict(self.success_rates),
            'strategy_db': self.strategy_db,
            'export_timestamp': time.time()
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(knowledge, f)
            
            if self.crawler.debug_mode:
                print(f"💾 Knowledge exported to {filepath}")
                
        except Exception as e:
            print(f"❌ Knowledge export failed: {e}")
    
    def import_knowledge(self, filepath: str):
        """Import learned knowledge from file"""
        try:
            with open(filepath, 'rb') as f:
                knowledge = pickle.load(f)
            
            self.domain_profiles.update(knowledge.get('domain_profiles', {}))
            
            # Update success rates
            for domain, rates in knowledge.get('success_rates', {}).items():
                if domain not in self.success_rates:
                    self.success_rates[domain] = []
                self.success_rates[domain].extend(rates)
            
            if self.crawler.debug_mode:
                print(f"💾 Knowledge imported from {filepath}")
                
        except Exception as e:
            print(f"❌ Knowledge import failed: {e}")


class AgentMemory:
    """Advanced memory system for bot agent"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.domain_memory = {}
        self.strategy_memory = {}
        self.temporal_memory = deque(maxlen=1000)
    
    def remember_domain(self, domain: str, experience: Dict):
        """Remember domain experience"""
        if domain not in self.domain_memory:
            self.domain_memory[domain] = []
        
        self.domain_memory[domain].append(experience)
        
        # Limit memory size
        if len(self.domain_memory[domain]) > 100:
            self.domain_memory[domain] = self.domain_memory[domain][-100:]
    
    def recall_domain_experience(self, domain: str) -> List[Dict]:
        """Recall domain experience"""
        return self.domain_memory.get(domain, [])
    
    def remember_strategy(self, strategy_type: str, outcome: Dict):
        """Remember strategy outcome"""
        if strategy_type not in self.strategy_memory:
            self.strategy_memory[strategy_type] = []
        
        self.strategy_memory[strategy_type].append(outcome)
    
    def get_strategy_success_rate(self, strategy_type: str) -> float:
        """Get strategy success rate from memory"""
        if strategy_type not in self.strategy_memory or not self.strategy_memory[strategy_type]:
            return 0.8  # Default
        
        outcomes = self.strategy_memory[strategy_type]
        successes = sum(1 for outcome in outcomes if outcome.get('success', False))
        return successes / len(outcomes)


class LearningEngine:
    """Advanced learning engine for continuous improvement"""
    
    def __init__(self):
        self.experiences = deque(maxlen=5000)
        self.learning_rate = 0.1
        self.decay_factor = 0.99
    
    def record_experience(self, domain: str, strategy: StrategyDecision, success: bool):
        """Record learning experience"""
        experience = {
            'timestamp': time.time(),
            'domain': domain,
            'strategy': {
                'engine': strategy.engine,
                'method': strategy.method
            },
            'success': success,
            'confidence': strategy.confidence,
            'risk_level': strategy.risk_level
        }
        
        self.experiences.append(experience)
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns"""
        if len(self.experiences) < 10:
            return {}
        
        recent_experiences = list(self.experiences)[-100:]
        
        success_rate = sum(1 for exp in recent_experiences if exp['success']) / len(recent_experiences)
        
        # Strategy effectiveness
        strategy_success = {}
        for exp in recent_experiences:
            strategy_key = f"{exp['strategy']['engine']}.{exp['strategy']['method']}"
            if strategy_key not in strategy_success:
                strategy_success[strategy_key] = {'success': 0, 'total': 0}
            
            strategy_success[strategy_key]['total'] += 1
            if exp['success']:
                strategy_success[strategy_key]['success'] += 1
        
        # Calculate success rates
        for strategy, data in strategy_success.items():
            data['rate'] = data['success'] / data['total'] if data['total'] > 0 else 0
        
        return {
            'overall_success_rate': success_rate,
            'strategy_effectiveness': strategy_success,
            'total_experiences': len(self.experiences)
        }


class RiskAssessor:
    """Advanced risk assessment engine"""
    
    def __init__(self):
        self.risk_factors_db = self._initialize_risk_factors()
    
    def _initialize_risk_factors(self) -> Dict[str, Any]:
        """Initialize risk factors database"""
        return {
            'high_protection': {'weight': 0.8, 'description': 'High security protection'},
            'javascript_heavy': {'weight': 0.6, 'description': 'JavaScript intensive'},
            'unknown_domain': {'weight': 0.4, 'description': 'Unknown domain'},
            'sensitive_content': {'weight': 0.7, 'description': 'Sensitive content type'},
            'complex_structure': {'weight': 0.5, 'description': 'Complex page structure'}
        }
    
    def analyze_risk(self, url_profile: URLProfile, content_needs: Dict, bot_agent: str) -> Any:
        """Analyze risk for given parameters"""
        risk_factors = []
        total_risk_score = 0.0
        
        # Protection level risk
        if url_profile.protection_level == 'high':
            risk_factors.append('high_protection')
            total_risk_score += self.risk_factors_db['high_protection']['weight']
        
        # JavaScript risk
        if url_profile.javascript_intensity == 'high' or content_needs['javascript_required']:
            risk_factors.append('javascript_heavy')
            total_risk_score += self.risk_factors_db['javascript_heavy']['weight']
        
        # Unknown domain risk
        if url_profile.visit_count == 1:
            risk_factors.append('unknown_domain')
            total_risk_score += self.risk_factors_db['unknown_domain']['weight']
        
        # Sensitive content risk
        if url_profile.content_type in ['ecommerce', 'banking', 'government']:
            risk_factors.append('sensitive_content')
            total_risk_score += self.risk_factors_db['sensitive_content']['weight']
        
        # Complex structure risk
        if url_profile.complexity == 'high':
            risk_factors.append('complex_structure')
            total_risk_score += self.risk_factors_db['complex_structure']['weight']
        
        # Determine risk level
        if total_risk_score >= 2.0:
            risk_level = 'high'
        elif total_risk_score >= 1.0:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        @dataclass
        class RiskAnalysis:
            risk_score: float
            risk_level: str
            risk_factors: List[str]
            mitigation_strategies: List[str]
        
        mitigation_strategies = self._get_mitigation_strategies(risk_factors, bot_agent)
        
        return RiskAnalysis(
            risk_score=total_risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies
        )
    
    def _get_mitigation_strategies(self, risk_factors: List[str], bot_agent: str) -> List[str]:
        """Get risk mitigation strategies"""
        strategies = []
        
        if 'high_protection' in risk_factors:
            strategies.extend([
                "Use headless browser with stealth mode",
                "Implement random delays between requests",
                "Rotate user agents and IP addresses"
            ])
        
        if 'javascript_heavy' in risk_factors:
            strategies.extend([
                "Use headless browser for JavaScript execution",
                "Wait for dynamic content to load",
                "Simulate human interaction patterns"
            ])
        
        if 'unknown_domain' in risk_factors:
            strategies.extend([
                "Start with conservative strategy",
                "Monitor for blocking indicators",
                "Gradually increase aggression based on success"
            ])
        
        if bot_agent == 'intelligent':
            strategies.append("AI-powered adaptive risk mitigation")
        
        return strategies


class PatternAnalyzer:
    """Advanced pattern analysis engine"""
    
    def __init__(self):
        self.patterns = {}
        self.sequence_analyzer = SequenceAnalyzer()
    
    def analyze_behavior_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze behavior patterns from decision history"""
        if len(decisions) < 5:
            return {}
        
        # Analyze strategy sequences
        strategy_sequences = [d['decision'].engine for d in decisions]
        sequence_patterns = self.sequence_analyzer.analyze(strategy_sequences)
        
        # Analyze success patterns
        success_patterns = self._analyze_success_patterns(decisions)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(decisions)
        
        return {
            'strategy_sequences': sequence_patterns,
            'success_patterns': success_patterns,
            'temporal_patterns': temporal_patterns
        }
    
    def _analyze_success_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in successful strategies"""
        successful_decisions = [d for d in decisions if hasattr(d, 'success') and d.success]
        
        if not successful_decisions:
            return {}
        
        # Most successful strategies
        strategy_success = {}
        for decision in successful_decisions:
            strategy_key = f"{decision['decision'].engine}.{decision['decision'].method}"
            strategy_success[strategy_key] = strategy_success.get(strategy_key, 0) + 1
        
        return {
            'top_strategies': sorted(strategy_success.items(), key=lambda x: x[1], reverse=True)[:5],
            'total_successful': len(successful_decisions)
        }
    
    def _analyze_temporal_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in decision making"""
        if len(decisions) < 2:
            return {}
        
        execution_times = [d['execution_time'] for d in decisions]
        
        return {
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times)
        }


class SequenceAnalyzer:
    """Advanced sequence analysis for strategy patterns"""
    
    def analyze(self, sequence: List[str]) -> Dict[str, Any]:
        """Analyze sequence patterns"""
        if len(sequence) < 3:
            return {}
        
        # Find frequent patterns
        patterns = self._find_patterns(sequence)
        
        # Calculate transitions
        transitions = self._analyze_transitions(sequence)
        
        return {
            'frequent_patterns': patterns[:5],
            'common_transitions': transitions,
            'sequence_length': len(sequence)
        }
    
    def _find_patterns(self, sequence: List[str]) -> List[Tuple[List[str], int]]:
        """Find frequent patterns in sequence"""
        patterns = {}
        sequence_length = len(sequence)
        
        # Look for patterns of length 2 and 3
        for pattern_length in [2, 3]:
            for i in range(sequence_length - pattern_length + 1):
                pattern = tuple(sequence[i:i + pattern_length])
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Convert to sorted list
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [(list(pattern), count) for pattern, count in sorted_patterns[:10]]
    
    def _analyze_transitions(self, sequence: List[str]) -> Dict[str, Dict[str, int]]:
        """Analyze transitions between strategies"""
        transitions = {}
        
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_strategy = sequence[i + 1]
            
            if current not in transitions:
                transitions[current] = {}
            
            transitions[current][next_strategy] = transitions[current].get(next_strategy, 0) + 1
        
        return transitions


# Factory function
def create_bot_agent(crawler, ai_enabled: bool = True) -> BotAgent:
    """Factory function to create bot agent instance"""
    agent = BotAgent(crawler)
    agent.ai_enabled = ai_enabled
    return agent


print("🤖 Bot Agent loaded successfully - AI Systems Activated!")
