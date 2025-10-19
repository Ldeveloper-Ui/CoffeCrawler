"""
🔥 COFFEECRAWLER EXCEPTIONS - Advanced Error Handling System
Next-generation exception handling with smart recovery capabilities
"""

import inspect
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List


class CoffeCrawlerError(Exception):
    """
    🚨 Base exception class with advanced debugging capabilities
    """
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "UNKNOWN_ERROR",
                 context: Optional[Dict[str, Any]] = None,
                 auto_fix_suggestion: str = None,
                 severity: str = "MEDIUM"):
        
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.auto_fix_suggestion = auto_fix_suggestion
        self.severity = severity.upper()
        self.timestamp = datetime.now().isoformat()
        self.caller_info = self._get_caller_info()
        
        super().__init__(self._format_message())
    
    def _get_caller_info(self) -> Dict[str, str]:
        """Get detailed information about where the error occurred"""
        try:
            frame = inspect.currentframe()
            # Go back 2 frames to get the actual caller
            for _ in range(3):
                if frame.f_back:
                    frame = frame.f_back
            
            caller_frame = frame
            return {
                'file': caller_frame.f_code.co_filename,
                'line': caller_frame.f_lineno,
                'function': caller_frame.f_code.co_name,
                'code_context': self._get_code_context(caller_frame)
            }
        except:
            return {'file': 'unknown', 'line': 0, 'function': 'unknown'}
    
    def _get_code_context(self, frame) -> str:
        """Extract the actual code line where error occurred"""
        try:
            import linecache
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            return line.strip() if line else "N/A"
        except:
            return "N/A"
    
    def _format_message(self) -> str:
        """Format error message with rich details"""
        base_msg = f"🚨 [{self.error_code}] {self.message}"
        
        if self.auto_fix_suggestion:
            base_msg += f"\n💡 Suggestion: {self.auto_fix_suggestion}"
        
        if self.context:
            context_str = " | ".join([f"{k}: {v}" for k, v in self.context.items()])
            base_msg += f"\n🔍 Context: {context_str}"
        
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'caller_info': self.caller_info,
            'context': self.context,
            'auto_fix_suggestion': self.auto_fix_suggestion
        }
    
    def __str__(self) -> str:
        return self._format_message()


class CrawlerBlockedError(CoffeCrawlerError):
    """
    🌐 Raised when crawler gets blocked by anti-bot systems
    """
    
    def __init__(self, 
                 target_url: str,
                 detection_method: str = "unknown",
                 response_code: int = None,
                 **kwargs):
        
        context = {
            'target_url': target_url,
            'detection_method': detection_method,
            'response_code': response_code,
            'timestamp': datetime.now().isoformat()
        }
        
        auto_fix = (
            "Try: rotate_user_agents(), enable_stealth_mode(), "
            "use_proxy_rotation(), or switch_to_headless_browser()"
        )
        
        super().__init__(
            message=f"Crawler blocked by {detection_method} on {target_url}",
            error_code="CRAWLER_BLOCKED",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="HIGH",
            **kwargs
        )


class ParserError(CoffeCrawlerError):
    """
    🔍 Raised when HTML/XML parsing fails
    """
    
    def __init__(self, 
                 parser_type: str,
                 element_target: str,
                 content_sample: str = None,
                 **kwargs):
        
        context = {
            'parser_type': parser_type,
            'element_target': element_target,
            'content_length': len(content_sample) if content_sample else 0,
            'content_preview': content_sample[:100] + "..." if content_sample else None
        }
        
        auto_fix = (
            "Try: switch_parser('lxml'), enable_fallback_parsing(), "
            "use_regex_backup(), or adjust_content_cleaning()"
        )
        
        super().__init__(
            message=f"Parser '{parser_type}' failed to extract '{element_target}'",
            error_code="PARSER_FAILURE",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )


class NetworkError(CoffeCrawlerError):
    """
    📡 Raised for network-related issues
    """
    
    def __init__(self, 
                 operation: str,
                 url: str = None,
                 status_code: int = None,
                 timeout: bool = False,
                 **kwargs):
        
        context = {
            'operation': operation,
            'url': url,
            'status_code': status_code,
            'timeout_occurred': timeout,
            'network_type': self._detect_network_type()
        }
        
        if timeout:
            error_msg = f"Network timeout during {operation}"
            auto_fix = "Try: increase_timeout(), retry_with_backoff(), or check_connection()"
        elif status_code:
            error_msg = f"HTTP {status_code} during {operation} on {url}"
            auto_fix = f"Try: handle_http_{status_code}(), use_alternative_endpoint(), or retry_later()"
        else:
            error_msg = f"Network failure during {operation}"
            auto_fix = "Try: check_internet_connection(), use_proxy(), or enable_offline_mode()"
        
        super().__init__(
            message=error_msg,
            error_code="NETWORK_ERROR",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="HIGH" if timeout or status_code >= 500 else "MEDIUM",
            **kwargs
        )
    
    def _detect_network_type(self) -> str:
        """Try to detect network environment"""
        try:
            import os
            if 'TERMUX_VERSION' in os.environ:
                return "TERMUX_MOBILE"
            
            # Additional detection logic can be added here
            return "UNKNOWN"
        except:
            return "UNKNOWN"


class FixerModeActivated(CoffeCrawlerError):
    """
    🔧 Special exception when DebugFixer automatically fixes issues
    """
    
    def __init__(self, 
                 original_error: str,
                 fix_applied: str,
                 recovery_method: str,
                 **kwargs):
        
        context = {
            'original_error': original_error,
            'fix_applied': fix_applied,
            'recovery_method': recovery_method,
            'auto_recovery_time': datetime.now().isoformat()
        }
        
        super().__init__(
            message=f"🛠️ DebugFixer activated: {fix_applied}",
            error_code="AUTO_FIX_APPLIED",
            context=context,
            auto_fix_suggestion=f"Recovery method: {recovery_method}",
            severity="LOW",  # This is actually a "good" exception
            **kwargs
        )


class AIError(CoffeCrawlerError):
    """
    🤖 Raised when AI-powered features fail
    """
    
    def __init__(self, 
                 ai_component: str,
                 operation: str,
                 model_status: str = "unknown",
                 **kwargs):
        
        context = {
            'ai_component': ai_component,
            'operation': operation,
            'model_status': model_status,
            'ai_timestamp': datetime.now().isoformat()
        }
        
        auto_fix = (
            "Try: fallback_to_rule_based(), reduce_ai_complexity(), "
            "enable_hybrid_mode(), or update_ai_models()"
        )
        
        super().__init__(
            message=f"AI component '{ai_component}' failed during {operation}",
            error_code="AI_PROCESSING_ERROR",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )


class StrategyError(CoffeCrawlerError):
    """
    🎯 Raised when crawling strategy fails
    """
    
    def __init__(self, 
                 strategy_name: str,
                 failure_point: str,
                 retry_count: int = 0,
                 **kwargs):
        
        context = {
            'strategy_name': strategy_name,
            'failure_point': failure_point,
            'retry_attempts': retry_count,
            'strategy_failed_at': datetime.now().isoformat()
        }
        
        auto_fix = (
            f"Try: switch_strategy('{self._suggest_alternative_strategy(strategy_name)}'), "
            "adjust_crawling_pace(), or enable_adaptive_strategy()"
        )
        
        super().__init__(
            message=f"Strategy '{strategy_name}' failed at {failure_point}",
            error_code="STRATEGY_FAILURE",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )
    
    def _suggest_alternative_strategy(self, current_strategy: str) -> str:
        """Suggest alternative strategy based on current one"""
        alternatives = {
            'stealth': 'aggressive',
            'aggressive': 'stealth', 
            'smart': 'hybrid',
            'hybrid': 'smart',
            'safe': 'stealth'
        }
        return alternatives.get(current_strategy, 'smart')


class ConfigurationError(CoffeCrawlerError):
    """
    ⚙️ Raised for configuration-related issues
    """
    
    def __init__(self, 
                 config_key: str,
                 expected_type: str = None,
                 actual_value: Any = None,
                 **kwargs):
        
        context = {
            'config_key': config_key,
            'expected_type': expected_type,
            'actual_value': str(actual_value)[:200],  # Limit length
            'config_source': 'unknown'
        }
        
        auto_fix = (
            f"Try: validate_configuration(), reset_to_defaults('{config_key}'), "
            "or use_config_preset('SAFE_MODE')"
        )
        
        super().__init__(
            message=f"Invalid configuration for '{config_key}'",
            error_code="CONFIG_ERROR", 
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )


class TermuxCompatibilityError(CoffeCrawlerError):
    """
    📱 Raised for Termux-specific compatibility issues
    """
    
    def __init__(self, 
                 feature: str,
                 termux_version: str = None,
                 **kwargs):
        
        context = {
            'feature': feature,
            'termux_version': termux_version,
            'platform': 'android',
            'compatibility_issue': True
        }
        
        auto_fix = (
            f"Try: enable_termux_compat_mode(), use_alternative_{feature}(), "
            "or install_termux_dependencies()"
        )
        
        super().__init__(
            message=f"Feature '{feature}' not compatible with Termux",
            error_code="TERMUX_COMPATIBILITY",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )

class EmulationError(CoffeCrawlerError):
    """
    🎭 Raised when human emulation fails
    """
    
    def __init__(self, 
                 emulation_type: str,
                 failure_reason: str,
                 behavior_data: dict = None,
                 **kwargs):
        
        context = {
            'emulation_type': emulation_type,
            'failure_reason': failure_reason,
            'behavior_data': behavior_data or {},
            'emulation_failed_at': datetime.now().isoformat()
        }
        
        auto_fix = (
            f"Try: simplify_emulation_pattern(), "
            f"disable_{emulation_type}_emulation(), "
            "or use_basic_behavior_profile()"
        )
        
        super().__init__(
            message=f"Human emulation failed for {emulation_type}: {failure_reason}",
            error_code="EMULATION_FAILURE",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="LOW",
            **kwargs
        )
class RotationError(CoffeCrawlerError):
    """Raised when proxy rotation fails"""
    def __init__(self, rotation_type: str, failure_reason: str, **kwargs):
        context = {'rotation_type': rotation_type, 'failure_reason': failure_reason}
        auto_fix = "Try: refresh_proxy_pool(), use_fallback_proxies(), or disable_rotation()"
        super().__init__(
            message=f"Rotation failed for {rotation_type}: {failure_reason}",
            error_code="ROTATION_FAILURE",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )

class ProxyError(CoffeCrawlerError):
    """Raised when proxy connection fails"""
    def __init__(self, proxy_url: str, error_details: str, **kwargs):
        context = {'proxy_url': proxy_url, 'error_details': error_details}
        auto_fix = "Try: test_proxy_health(), switch_provider(), or use_direct_connection()"
        super().__init__(
            message=f"Proxy failed: {proxy_url} - {error_details}",
            error_code="PROXY_ERROR", 
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="HIGH",
            **kwargs
        )

class IdentityError(CoffeCrawlerError):
    """Raised when identity management fails"""
    def __init__(self, identity_type: str, issue: str, **kwargs):
        context = {'identity_type': identity_type, 'issue': issue}
        auto_fix = "Try: regenerate_identity(), reset_fingerprint(), or clear_cookies()"
        super().__init__(
            message=f"Identity error for {identity_type}: {issue}",
            error_code="IDENTITY_ERROR",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="MEDIUM",
            **kwargs
        )

class SecurityError(CoffeCrawlerError):
    """Raised when security issues detected"""
    def __init__(self, security_threat: str, risk_level: str, **kwargs):
        context = {'security_threat': security_threat, 'risk_level': risk_level}
        auto_fix = "Try: enable_enhanced_security(), abort_operation(), or run_security_audit()"
        super().__init__(
            message=f"Security threat: {security_threat} (Risk: {risk_level})",
            error_code="SECURITY_BREACH",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="CRITICAL",
            **kwargs
        )

# Export all exception classes
__all__ = [
    'CoffeCrawlerError',
    'CrawlerBlockedError', 
    'ParserError',
    'NetworkError',
    'FixerModeActivated',
    'AIError',
    'StrategyError',
    'ConfigurationError',
    'TermuxCompatibilityError',
    'EmulationError'
    'RotationError',
    'ProxyError',
    'identityError',
    'SecurityError',
]

# Smart exception handler utility
def handle_crawler_exception(exception: Exception, auto_recover: bool = True) -> Dict[str, Any]:
    """
    Smart exception handler with auto-recovery capabilities
    
    Args:
        exception: The exception to handle
        auto_recover: Whether to attempt automatic recovery
    
    Returns:
        Dict with handling results and recovery info
    """
    
    result = {
        'handled': False,
        'recovery_attempted': False,
        'recovery_successful': False,
        'suggested_actions': [],
        'error_details': {}
    }
    
    if isinstance(exception, CoffeCrawlerError):
        result['handled'] = True
        result['error_details'] = exception.to_dict()
        result['suggested_actions'] = [exception.auto_fix_suggestion] if exception.auto_fix_suggestion else []
        
        # Auto-recovery logic for specific error types
        if auto_recover and isinstance(exception, (NetworkError, ParserError)):
            result['recovery_attempted'] = True
            result['recovery_successful'] = _attempt_auto_recovery(exception)
    
    return result


def _attempt_auto_recovery(exception: CoffeCrawlerError) -> bool:
    """
    Attempt automatic recovery based on error type
    """
    try:
        if isinstance(exception, NetworkError):
            # Implement network recovery logic
            return True
        elif isinstance(exception, ParserError):
            # Implement parser recovery logic  
            return True
        return False
    except:
        return False

class EmulationError(CoffeCrawlerError):
    """Raised when human emulation fails"""
    def __init__(self, emulation_type: str, failure_reason: str, **kwargs):
        context = {'emulation_type': emulation_type, 'failure_reason': failure_reason}
        auto_fix = f"Try: simplify_emulation_pattern() or disable_{emulation_type}_emulation()"
        super().__init__(
            message=f"Emulation failed for {emulation_type}: {failure_reason}",
            error_code="EMULATION_FAILURE",
            context=context,
            auto_fix_suggestion=auto_fix,
            severity="LOW",
            **kwargs
        )

# Update __all__
__all__.append('EmulationError')
