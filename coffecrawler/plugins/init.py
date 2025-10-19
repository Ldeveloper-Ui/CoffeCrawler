"""
PLUGINS MODULE - Quantum Extensible Plugins for CoffeCrawler
"""

from .browser_plugins import QuantumBrowserPlugin, create_quantum_browser_plugin, QuantumBrowserProfile

# SpeedForce will be imported if available
try:
    from .SpeedForce import SpeedForce
    SPEEDFORCE_AVAILABLE = True
except ImportError:
    SPEEDFORCE_AVAILABLE = False
    SpeedForce = None

__all__ = [
    'QuantumBrowserPlugin',
    'create_quantum_browser_plugin', 
    'QuantumBrowserProfile',
    'SpeedForce',
    'SPEEDFORCE_AVAILABLE'
]

print("🔌 Quantum Plugins loaded successfully!")
if SPEEDFORCE_AVAILABLE:
    print("   🚀 SpeedForce C++ Quantum Accelerator: ENABLED - MAXIMUM PERFORMANCE!")
else:
    print("   ⚠️ SpeedForce C++ Quantum Accelerator: NOT AVAILABLE")
    print("   💡 Compile with: python setup_speedforce.py build_ext --inplace")
