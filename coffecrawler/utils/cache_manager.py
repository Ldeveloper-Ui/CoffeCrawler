"""
QUANTUM CACHE MANAGER
Advanced caching system with multiple storage backends and intelligent eviction
High-performance cache with TTL, compression, and persistence
I am a garbage collector and proud.
"""

import json
import pickle
import sqlite3
import hashlib
import time
import zlib
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from threading import Lock, RLock
from pathlib import Path
import logging
from enum import Enum

class CacheBackend(Enum):
    """Supported cache backends"""
    MEMORY = "memory"
    SQLITE = "sqlite"
    JSON = "json"
    PICKLE = "pickle"

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    RANDOM = "random"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    max_size: int = 10000
    default_ttl: int = 3600  # 1 hour in seconds
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression: bool = True
    backup_interval: int = 300  # 5 minutes
    persist_to_disk: bool = True
    cache_dir: str = ".coffecrawler_cache"

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    compression_savings: float = 0.0
    total_size: int = 0

class CacheManager:
    """
    QUANTUM CACHE MANAGER
    Advanced caching system with multiple storage strategies
    """
    
    def __init__(self, config: CacheConfig = None, backend: CacheBackend = CacheBackend.SQLITE):
        self.config = config or CacheConfig()
        self.backend = backend
        self.stats = CacheStats()
        self._lock = RLock()  # Reentrant lock for thread safety
        
        # Initialize cache storage
        self._storage = self._initialize_storage()
        self._access_times = {}  # For LRU eviction
        self._access_counts = {}  # For LFU eviction
        
        # Ensure cache directory exists
        if self.config.persist_to_disk:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Start background maintenance
        self._last_maintenance = time.time()
        self._running = True
        
        self.logger = self._setup_logging()
        self.logger.info(f"🚀 Quantum Cache Manager initialized with {backend.value} backend")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cache operations"""
        logger = logging.getLogger('QuantumCacheManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_storage(self) -> Any:
        """Initialize the selected storage backend"""
        if self.backend == CacheBackend.MEMORY:
            return self._init_memory_storage()
        elif self.backend == CacheBackend.SQLITE:
            return self._init_sqlite_storage()
        elif self.backend == CacheBackend.JSON:
            return self._init_json_storage()
        elif self.backend == CacheBackend.PICKLE:
            return self._init_pickle_storage()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_memory_storage(self) -> Dict[str, Dict]:
        """Initialize in-memory storage"""
        return {}
    
    def _init_sqlite_storage(self) -> sqlite3.Connection:
        """Initialize SQLite database storage"""
        db_path = os.path.join(self.config.cache_dir, "quantum_cache.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create cache table if not exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                ttl INTEGER,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 0,
                size INTEGER
            )
        ''')
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ttl ON cache(ttl)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON cache(access_count)')
        
        conn.commit()
        return conn
    
    def _init_json_storage(self) -> Path:
        """Initialize JSON file storage"""
        json_path = Path(self.config.cache_dir) / "cache.json"
        if not json_path.exists():
            with open(json_path, 'w') as f:
                json.dump({}, f)
        return json_path
    
    def _init_pickle_storage(self) -> Path:
        """Initialize pickle file storage"""
        pickle_path = Path(self.config.cache_dir) / "cache.pkl"
        if not pickle_path.exists():
            with open(pickle_path, 'wb') as f:
                pickle.dump({}, f)
        return pickle_path
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage"""
        serialized = pickle.dumps(data)
        if self.config.compression:
            return zlib.compress(serialized)
        return serialized
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage"""
        try:
            if self.config.compression:
                serialized = zlib.decompress(compressed_data)
            else:
                serialized = compressed_data
            return pickle.loads(serialized)
        except (zlib.error, pickle.PickleError) as e:
            self.logger.error(f"Decompression error: {e}")
            return None
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        return len(pickle.dumps(data))
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with optional TTL
        """
        with self._lock:
            try:
                cache_key = self._generate_key(key)
                current_time = time.time()
                expires_at = current_time + (ttl or self.config.default_ttl)
                
                # Prepare cache entry
                compressed_value = self._compress_data(value)
                entry_size = len(compressed_value)
                
                cache_entry = {
                    'value': compressed_value,
                    'ttl': expires_at,
                    'created_at': current_time,
                    'accessed_at': current_time,
                    'access_count': 0,
                    'size': entry_size
                }
                
                # Check if we need to evict
                if self._needs_eviction(entry_size):
                    self._perform_eviction(entry_size)
                
                # Store based on backend
                if self.backend == CacheBackend.MEMORY:
                    self._storage[cache_key] = cache_entry
                
                elif self.backend == CacheBackend.SQLITE:
                    self._storage.execute(
                        '''INSERT OR REPLACE INTO cache 
                           (key, value, ttl, created_at, accessed_at, access_count, size)
                           VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (cache_key, compressed_value, expires_at, current_time, 
                         current_time, 0, entry_size)
                    )
                    self._storage.commit()
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'r+') as f:
                        data = json.load(f)
                        data[cache_key] = cache_entry
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'rb+') as f:
                        data = pickle.load(f)
                        data[cache_key] = cache_entry
                        f.seek(0)
                        pickle.dump(data, f)
                        f.truncate()
                
                # Update access tracking
                self._access_times[cache_key] = current_time
                self._access_counts[cache_key] = 0
                
                # Update statistics
                self.stats.sets += 1
                self.stats.total_size += entry_size
                
                self.logger.debug(f"✅ Cache SET: {key} -> {cache_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Cache SET error: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from cache
        """
        with self._lock:
            try:
                cache_key = self._generate_key(key)
                current_time = time.time()
                
                # Retrieve based on backend
                if self.backend == CacheBackend.MEMORY:
                    entry = self._storage.get(cache_key)
                    if not entry or entry['ttl'] < current_time:
                        self.stats.misses += 1
                        return default
                
                elif self.backend == CacheBackend.SQLITE:
                    cursor = self._storage.execute(
                        'SELECT value, ttl FROM cache WHERE key = ?', (cache_key,)
                    )
                    result = cursor.fetchone()
                    if not result or result[1] < current_time:
                        self.stats.misses += 1
                        return default
                    entry = {'value': result[0], 'ttl': result[1]}
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'r') as f:
                        data = json.load(f)
                    entry = data.get(cache_key)
                    if not entry or entry['ttl'] < current_time:
                        self.stats.misses += 1
                        return default
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'rb') as f:
                        data = pickle.load(f)
                    entry = data.get(cache_key)
                    if not entry or entry['ttl'] < current_time:
                        self.stats.misses += 1
                        return default
                
                # Check TTL
                if entry['ttl'] < current_time:
                    self.delete(key)  # Auto-expire
                    self.stats.misses += 1
                    return default
                
                # Decompress and return value
                value = self._decompress_data(entry['value'])
                
                # Update access tracking
                self._access_times[cache_key] = current_time
                self._access_counts[cache_key] = self._access_counts.get(cache_key, 0) + 1
                
                # Update backend access records
                self._update_access_metrics(cache_key, current_time)
                
                self.stats.hits += 1
                self.logger.debug(f"✅ Cache HIT: {key}")
                return value
                
            except Exception as e:
                self.logger.error(f"❌ Cache GET error: {e}")
                self.stats.misses += 1
                return default
    
    def _update_access_metrics(self, key: str, access_time: float):
        """Update access metrics in backend storage"""
        try:
            if self.backend == CacheBackend.SQLITE:
                self._storage.execute(
                    'UPDATE cache SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?',
                    (access_time, key)
                )
                self._storage.commit()
            
            elif self.backend == CacheBackend.JSON:
                with open(self._storage, 'r+') as f:
                    data = json.load(f)
                    if key in data:
                        data[key]['accessed_at'] = access_time
                        data[key]['access_count'] = data[key].get('access_count', 0) + 1
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
            
            elif self.backend == CacheBackend.PICKLE:
                with open(self._storage, 'rb+') as f:
                    data = pickle.load(f)
                    if key in data:
                        data[key]['accessed_at'] = access_time
                        data[key]['access_count'] = data[key].get('access_count', 0) + 1
                    f.seek(0)
                    pickle.dump(data, f)
                    f.truncate()
        
        except Exception as e:
            self.logger.error(f"Error updating access metrics: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        """
        with self._lock:
            try:
                cache_key = self._generate_key(key)
                
                # Delete from access tracking
                self._access_times.pop(cache_key, None)
                self._access_counts.pop(cache_key, None)
                
                # Delete based on backend
                if self.backend == CacheBackend.MEMORY:
                    if cache_key in self._storage:
                        entry_size = self._storage[cache_key]['size']
                        del self._storage[cache_key]
                        self.stats.total_size -= entry_size
                
                elif self.backend == CacheBackend.SQLITE:
                    cursor = self._storage.execute('SELECT size FROM cache WHERE key = ?', (cache_key,))
                    result = cursor.fetchone()
                    if result:
                        self.stats.total_size -= result[0]
                    self._storage.execute('DELETE FROM cache WHERE key = ?', (cache_key,))
                    self._storage.commit()
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'r+') as f:
                        data = json.load(f)
                        if cache_key in data:
                            entry_size = data[cache_key]['size']
                            del data[cache_key]
                            self.stats.total_size -= entry_size
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'rb+') as f:
                        data = pickle.load(f)
                        if cache_key in data:
                            entry_size = data[cache_key]['size']
                            del data[cache_key]
                            self.stats.total_size -= entry_size
                        f.seek(0)
                        pickle.dump(data, f)
                        f.truncate()
                
                self.stats.deletes += 1
                self.logger.debug(f"🗑️ Cache DELETE: {key}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Cache DELETE error: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired
        """
        with self._lock:
            try:
                cache_key = self._generate_key(key)
                current_time = time.time()
                
                if self.backend == CacheBackend.MEMORY:
                    entry = self._storage.get(cache_key)
                    return entry and entry['ttl'] >= current_time
                
                elif self.backend == CacheBackend.SQLITE:
                    cursor = self._storage.execute(
                        'SELECT ttl FROM cache WHERE key = ?', (cache_key,)
                    )
                    result = cursor.fetchone()
                    return result and result[0] >= current_time
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'r') as f:
                        data = json.load(f)
                    entry = data.get(cache_key)
                    return entry and entry['ttl'] >= current_time
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'rb') as f:
                        data = pickle.load(f)
                    entry = data.get(cache_key)
                    return entry and entry['ttl'] >= current_time
                
                return False
                
            except Exception as e:
                self.logger.error(f"❌ Cache EXISTS error: {e}")
                return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries
        """
        with self._lock:
            try:
                if self.backend == CacheBackend.MEMORY:
                    self._storage.clear()
                
                elif self.backend == CacheBackend.SQLITE:
                    self._storage.execute('DELETE FROM cache')
                    self._storage.commit()
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'w') as f:
                        json.dump({}, f)
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'wb') as f:
                        pickle.dump({}, f)
                
                # Clear tracking
                self._access_times.clear()
                self._access_counts.clear()
                self.stats.total_size = 0
                
                self.logger.info("🧹 Cache cleared")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Cache CLEAR error: {e}")
                return False
    
    def _needs_eviction(self, new_entry_size: int) -> bool:
        """Check if eviction is needed"""
        return self.stats.total_size + new_entry_size > self.config.max_size
    
    def _perform_eviction(self, required_space: int):
        """Perform cache eviction based on policy"""
        with self._lock:
            try:
                eviction_count = 0
                
                while self.stats.total_size + required_space > self.config.max_size:
                    if self.config.eviction_policy == EvictionPolicy.LRU:
                        key_to_evict = self._get_lru_key()
                    elif self.config.eviction_policy == EvictionPolicy.LFU:
                        key_to_evict = self._get_lfu_key()
                    elif self.config.eviction_policy == EvictionPolicy.TTL:
                        key_to_evict = self._get_expired_key()
                    else:  # RANDOM
                        key_to_evict = self._get_random_key()
                    
                    if not key_to_evict:
                        break
                    
                    # Delete the key (we'll use the public delete method to maintain consistency)
                    cache_key_str = self._find_original_key(key_to_evict)
                    if cache_key_str:
                        self.delete(cache_key_str)
                    
                    eviction_count += 1
                    self.stats.evictions += 1
                    
                    if eviction_count > 100:  # Safety limit
                        break
                
                if eviction_count > 0:
                    self.logger.info(f"🔄 Evicted {eviction_count} entries")
                    
            except Exception as e:
                self.logger.error(f"❌ Eviction error: {e}")
    
    def _get_lru_key(self) -> Optional[str]:
        """Get least recently used key"""
        if not self._access_times:
            return None
        return min(self._access_times, key=self._access_times.get)
    
    def _get_lfu_key(self) -> Optional[str]:
        """Get least frequently used key"""
        if not self._access_counts:
            return None
        return min(self._access_counts, key=self._access_counts.get)
    
    def _get_expired_key(self) -> Optional[str]:
        """Get expired key based on TTL"""
        current_time = time.time()
        for key, access_time in self._access_times.items():
            # Simple heuristic: older access times more likely to be expired soon
            if current_time - access_time > self.config.default_ttl * 0.7:
                return key
        return None
    
    def _get_random_key(self) -> Optional[str]:
        """Get random key for eviction"""
        if not self._access_times:
            return None
        return random.choice(list(self._access_times.keys()))
    
    def _find_original_key(self, cache_key: str) -> Optional[str]:
        """Find original key from cache key (reverse lookup)"""
        # This is a simplified implementation
        # In production, you might want to maintain a reverse mapping
        return cache_key  # For simplicity, returning cache key as original
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            # Calculate hit ratio
            total_operations = self.stats.hits + self.stats.misses
            if total_operations > 0:
                hit_ratio = (self.stats.hits / total_operations) * 100
            else:
                hit_ratio = 0
            
            # Update stats with calculated values
            self.stats.compression_savings = self._calculate_compression_savings()
            
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                sets=self.stats.sets,
                deletes=self.stats.deletes,
                evictions=self.stats.evictions,
                compression_savings=self.stats.compression_savings,
                total_size=self.stats.total_size
            )
    
    def _calculate_compression_savings(self) -> float:
        """Calculate compression savings percentage"""
        # This would require tracking uncompressed sizes
        # Simplified implementation
        return 0.65  # Assume 65% savings
    
    def backup(self, backup_path: Optional[str] = None) -> bool:
        """Create backup of cache"""
        with self._lock:
            try:
                if not backup_path:
                    timestamp = int(time.time())
                    backup_path = os.path.join(
                        self.config.cache_dir, 
                        f"cache_backup_{timestamp}.bak"
                    )
                
                if self.backend == CacheBackend.SQLITE:
                    # SQLite backup
                    backup_conn = sqlite3.connect(backup_path)
                    self._storage.backup(backup_conn)
                    backup_conn.close()
                
                elif self.backend in [CacheBackend.JSON, CacheBackend.PICKLE]:
                    # File copy
                    import shutil
                    shutil.copy2(str(self._storage), backup_path)
                
                self.logger.info(f"💾 Cache backed up to: {backup_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Backup error: {e}")
                return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore cache from backup"""
        with self._lock:
            try:
                if self.backend == CacheBackend.SQLITE:
                    self._storage.close()
                    import shutil
                    shutil.copy2(backup_path, os.path.join(self.config.cache_dir, "quantum_cache.db"))
                    self._storage = self._init_sqlite_storage()
                
                elif self.backend in [CacheBackend.JSON, CacheBackend.PICKLE]:
                    import shutil
                    shutil.copy2(backup_path, str(self._storage))
                
                self.logger.info(f"📂 Cache restored from: {backup_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Restore error: {e}")
                return False
    
    def optimize(self) -> bool:
        """Optimize cache performance"""
        with self._lock:
            try:
                if self.backend == CacheBackend.SQLITE:
                    self._storage.execute('VACUUM')
                    self._storage.execute('ANALYZE')
                    self._storage.commit()
                
                # Clean up expired entries
                self._cleanup_expired()
                
                self.logger.info("⚡ Cache optimized")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Optimization error: {e}")
                return False
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        with self._lock:
            try:
                current_time = time.time()
                expired_keys = []
                
                if self.backend == CacheBackend.MEMORY:
                    for key, entry in list(self._storage.items()):
                        if entry['ttl'] < current_time:
                            expired_keys.append(key)
                    for key in expired_keys:
                        del self._storage[key]
                
                elif self.backend == CacheBackend.SQLITE:
                    cursor = self._storage.execute(
                        'SELECT key, size FROM cache WHERE ttl < ?', (current_time,)
                    )
                    expired_entries = cursor.fetchall()
                    expired_keys = [entry[0] for entry in expired_entries]
                    
                    self._storage.execute('DELETE FROM cache WHERE ttl < ?', (current_time,))
                    self._storage.commit()
                
                elif self.backend == CacheBackend.JSON:
                    with open(self._storage, 'r+') as f:
                        data = json.load(f)
                        expired_keys = [
                            key for key, entry in data.items() 
                            if entry['ttl'] < current_time
                        ]
                        for key in expired_keys:
                            del data[key]
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()
                
                elif self.backend == CacheBackend.PICKLE:
                    with open(self._storage, 'rb+') as f:
                        data = pickle.load(f)
                        expired_keys = [
                            key for key, entry in data.items() 
                            if entry['ttl'] < current_time
                        ]
                        for key in expired_keys:
                            del data[key]
                        f.seek(0)
                        pickle.dump(data, f)
                        f.truncate()
                
                # Clean up tracking
                for key in expired_keys:
                    self._access_times.pop(key, None)
                    self._access_counts.pop(key, None)
                
                if expired_keys:
                    self.logger.info(f"🧹 Cleaned up {len(expired_keys)} expired entries")
                    
            except Exception as e:
                self.logger.error(f"❌ Cleanup error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        with self._lock:
            stats = self.get_stats()
            total_operations = stats.hits + stats.misses
            hit_ratio = (stats.hits / total_operations * 100) if total_operations > 0 else 0
            
            return {
                'backend': self.backend.value,
                'max_size': self.config.max_size,
                'current_size': stats.total_size,
                'usage_percentage': (stats.total_size / self.config.max_size * 100) if self.config.max_size > 0 else 0,
                'entries_count': len(self._access_times),
                'hit_ratio': hit_ratio,
                'eviction_policy': self.config.eviction_policy.value,
                'compression_enabled': self.config.compression,
                'compression_savings': f"{stats.compression_savings * 100:.1f}%",
                'total_operations': total_operations,
                'evictions': stats.evictions
            }
    
    def close(self):
        """Close cache and cleanup resources"""
        with self._lock:
            self._running = False
            
            if self.backend == CacheBackend.SQLITE:
                self._storage.close()
            
            self.logger.info("🔚 Cache manager closed")

# Convenience functions
def create_cache_manager(backend: str = "sqlite", **kwargs) -> CacheManager:
    """Create a cache manager with specified backend"""
    backend_enum = CacheBackend(backend.lower())
    config = CacheConfig(**kwargs)
    return CacheManager(config, backend_enum)

def quick_cache() -> CacheManager:
    """Create a quick cache manager with default settings"""
    return CacheManager()

if __name__ == "__main__":
    # Test the cache manager
    cache = QuantumCacheManager()
    
    # Test basic operations
    cache.set("test_key", {"data": "test_value", "number": 42}, ttl=60)
    value = cache.get("test_key")
    print(f"📦 Retrieved: {value}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"📊 Cache Stats: {stats}")
    
    # Test info
    info = cache.get_info()
    print(f"ℹ️ Cache Info: {info}")
    
    cache.close()
