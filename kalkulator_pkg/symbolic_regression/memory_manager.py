"""Memory Management abstraction for parallel symbolic regression.

v3.4 Audit Remediation: Decouples EvolutionTrainer from memory management.

This module provides:
1. MemoryManager Protocol - Abstract interface for memory backends
2. LocalMemoryManager - SharedMemory for single-machine parallel execution
3. NoOpMemoryManager - Serial execution (no shared memory)

By injecting a MemoryManager, the evolution algorithm becomes:
- Platform-agnostic (can swap local for cluster backends)
- Testable (mock manager for unit tests)
- Fault-tolerant (memory errors handled by manager, not algorithm)
"""
from __future__ import annotations

from typing import Protocol, Any, runtime_checkable
import numpy as np
from concurrent.futures import Executor


@runtime_checkable
class MemoryManager(Protocol):
    """Protocol for memory management backends.
    
    Implementations handle allocation, sharing, and cleanup of data
    for parallel workers. The evolution trainer calls these methods
    without knowing the underlying storage mechanism.
    """
    
    def prepare(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Prepare data for parallel access.
        
        Args:
            X: Training features array
            y: Training targets array
            
        Returns:
            Dict with 'shm_X_info' and 'shm_y_info' for workers
        """
        ...
    
    def get_executor(self, n_jobs: int) -> Executor | None:
        """Get executor for parallel work.
        
        Args:
            n_jobs: Number of parallel workers
            
        Returns:
            Executor instance or None for serial execution
        """
        ...
    
    def cleanup(self) -> None:
        """Release all managed resources."""
        ...


class LocalMemoryManager:
    """Local shared memory manager for single-machine parallel execution.
    
    Uses SharedMemoryOwner pattern to safely manage memory lifecycle.
    Only the owner process (main) calls unlink() on cleanup.
    """
    
    def __init__(self):
        self._owners = []
        self._executor = None
    
    def prepare(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Create shared memory segments for X and y."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("LocalMemoryManager: preparing shared memory...")
        print("[Debug] LocalMemoryManager: preparing shared memory...") # Force print for user visibility
        
        from .parallel import SharedMemoryOwner
        
        owner_X = SharedMemoryOwner.create(X)
        logger.debug("LocalMemoryManager: X shared memory created")
        print("[Debug] LocalMemoryManager: X shared memory created")

        owner_y = SharedMemoryOwner.create(y)
        logger.debug("LocalMemoryManager: y shared memory created")
        print("[Debug] LocalMemoryManager: y shared memory created")
        
        self._owners.extend([owner_X, owner_y])
        
        return {
            'shm_X_info': owner_X.get_info(),
            'shm_y_info': owner_y.get_info(),
        }
    
    def get_executor(self, n_jobs: int) -> Executor | None:
        """Create ProcessPoolExecutor for parallel work."""
        print(f"[Debug] LocalMemoryManager: starting executor with {n_jobs} workers...")
        from concurrent.futures import ProcessPoolExecutor
        
        self._executor = ProcessPoolExecutor(max_workers=n_jobs)
        print("[Debug] LocalMemoryManager: executor started.")
        return self._executor
    
    def cleanup(self) -> None:
        """Shutdown executor and unlink shared memory."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        for owner in self._owners:
            owner.cleanup()
        self._owners.clear()


class NoOpMemoryManager:
    """No-operation memory manager for serial execution.
    
    Used as default when no memory manager is provided.
    Passes data directly without shared memory.
    """
    
    def prepare(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return data directly (no sharing needed for serial)."""
        return {
            'X': X,
            'y': y,
            'shm_X_info': None,
            'shm_y_info': None,
        }
    
    def get_executor(self, n_jobs: int) -> None:
        """No executor for serial execution."""
        return None
    
    def cleanup(self) -> None:
        """Nothing to clean up."""
        pass
