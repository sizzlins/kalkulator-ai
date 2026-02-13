"""Parallel execution utilities using Zero-Copy Shared Memory.

This module implements:
1. SharedMemoryOwner: Explicit ownership model to avoid resource_tracker bugs (v3.3)
2. SharedMemoryContext: Manages lifecycle of shared memory blocks.
3. Parallel evolution of genetic islands to bypass GIL.

v3.3 Audit Remediation: Manager-Worker pattern replaces resource_tracker.unregister hack.
The SharedMemoryOwner class ensures only the owner process calls unlink(), preventing
the race condition where workers prematurely delete shared segments.
"""
from __future__ import annotations

import numpy as np
import logging
import os
from multiprocessing import shared_memory
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger("parallel")


# =============================================================================
# SharedMemoryOwner: v3.3 Audit Remediation
# =============================================================================
# This class implements explicit ownership tracking to eliminate the
# resource_tracker.unregister() hack. Only the owner PID can unlink.
# =============================================================================

class SharedMemoryOwner:
    """Shared memory with explicit ownership for safe multi-process usage.
    
    v3.3 Audit Remediation: Replaces resource_tracker.unregister hack.
    
    The owner process (PID that created) is the ONLY process that can unlink.
    Worker processes attach read-only and only close() on exit.
    
    Usage:
        # Owner process
        owner = SharedMemoryOwner.create(data)
        info = owner.get_info()
        
        # Pass info to workers via pickle
        # Workers:
        arr, handle = SharedMemoryOwner.attach(info)
        # ... use arr ...
        handle.close()  # Safe - won't unlink
        
        # Back in owner:
        owner.cleanup()  # Safe unlink
    """
    
    def __init__(self, shm: shared_memory.SharedMemory, shape: tuple, dtype: np.dtype, is_owner: bool):
        self.shm = shm
        self.shape = shape
        self.dtype = dtype
        self.is_owner = is_owner
        self._owner_pid = os.getpid() if is_owner else None
    
    @classmethod
    def create(cls, data: np.ndarray) -> 'SharedMemoryOwner':
        """Create new shared memory segment (owner)."""
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shm_arr[:] = data[:]
        return cls(shm, data.shape, data.dtype, is_owner=True)
    
    @classmethod
    def attach(cls, info: dict) -> tuple[np.ndarray, 'SharedMemoryOwner']:
        """Attach to existing shared memory (worker - read only)."""
        shm = shared_memory.SharedMemory(name=info['name'])
        arr = np.ndarray(info['shape'], dtype=info['dtype'], buffer=shm.buf)
        owner = cls(shm, info['shape'], info['dtype'], is_owner=False)
        return arr, owner
    
    def get_info(self) -> dict:
        """Get serializable info for passing to workers."""
        return {
            'name': self.shm.name,
            'shape': self.shape,
            'dtype': self.dtype,
        }
    
    def close(self):
        """Close handle (safe for both owner and workers)."""
        if self.shm:
            self.shm.close()
    
    def cleanup(self):
        """Unlink shared memory - ONLY call from owner process!
        
        v3.3: Explicit PID check prevents workers from unlinking.
        """
        if not self.is_owner:
            logger.warning("Non-owner tried to cleanup shared memory - ignoring")
            return
            
        if self._owner_pid != os.getpid():
            logger.warning("PID mismatch in cleanup - shared memory may be orphaned")
            return
            
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass  # Already unlinked
            self.shm = None

@contextmanager
def managed_shared_memory(data: np.ndarray, name=None):
    """Context manager for creating and cleaning up shared memory.
    
    Args:
        data: Numpy array to share
        name: Optional name for the block (auto-generated if None)
        
    Yields:
        SharedMemory object
    """
    shm = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=name)
        # Create array backed by shm and copy data
        shm_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shm_arr[:] = data[:]
        yield shm
    except Exception as e:
        logger.error(f"Failed to create shared memory: {e}")
        if shm:
            shm.close()
            shm.unlink()
        raise
    finally:
        if shm:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def get_shared_array(shm_name: str, shape: tuple, dtype: np.dtype) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to existing shared memory and return numpy array.
    
    Returns:
        (array, shm_object) - You must close shm_object when done!
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr, shm


def evolve_island_worker(
    population: list,
    shm_X_info: dict,
    shm_y_info: dict,
    config: Any, # GeneticConfig
    gen: int,
    strategy: Any # EvolutionStrategy state
) -> list:
    """Worker function to evolve an island in a separate process.
    
    v3.3 Audit Remediation: Uses SharedMemoryOwner.attach() pattern
    instead of resource_tracker.unregister hack.
    
    Args:
        population: List of ExpressionTrees
        shm_X_info: Dict with name, shape, dtype for X
        shm_y_info: Dict with name, shape, dtype for y
        config: GeneticConfig object
        gen: Current generation index
        strategy: EvolutionStrategy instance (pickled)
        
    Returns:
        Evolved population
    """
    owner_X = None
    owner_y = None
    
    try:
        # v3.3: Use SharedMemoryOwner.attach() - no unregister hack needed!
        # Workers attach as non-owners, so close() won't trigger unlink()
        X, owner_X = SharedMemoryOwner.attach(shm_X_info)
        y, owner_y = SharedMemoryOwner.attach(shm_y_info)
        
        # v3.6 Audit Fix: Explicitly unregister from resource_tracker to avoid crash on exit.
        # The runtime auto-registers segments on attach, leading to "File Not Found" errors
        # when workers exit and try to unlink segments the main process still needs.
        # v3.6 Audit Fix: Removed resource_tracker.unregister hack.
        # The runtime auto-registers segments on attach. We allow this, as
        # the main process (SharedMemoryOwner) is responsible for unlink().
        # Explicit unregister calls here caused race conditions and crashes.
        
        # Run evolution
        new_pop = strategy.evolve(population, X, y, gen)
        return new_pop
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Worker process failed unexpectedly.")
        # We re-raise to ensure the process exit code reflects failure, 
        # allowing the main process to detect it.
        raise e
        return population  # Return original on failure
        
    finally:
        # Cleanup worker-side handles (safe - won't unlink)
        if owner_X: owner_X.close()
        if owner_y: owner_y.close()

