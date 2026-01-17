"""Parallel execution utilities using Zero-Copy Shared Memory.

This module implements:
1. SharedMemoryContext: Manages lifecycle of shared memory blocks.
2. Parallel evolution of genetic islands to bypass GIL.
"""
from __future__ import annotations

import time
import numpy as np
import logging
from multiprocessing import shared_memory
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .expression_tree import ExpressionTree
    from .strategies import EvolutionStrategy
    from .genetic_config import GeneticConfig

logger = logging.getLogger("parallel")

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
    # Reconstruct arrays from shared memory
    X = None
    y = None
    shm_X = None
    shm_y = None
    
    try:
        X, shm_X = get_shared_array(shm_X_info['name'], shm_X_info['shape'], shm_X_info['dtype'])
        y, shm_y = get_shared_array(shm_y_info['name'], shm_y_info['shape'], shm_y_info['dtype'])
        
        # Audit Remediation (Priority 2): Explicit Unregister in Worker
        # Strictly following "Gold Standard" requirement to unregister immediately in worker.
        try:
            from multiprocessing import resource_tracker
            # Unregister both segments to prevent resource_tracker from deleting them on worker exit
            # The main process (ExitStack) handles the real unlink.
            resource_tracker.unregister(shm_X._name, "shared_memory")
            resource_tracker.unregister(shm_y._name, "shared_memory")
        except Exception:
            pass

        
        # Run evolution
        # Note: We use the passed strategy object which contains method logic
        # Ideally strategy is stateless or carries config.
        # We need to ensure strategy.evolve works here.
        
        # Need to re-attach config if it wasn't pickled correctly?
        # Usually it is.
        
        # Using the standard evolve method
        new_pop = strategy.evolve(population, X, y, gen)
        return new_pop
        
    except Exception as e:
        # Log error? standardized logging might not be set up in worker
        print(f"Worker Error: {e}")
        return population # Return original on failure
        
    finally:
        # Cleanup worker-side resources
        if shm_X: shm_X.close()
        if shm_y: shm_y.close()
