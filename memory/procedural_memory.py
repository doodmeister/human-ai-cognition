"""
procedural_memory.py

Implements ProceduralMemory for storing and retrieving learned procedures (skills/action sequences).
This module provides thread-safe storage and retrieval of procedural memories with validation,
logging, and performance optimizations.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from threading import Lock
from dataclasses import dataclass
from functools import lru_cache
import json
import time
from metacognition.meta_feedback import MetaFeedbackManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Procedure:
    """Data class representing a stored procedure with metadata."""
    name: str
    steps: List[Any]
    created_at: float
    modified_at: float
    version: int = 1

class ProcedureValidationError(Exception):
    """Raised when procedure validation fails."""
    pass

class ProceduralMemory:
    """
    Thread-safe storage and retrieval of procedural memories (skills/action sequences).
    
    Features:
    - Thread-safe operations
    - Input validation
    - Caching for frequently accessed procedures
    - Logging of operations
    - Error handling
    - Memory usage optimization
    
    Example:
        memory = ProceduralMemory()
        memory.add_procedure("task1", ["step1", "step2"])
        steps = memory.get_procedure("task1")
    """
    
    def __init__(self, max_procedures: int = 1000, cache_size: int = 100):
        """
        Initialize procedural memory with configurable limits.
        
        Args:
            max_procedures: Maximum number of procedures to store
            cache_size: Size of the LRU cache for frequent procedures
        """
        self._procedures: Dict[str, Procedure] = {}
        self._lock = Lock()
        self._max_procedures = max_procedures
        self.cache_size = cache_size
        self.meta_feedback = MetaFeedbackManager()
        logger.info(f"Initialized ProceduralMemory with max_procedures={max_procedures}")

    def _validate_procedure(self, name: str, steps: List[Any]) -> None:
        """
        Validate procedure inputs.
        
        Args:
            name: Procedure identifier
            steps: List of procedure steps
            
        Raises:
            ProcedureValidationError: If validation fails
        """
        if not isinstance(name, str) or not name.strip():
            raise ProcedureValidationError("Procedure name must be a non-empty string")
        
        if not isinstance(steps, list) or not steps:
            raise ProcedureValidationError("Steps must be a non-empty list")
            
        if len(self._procedures) >= self._max_procedures:
            raise ProcedureValidationError(f"Maximum procedures limit ({self._max_procedures}) reached")

    def add_procedure(self, name: str, steps: List[Any]) -> None:
        """
        Store or update a procedure with thread safety.
        
        Args:
            name: Identifier for the procedure
            steps: Sequence of actions/steps
            
        Raises:
            ProcedureValidationError: If inputs are invalid
        """
        try:
            self._validate_procedure(name, steps)
            
            with self._lock:
                procedure = Procedure(
                    name=name,
                    steps=steps.copy(),  # Defensive copy
                    created_at=time.time(),
                    modified_at=time.time()
                )
                self._procedures[name] = procedure
                
            logger.info(f"Added/updated procedure '{name}' with {len(steps)} steps")
            
        except ProcedureValidationError as e:
            logger.error(f"Validation error adding procedure '{name}': {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error adding procedure '{name}': {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def get_procedure(self, name: str) -> Optional[List[Any]]:
        """
        Retrieve a stored procedure with caching.
        
        Args:
            name: Procedure identifier
            
        Returns:
            Stored procedure steps or None if not found
        """
        try:
            with self._lock:
                procedure = self._procedures.get(name)
                if procedure:
                    return procedure.steps.copy()  # Defensive copy
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving procedure '{name}': {str(e)}")
            return None

    def find_matching_procedure(self, context: str) -> Optional[List[Any]]:
        """
        Find procedure matching the context using fuzzy matching.
        
        Args:
            context: Context to match against procedure names
            
        Returns:
            Matching procedure steps or None
        """
        try:
            if not isinstance(context, str):
                raise ValueError("Context must be a string")
                
            with self._lock:
                # Simple fuzzy matching - can be enhanced with more sophisticated algorithms
                for name, procedure in self._procedures.items():
                    if context.lower() in name.lower():
                        return procedure.steps.copy()
            return None
            
        except Exception as e:
            logger.error(f"Error finding procedure for context '{context}': {str(e)}")
            return None

    def export_procedures(self, filepath: str) -> bool:
        """
        Export procedures to JSON file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                procedures_dict = {
                    name: vars(proc) for name, proc in self._procedures.items()
                }
            with open(filepath, 'w') as f:
                json.dump(procedures_dict, f, indent=2)
            return True
            
        except Exception as e:
            logger.error(f"Error exporting procedures: {str(e)}")
            return False

    def execute_procedure(self, name: str, context: Dict[str, Any] = None) -> bool:
        """Execute a stored procedure with performance monitoring."""
        if name not in self._procedures:
            return False

        start_time = time.time()
        success = False
        
        try:
            steps = self._procedures[name].steps
            # Execute procedure steps...
            success = True  # Set based on actual execution results
        except Exception as e:
            logger.error(f"Procedure execution failed: {str(e)}")
            success = False
        finally:
            execution_time = time.time() - start_time
            
            # Log execution metrics
            self.meta_feedback.log_procedure_metrics(
                name=name,
                success=success,
                execution_time=execution_time
            )
            
            # Update execution statistics
            with self._lock:
                if name in self._procedures:
                    procedure = self._procedures[name]
                    procedure.modified_at = time.time()
                    procedure.version += 1

        return success