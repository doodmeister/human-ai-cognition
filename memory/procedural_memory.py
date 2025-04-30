"""
procedural_memory.py

Implements ProceduralMemory for storing and retrieving learned procedures (skills/action sequences).
"""

class ProceduralMemory:
    """
    ProceduralMemory stores and retrieves learned procedures (skills or action sequences).
    """
    def __init__(self):
        """
        Initialize an empty procedural memory store.
        """
        # Mapping from procedure name or context key to list of actions (the procedure).
        self._procedures = {}

    def add_procedure(self, name, steps):
        """
        Store or update a procedure.
        
        Args:
            name (str): Identifier for the procedure (e.g., task name or context).
            steps (list): Sequence of actions or steps constituting the procedure.
        """
        self._procedures[name] = steps

    def get_procedure(self, name):
        """
        Retrieve a stored procedure by name.
        
        Args:
            name (str): Identifier of the procedure to retrieve.
        
        Returns:
            list or None: The stored sequence of actions if found, else None.
        """
        return self._procedures.get(name)

    def find_matching_procedure(self, context):
        """
        Retrieve a procedure matching the given context.
        This method can be extended for more advanced matching strategies.
        
        Args:
            context (str): The context key to search for a matching procedure.
        
        Returns:
            list or None: The matching procedure if found, else None.
        """
        # Currently using exact match on context; implement fuzzy or partial matching here if needed.
        return self._procedures.get(context)