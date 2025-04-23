# core/executive_planner.py

class ExecutivePlanner:
    """High-level executive planning module.
    
    Breaks down complex goals into sub-tasks and coordinates memory and reasoning modules.
    Acts as the 'executive function' or planner for the AI.
    """
    def __init__(self, memory_system, meta_cognition, llm):
        """
        Args:
            memory_system: an object combining STM, EpisodicMemory, SemanticMemory.
            meta_cognition: the meta-cognitive module to query internal state.
            llm: a language model interface for reasoning and generating sub-task instructions.
        """
        self.memory = memory_system
        self.meta = meta_cognition
        self.llm = llm
    
    def plan(self, high_level_goal):
        """Generate a plan (sequence of actions or queries) for a given high-level goal.
        
        This might use the LLM to breakdown the goal.
        Returns:
            list of steps (each step could be a dict describing action or query).
        """
        # Simplistic approach: use LLM prompt to break down the goal
        prompt = (f"You are a planning module. Goal: {high_level_goal}\n"
                  "Break this goal into a sequence of sub-tasks. "
                  "Provide each sub-task as a brief imperative sentence.")
        plan_text = self.llm.complete(prompt)
        steps = [line.strip() for line in plan_text.split('\n') if line.strip()]
        # Convert steps into structured actions (placeholder: just textual for now)
        plan = [{"action": "SUBTASK", "instruction": step} for step in steps]
        return plan
    
    def execute_plan(self, plan):
        """Execute a given plan step-by-step.
        
        For each step, decides whether to query memory, use DPAD, or call LLM, etc.
        """
        results = []
        for step in plan:
            instr = step.get("instruction", "")
            # Basic parsing of instruction keywords to decide action:
            if instr.lower().startswith("recall"):
                # if instruction asks to recall something, query memory
                query = instr.split(" ", 1)[1] if " " in instr else instr
                result = self.memory.query(query)
            elif instr.lower().startswith("find out"):
                # if instruction to find external info, use semantic memory or external search
                query = instr.split(" ", 2)[2] if " " in instr else instr
                result = self.memory.semantic.query(query, top_k=1)
            else:
                # default: ask the LLM to handle this subtask (like a reasoning or answer task)
                result = self.llm.complete(f"Task: {instr}\nSolution:")
            results.append({ "step": instr, "result": result })
        return results
