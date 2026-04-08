import pydantic
from typing import Dict, Any, Tuple

class Observation(pydantic.BaseModel):
    code: str
    error_log: str
    test_results: str

class Action(pydantic.BaseModel):
    action_type: str # 'edit' or 'test'
    payload: str

class OpenEnvDebug:
    def __init__(self):
        self.tasks = {
            "easy": "def add(a, b): return a - b", # Logic error
            "medium": "def find_max(arr): return max(arr", # Syntax error
            "hard": "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)" # Optimization needed
        }
        self.current_task = "easy"
        self.code = ""
        self.steps = 0

    def reset(self, task_name="easy"):
        self.current_task = task_name
        self.code = self.tasks[task_name]
        self.steps = 0
        return Observation(code=self.code, error_log="None", test_results="Untested")

    def step(self, action_str: str) -> Tuple[Observation, float, bool, dict]:
        self.steps += 1
        reward = -0.01 # Step penalty
        done = False
        info = {"error": None}
        
        # Simple Logic for Action Parsing
        try:
            if "edit" in action_str:
                self.code = action_str.split("edit(")[1].rstrip(")")
                reward += 0.1 # Small reward for attempting an edit
            
            # Grader Logic
            if self.current_task == "easy" and "a + b" in self.code:
                reward = 1.0
                done = True
            elif self.current_task == "medium" and "max(arr)" in self.code:
                reward = 1.0
                done = True
        except Exception as e:
            info["error"] = str(e)

        if self.steps >= 5: done = True
        
        return Observation(code=self.code, error_log="None", test_results="Pass" if done else "Fail"), reward, done, info
