import os
import json
from openai import OpenAI
from main import OpenEnvDebug # Import your env

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.5-preview") # Or your chosen model
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def call_llm(role_prompt, user_input):
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return res.choices[0].message.content

def run_inference():
    env = OpenEnvDebug()
    task_name = "easy"
    obs = env.reset(task_name)
    
    print(f"[START] task={task_name} env=CodeDebug model={MODEL_NAME}")
    
    total_reward = 0
    rewards_list = []
    done = False
    step = 0
    
    # SYSTEM PROMPTS
    ACTOR_PROMPT = "You are ChatGPT. Propose a Python fix. Output ONLY: Action: edit('new_code')"
    CRITIC_PROMPT = "You are Claude. Critique the fix and output: Improved Action: edit('better_code')"

    while not done and step < 5:
        step += 1
        # 1. Actor Proposes
        raw_action = call_llm(ACTOR_PROMPT, f"Code: {obs.code}")
        
        # 2. Critic Refines
        critique_result = call_llm(CRITIC_PROMPT, f"Proposed: {raw_action} for Code: {obs.code}")
        
        # 3. Extract final action (Cleaning the string)
        final_action = critique_result.split("Improved Action: ")[-1].strip()
        
        # 4. Env Step
        obs, reward, done, info = env.step(final_action)
        total_reward += reward
        rewards_list.append(f"{reward:.2f}")
        
        print(f"[STEP] step={step} action={final_action[:20]}... reward={reward:.2f} done={str(done).lower()} error={info['error'] or 'null'}")

    print(f"[END] success={str(done).lower()} steps={step} rewards={','.join(rewards_list)}")

if __name__ == "__main__":
    run_inference()


import os
from openai import OpenAI

# Env vars with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # or whatever you have access to
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def run_inference(env, task_name: str = "code-debug"):
    obs = env.reset()
    done = False
    step_num = 0
    rewards = []
    success = False
    critic_feedback = "None"
    history = []  # for loop prevention

    print(f"[START] task={task_name} env=openenv model={MODEL_NAME}")

    try:
        while not done:
            step_num += 1

            # Actor (ChatGPT-style)
            actor_prompt = f"""Task: {task_name}
Observation: {obs}
Critic Feedback: {critic_feedback}
Previous Actions: {history[-3:]}"""
            action = call_llm(actor_prompt)  # assumes it outputs "Action: xxx"

            # Critic (Claude-style)
            critic_prompt = f"""Task: {task_name}
Observation: {obs}
Proposed Action: {action}"""
            critique = call_llm(critic_prompt)

            # Refine
            refine_prompt = f"""Task: {task_name}
Observation: {obs}
Critique: {critique}"""
            final_action = call_llm(refine_prompt)

            # Clean extraction if needed
            if "Action:" in final_action:
                final_action = final_action.split("Action:")[-1].strip()

            history.append(final_action)
            observation, reward, done, info = env.step(final_action)

            rewards.append(f"{reward:.2f}")
            error = info.get("error", "null") if info else "null"

            print(f"[STEP] step={step_num} action={final_action} reward={reward:.2f} done={str(done).lower()} error={error}")

            obs = observation
            critic_feedback = critique

        success = True
    except Exception as e:
        print(f"[STEP] step={step_num} action=null reward=0.00 done=true error={str(e)}")
    finally:
        env.close()
        print(f"[END] success={str(success).lower()} steps={step_num} rewards={','.join(rewards)}")
