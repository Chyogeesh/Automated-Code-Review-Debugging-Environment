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
