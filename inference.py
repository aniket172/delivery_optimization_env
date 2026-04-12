"""
Inference Script — Delivery Dispatch Environment
===================================

Required environment variables:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from delivery_optimisation import DeliveryOptimisationEnv, DeliveryOptimisationAction
from delivery_optimisation.tasks import TASKS

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/groq/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-4-Scout-17B-16E-Instruct"

ENV_URL = os.getenv("ENV_URL") or "http://localhost:8000"

BENCHMARK = "delivery_dispatch_env"
MAX_STEPS = 30
TEMPERATURE = 0.0
MAX_TOKENS = 300


SYSTEM_PROMPT = textwrap.dedent(
    """
You are a logistics dispatcher for a food delivery platform.

Your job:
- Assign drivers to orders
- Batch compatible orders
- Batch assignments must contain at least 2 order IDs.Do not create a batch with only one order.
- Reposition idle drivers to high-demand areas

Goals:
- Maximize on-time deliveries
- Prioritize urgent (priority) orders
- Minimize cancellations and fuel usage
- Batch nearby orders efficiently

You will receive:
- driver locations
- pending orders
- demand heatmap
- traffic and weather


Respond ONLY with valid JSON.
Do NOT include markdown or code blocks.
{
  "assignments":[{"driver_id":int,"order_id":int}],
  "batch_assignments":[{"driver_id":int,"orders":[int,int]}],
  "reposition":[{"driver_id":int,"target_node":int}]
}
"""
).strip()


# ───────────────────────── Logging helpers ─────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ───────────────────────── LLM prompt ─────────────────────────

def build_user_prompt(obs, step: int) -> str:

    return textwrap.dedent(
        f"""
Step: {step}

Drivers:
{obs.drivers}

Pending Orders:
{obs.pending_orders}

Traffic Level: {obs.traffic_level}
Weather: {obs.weather}

Demand Heatmap:
{obs.demand_heatmap}

Decide:
- which drivers should take which orders
- which drivers should batch orders
- which drivers should reposition

Respond ONLY with valid JSON.
Do NOT include markdown or code blocks.
"""
    ).strip()


def get_model_action(client: OpenAI, obs, step: int) -> DeliveryOptimisationAction:

    prompt = build_user_prompt(obs, step)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        text = (completion.choices[0].message.content or "").strip()

        data = json.loads(text)
        
        return DeliveryOptimisationAction(**data)

    except Exception as e:
        print(f"[ERROR] Exception in get_model_action: {e}", flush=True)
        return DeliveryOptimisationAction(assignments=[], batch_assignments=[], reposition=[])


# ───────────────────────── Episode loop ─────────────────────────

async def run_task(llm: OpenAI, task_name: str):

    env = DeliveryOptimisationEnv(base_url=ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:

        result = await env.reset(task_name=task_name)

        obs = result.observation

        for step in range(1, MAX_STEPS + 1):

            if result.done:
                break

            action = get_model_action(llm, obs, step)

            result = await env.step(action)

            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action.dict()),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    finally:

        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ───────────────────────── Main runner ─────────────────────────

async def main():

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_task(llm, task.name)


if __name__ == "__main__":
    asyncio.run(main())
