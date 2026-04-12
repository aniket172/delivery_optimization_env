def compute_reward(
    completed,
    on_time,
    late,
    cancelled,
    batched,
    priority_success,
    fuel_used,
    max_score=100,
):

    score = (
        1.0 * completed
        + 1.5 * on_time
        + 2.0 * priority_success
        + 0.5 * batched
        - 1.0 * cancelled
        - 0.7 * late
        - 0.2 * fuel_used
    )

    reward = score / max_score

    return max(0.0, min(reward, 1.0))