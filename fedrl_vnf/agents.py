# fedrl_vnf/agents.py
from typing import Optional, List
import numpy as np
import random


class DomainRLAgent:
    """
    ε-greedy bandit agent.
    action = choose one domain-path index among K candidate paths.

    IMPORTANT:
    - num_actions should match k_paths (or an upper bound of max candidates).
    """

    def __init__(
        self,
        domain_id: str,
        num_actions: int,
        lr: float = 0.15,
        epsilon: float = 1.0,
        optimistic_init: float = 0.0,
        reward_clip: float = 5.0,
    ):
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.domain_id = domain_id
        self.num_actions = int(num_actions)
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.reward_clip = float(reward_clip)

        self.q_values = np.full(self.num_actions, float(optimistic_init), dtype=float)

        self.last_action: Optional[int] = None
        self.last_valid_actions: Optional[List[int]] = None

    def get_params(self) -> np.ndarray:
        return self.q_values.copy()

    def set_params(self, params: np.ndarray):
        params = np.asarray(params, dtype=float).reshape(-1)
        if params.shape[0] != self.num_actions:
            return
        self.q_values = params.copy()

    def select_action(self, valid_actions: Optional[List[int]] = None) -> int:
        """
        valid_actions: list of allowed action indices (e.g., [0,1,2]).
        Will be filtered into [0, num_actions-1].

        ✅ 关键修复：
        - eval 时 epsilon=0 仍可能出现多个 action 的 Q 完全相同
        - 原来随机 tie-break 会导致 eval 曲线抖动
        - 现在：tie-break 选“最小下标”，保证 eval 完全确定性
        """
        if valid_actions is None:
            valid_actions = list(range(self.num_actions))

        valid_actions = [a for a in valid_actions if 0 <= a < self.num_actions]
        if not valid_actions:
            valid_actions = list(range(self.num_actions))

        eps = float(self.epsilon)

        # exploration
        if eps > 1e-12 and random.random() < eps:
            a = int(random.choice(valid_actions))
        else:
            q_sub = self.q_values[valid_actions]
            max_q = float(np.max(q_sub))
            cand = [a for a in valid_actions if float(self.q_values[a]) == max_q]
            a = int(min(cand))  # ✅ deterministic tie-break

        self.last_action = int(a)
        self.last_valid_actions = list(valid_actions)
        return int(a)

    def update(self, reward: float):
        if self.last_action is None:
            return

        a = int(self.last_action)
        if not (0 <= a < self.num_actions):
            return

        r = float(reward)
        if self.reward_clip is not None and self.reward_clip > 0:
            r = float(np.clip(r, -self.reward_clip, self.reward_clip))

        old_q = float(self.q_values[a])
        new_q = (1.0 - self.lr) * old_q + self.lr * r
        self.q_values[a] = float(new_q)

