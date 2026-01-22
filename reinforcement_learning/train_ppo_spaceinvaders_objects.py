"""
Projekt: Space Invaders – agent RL (PPO) na obiektach z OCAtari

Opis problemu:
Program trenuje agenta w środowisku Atari SpaceInvaders (ALE) metodą
reinforcement learning (PPO – Proximal Policy Optimization).

Zamiast trenować na surowych pikselach, wykorzystujemy bibliotekę OCAtari,
która dostarcza listę obiektów w scenie (np. Player, Alien, Bullet).
Na podstawie obiektów budujemy wektor cech (8 wartości) i to on jest obserwacją
dla agenta uczącego się.

Nagroda:
- bazowa nagroda pochodzi bezpośrednio z gry (za trafienia itd.)
- dodatkowo stosowany jest reward shaping (małe premie/kary), aby przyspieszyć naukę:
  * mały bonus za ustawianie się pod celem (alienem)
  * kara za zagrożenie pociskiem wroga (enemy bullet)
  * bardzo mała kara za NOOP (żeby agent nie stał w miejscu)

Rozróżnianie pocisków:
OCAtari nie zawsze rozróżnia wprost “czyj” pocisk, dlatego wykrywamy pociski wroga
na podstawie kierunku ruchu (dy > 0 => pocisk spada w dół => enemy bullet).

Pliki wyjściowe:
- ppo_spaceinvaders_objects.zip  (wytrenowany model PPO)
- vecnormalize_spaceinvaders.pkl (statystyki normalizacji obserwacji)

Autorzy:
Błażej Kanczkowski s26836
Adam Rzepa s27424

Instrukcja uruchomienia:
1) Utwórz środowisko wirtualne (venv):
   python -m venv venv
   ---
2) Aktywuj środowisko wirtualne:
   Windows:
   venv\Scripts\activate
   Linux / macOS:
   source venv/bin/activate
   ---
3) Zainstaluj wymagane biblioteki:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ---
4) Uruchom trening agenta PPO (od zera):
   python train_ppo_spaceinvaders_objects.py
   ---
5) Ewaluacja wytrenowanego agenta (bez renderowania):
   python eval_ppo.py
   ---
6) Porównanie z losowym agentem (baseline):
   python eval_random.py
   ---
7) Uruchomienie gry z wytrenowanym agentem (render):
   python play_trained_ppo.py
   ---
8) Kontynuowanie treningu (dotrenowanie modelu):
   python continue_train_ppo.py
--------------------------------------------------------------
1) python train_ppo_spaceinvaders_objects.py
2) (opcjonalnie) uruchom eval_ppo.py / play_trained_ppo.py
3) kontynuowanie treningu continue_train_ppo.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ocatari.core import OCAtari

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Atari mapping (SpaceInvaders Discrete(6))
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5


def is_type(obj, name: str) -> bool:
    """
    Sprawdza typ obiektu OCAtari na podstawie jego reprezentacji tekstowej.
    W praktyce str(obj) ma postać np. 'Player at (x, y), (w, h)'.
    """
    return str(obj).lower().startswith(name.lower())


def get_xy(obj):
    """
    Zwraca (x, y) obiektu OCAtari, jeśli atrybuty istnieją.
    """
    return getattr(obj, "x", None), getattr(obj, "y", None)


class SpaceInvadersObjectEnv(gym.Env):
    """
    Wrapper środowiska OCAtari do postaci kompatybilnej z Stable-Baselines3:

    Obserwacja:
        wektor 8 cech (float32) w zakresie ~[-1, 1]
        [px, target_dx, target_y, danger, bullet_dx, bullet_y, aliens_alive, bias]

    Akcje:
        Discrete(6) (NOOP/FIRE/LEFT/RIGHT/LEFTFIRE/RIGHTFIRE)

    Nagroda:
        reward z gry + mały reward shaping (opcjonalnie).
    """
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render: bool = False, shaping: bool = True):
        """
        Parametry:
            render  – czy uruchamiać okno gry (human)
            shaping – czy dodawać reward shaping do nagrody z gry
        """
        super().__init__()
        self.render = render
        self.shaping = shaping

        self.env = OCAtari(
            "ALE/SpaceInvaders-v5",
            mode="ram",
            render_mode="human" if render else None
        )

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._step = 0
        self._prev_bullets = []

    def _extract_features(self) -> np.ndarray:
        """
        Ekstrakcja cech z env.objects:
        - pozycja gracza (px)
        - wektor do wybranego celu (target_dx, target_y)
        - wykrywanie zagrożenia pociskiem wroga (danger)
        - pozycja “najgroźniejszego” enemy bullet (bullet_dx, bullet_y)
        - flaga obecności kosmitów (aliens_alive)
        """
        objs = self.env.objects

        players = [o for o in objs if is_type(o, "Player")]
        aliens = [o for o in objs if is_type(o, "Alien")]
        bullets = [o for o in objs if any(str(o).lower().startswith(t) for t in ["bullet", "missile", "shot"])]

        px = 0.0
        target_dx = 0.0
        target_y = -1.0
        danger = 0.0
        bullet_dx = 0.0
        bullet_y = -1.0
        aliens_alive = 0.0

        if players:
            pxx, _ = get_xy(players[0])
            if pxx is not None:
                px = (pxx / 80.0) - 1.0  # 0..160 -> -1..1

        px_pix = None
        if players:
            px_pix, _ = get_xy(players[0])

        if aliens and px_pix is not None:
            aliens_alive = 1.0
            best = None
            best_score = 1e9

            for a in aliens:
                ax, ay = get_xy(a)
                if ax is None or ay is None:
                    continue
                dx = abs(ax - px_pix)
                score = dx + (60 - ay) * 0.05
                if score < best_score:
                    best_score = score
                    best = a

            if best is not None:
                ax, ay = get_xy(best)
                target_dx = np.clip((ax - px_pix) / 80.0, -1.0, 1.0)
                target_y = np.clip((ay / 105.0) - 1.0, -1.0, 1.0)

        enemy_bullets = []
        if bullets:
            curr = []
            for b in bullets:
                bx, by = get_xy(b)
                if bx is None or by is None:
                    continue
                curr.append((bx, by))

            for (bx, by) in curr:
                dy = None
                if self._prev_bullets:
                    px0, py0 = min(
                        self._prev_bullets,
                        key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2
                    )
                    if (px0 - bx) ** 2 + (py0 - by) ** 2 < 25:
                        dy = by - py0

                if dy is not None and dy > 0:
                    enemy_bullets.append((bx, by))

            self._prev_bullets = curr
        else:
            self._prev_bullets = []

        if enemy_bullets and px_pix is not None:
            best_by = -1
            best_bx = None
            best_by_val = None

            for bx, by in enemy_bullets:
                if by > best_by:
                    best_by = by
                    best_bx = bx
                    best_by_val = by

                if by > 80 and abs(bx - px_pix) <= 6:
                    danger = 1.0

            if best_bx is not None:
                bullet_dx = np.clip((best_bx - px_pix) / 80.0, -1.0, 1.0)
                bullet_y = np.clip((best_by_val / 105.0) - 1.0, -1.0, 1.0)

        return np.array([px, target_dx, target_y, danger, bullet_dx, bullet_y, aliens_alive, 1.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset środowiska (start nowego epizodu).
        Czyści też pamięć poprzednich pocisków.
        """
        _, info = self.env.reset(seed=seed)
        self._step = 0
        self._prev_bullets = []
        return self._extract_features(), info

    def step(self, action):
        """
        Wykonuje akcję w środowisku i zwraca:
        (obs, reward_total, terminated, truncated, info)
        """
        _, reward, terminated, truncated, info = self.env.step(int(action))
        self._step += 1

        obs = self._extract_features()

        shaping_reward = 0.0
        if self.shaping:
            _, target_dx, _, danger, _, _, aliens_alive, _ = obs
            shaping_reward += 0.01 * (1.0 - min(1.0, abs(target_dx))) if aliens_alive > 0.5 else 0.0
            shaping_reward -= 0.02 * danger
            if int(action) == NOOP:
                shaping_reward -= 0.001

        return obs, float(reward) + float(shaping_reward), terminated, truncated, info

    def close(self):
        """Zamyka wewnętrzne środowisko OCAtari."""
        self.env.close()


def main():
    """
    Uruchamia trening PPO na środowisku SpaceInvadersObjectEnv,
    zapisuje model i statystyki normalizacji.
    """
    def make_env():
        return SpaceInvadersObjectEnv(render=False, shaping=True)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=300_000)
    model.save("ppo_spaceinvaders_objects")
    vec_env.save("vecnormalize_spaceinvaders.pkl")

    vec_env.close()
    print("Saved: ppo_spaceinvaders_objects + vecnormalize_spaceinvaders.pkl")


if __name__ == "__main__":
    main()
