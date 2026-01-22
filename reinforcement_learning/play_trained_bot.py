"""
Projekt: Space Invaders – uruchomienie wytrenowanego agenta PPO

Opis:
Skrypt służy do uruchomienia (playback) wcześniej wytrenowanego agenta
reinforcement learning (PPO – Proximal Policy Optimization) w środowisku
Atari SpaceInvaders (ALE).

Agent został wytrenowany na obserwacjach obiektowych (OCAtari),
a nie na surowych pikselach. W tym trybie:
- model NIE jest trenowany dalej,
- polityka działa deterministycznie (deterministic=True),
- środowisko renderuje przebieg gry w oknie.

Wymagane pliki:
- ppo_spaceinvaders_objects.zip   – zapisany model PPO
- vecnormalize_spaceinvaders.pkl – statystyki normalizacji obserwacji

Autorzy:
Błażej Kanczkowski s26836
Adam Rzepa s27424
"""

import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_ppo_spaceinvaders_objects import SpaceInvadersObjectEnv


def main():
    """
    Uruchamia wytrenowanego agenta PPO w trybie gry (bez dalszego uczenia).

    Agent wykonuje akcje w pętli nieskończonej, a po zakończeniu epizodu
    środowisko jest resetowane i gra rozpoczyna się od nowa.
    """

    def make_env():
        """
        Fabryka środowiska – render=True umożliwia obserwację gry,
        shaping=False wyłącza reward shaping (czysta gra).
        """
        return SpaceInvadersObjectEnv(render=True, shaping=False)

    vec_env = DummyVecEnv([make_env])

    vec_env = VecNormalize.load("vecnormalize_spaceinvaders.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load("ppo_spaceinvaders_objects", env=vec_env)

    obs = vec_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        time.sleep(1 / 60)

        if done[0]:
            obs = vec_env.reset()


if __name__ == "__main__":
    main()
