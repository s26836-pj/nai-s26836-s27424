"""
Projekt: Space Invaders – ewaluacja losowego agenta (baseline)

Opis:
Skrypt służy do oceny zachowania losowego agenta (random policy)
w środowisku Atari SpaceInvaders (ALE) z obserwacjami obiektowymi (OCAtari).

Losowy agent:
- w każdej klatce wybiera akcję losowo,
- NIE uczy się,
- stanowi punkt odniesienia (baseline) do porównania z agentem PPO.

Procedura:
- uruchamiane jest N epizodów gry,
- dla każdego epizodu liczona jest suma nagród,
- na końcu obliczana jest średnia nagroda.

Wymagane pliki:
- vecnormalize_spaceinvaders.pkl (statystyki normalizacji z treningu)

Autorzy:
Błażej Kanczkowski s26836
Adam Rzepa s27424
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_ppo_spaceinvaders_objects import SpaceInvadersObjectEnv


def main():
    """
    Uruchamia ewaluację losowego agenta w środowisku SpaceInvaders
    i wypisuje nagrody dla kolejnych epizodów oraz średnią.
    """

    def make_env():
        """
        Fabryka środowiska:
        render=False – brak renderowania (szybsza ewaluacja),
        shaping=False – brak reward shaping (czysta gra).
        """
        return SpaceInvadersObjectEnv(render=False, shaping=False)

    vec_env = DummyVecEnv([make_env])

    # Wczytaj normalizację obserwacji z treningu
    vec_env = VecNormalize.load("vecnormalize_spaceinvaders.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    n_episodes = 10
    rewards = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = [False]
        ep_rew = 0.0

        while not done[0]:
            # losowa akcja (baseline)
            action = np.array([vec_env.action_space.sample()])
            obs, reward, done, info = vec_env.step(action)
            ep_rew += float(reward[0])

        rewards.append(ep_rew)
        print(f"Episode {ep + 1}: reward = {ep_rew:.2f}")

    avg = sum(rewards) / len(rewards)
    print(f"\nAVG (random) reward over {n_episodes} episodes: {avg:.2f}")


if __name__ == "__main__":
    main()
