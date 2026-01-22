"""
Projekt: Space Invaders – ewaluacja wytrenowanego agenta PPO

Opis:
Skrypt służy do ilościowej oceny wytrenowanego agenta
reinforcement learning (PPO – Proximal Policy Optimization)
w środowisku Atari SpaceInvaders (ALE) z obserwacjami obiektowymi (OCAtari).

Agent:
- korzysta z wcześniej zapisanego modelu PPO,
- działa deterministycznie (bez eksploracji),
- NIE uczy się podczas ewaluacji.

Procedura:
- uruchamiane jest N epizodów gry,
- dla każdego epizodu obliczana jest suma nagród,
- na końcu wyznaczana jest średnia nagroda (AVG reward).

Wymagane pliki:
- ppo_spaceinvaders_objects.zip   – wytrenowany model PPO
- vecnormalize_spaceinvaders.pkl – statystyki normalizacji obserwacji

Autorzy:
Błażej Kanczkowski s26836
Adam Rzepa s27424
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_ppo_spaceinvaders_objects import SpaceInvadersObjectEnv


def main():
    """
    Uruchamia ewaluację wytrenowanego agenta PPO
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

    vec_env = VecNormalize.load("vecnormalize_spaceinvaders.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load("ppo_spaceinvaders_objects", env=vec_env)

    n_episodes = 10
    rewards = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = [False]
        ep_rew = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_rew += float(reward[0])

        rewards.append(ep_rew)
        print(f"Episode {ep + 1}: reward = {ep_rew:.2f}")

    avg = sum(rewards) / len(rewards)
    print(f"\nAVG (trained) reward over {n_episodes} episodes: {avg:.2f}")


if __name__ == "__main__":
    main()
