"""
Projekt: Space Invaders – kontynuacja treningu agenta PPO

Opis:
Skrypt umożliwia dalsze trenowanie (dotrenowanie) agenta
reinforcement learning (PPO – Proximal Policy Optimization)
w środowisku Atari SpaceInvaders (ALE) z obserwacjami obiektowymi (OCAtari).

Kontynuacja treningu:
- wczytywany jest wcześniej zapisany model PPO,
- wczytywane są te same statystyki normalizacji obserwacji (VecNormalize),
- licznik kroków NIE jest resetowany (reset_num_timesteps=False),
- model uczy się dalej na nowych krokach środowiska.

Pliki wejściowe / wyjściowe:
- ppo_spaceinvaders_objects.zip   – model PPO (nadpisywany)
- vecnormalize_spaceinvaders.pkl – statystyki normalizacji (nadpisywane)

Autorzy:
Błażej Kanczkowski s26836
Adam Rzepa s27424
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_ppo_spaceinvaders_objects import SpaceInvadersObjectEnv


def main():
    """
    Kontynuuje trening wcześniej wytrenowanego agenta PPO
    i zapisuje zaktualizowany model oraz statystyki normalizacji.
    """

    def make_env():
        """
        Fabryka środowiska:
        render=False – brak renderowania (szybszy trening),
        shaping=True – reward shaping włączony (jak w treningu).
        """
        return SpaceInvadersObjectEnv(render=False, shaping=True)

    vec_env = DummyVecEnv([make_env])

    vec_env = VecNormalize.load("vecnormalize_spaceinvaders.pkl", vec_env)
    vec_env.training = True
    vec_env.norm_reward = False

    model = PPO.load("ppo_spaceinvaders_objects", env=vec_env)

    model.learn(total_timesteps=800_000, reset_num_timesteps=False)

    model.save("ppo_spaceinvaders_objects")
    vec_env.save("vecnormalize_spaceinvaders.pkl")

    vec_env.close()
    print("Done. Continued training + saved.")


if __name__ == "__main__":
    main()
