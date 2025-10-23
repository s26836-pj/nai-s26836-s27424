"""
Projekt: Sterowanie odwróconym wahadłem (CartPole) przy użyciu logiki rozmytej

Opis problemu:
Program implementuje regulator logiki rozmytej (FLC) do rozwiązania problemu
CartPole-v1 ze środowiska Gymnasium. Celem jest utrzymanie kija (wahadła)
w pozycji pionowej poprzez sterowanie wózkiem w lewo lub w prawo.

Regulator bazuje na systemie Mamdani (biblioteka scikit-fuzzy) i wykorzystuje
cztery zmienne wejściowe:
1.  theta: Kąt nachylenia kija.
2.  theta_dot: Prędkość kątowa kija.
3.  x_tip: Przewidywana pozycja pozioma czubka kija.
4.  x_dot: Prędkość wózka.
Wyjściem jest 'force_dir', określająca kierunek i siłę (w zakresie [-1, 1]),
która następnie jest dyskretyzowana na akcje (0 lub 1) z uwzględnieniem histerezy.

Autorzy:
    - [Błażej Kanczkowski s26836]
    - [Adam Rzepa s27424]

Instrukcja przygotowania środowiska:
Aby uruchomić program, wymagany jest Python 3.10+ oraz następujące biblioteki:

1.  Utwórz wirtualne środowisko (opcjonalnie, ale zalecane):
    python -m venv venv
    source venv/bin/activate  # (Linux/macOS)
    venv\Scripts\activate      # (Windows)

2.  Zainstaluj wymagane pakiety:
    pip install -r requirements.txt

3.  Uruchom symulację:
    python nazwa_twojego_pliku.py

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import gymnasium as gym
import skfuzzy as fuzz
import skfuzzy.control as ctrl

env = gym.make("CartPole-v1")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

print("x_threshold =", env.unwrapped.x_threshold)
print("theta_threshold_radians =", env.unwrapped.theta_threshold_radians)

env = gym.make("CartPole-v1")
theta_dot_values = []

for _ in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        _, _, _, theta_dot = obs
        theta_dot_values.append(theta_dot)
        if terminated or truncated:
            break

print("theta_dot min:", np.min(theta_dot_values))
print("theta_dot max:", np.max(theta_dot_values))


@dataclass
class FuzzyConfiguration:
    """
    Przechowuje parametry konfiguracyjne dla regulatora rozmytego CartPole.

    Atrybuty:
        theta_range (tuple): Zakres dla zmiennej 'theta' [rad].
        theta_dot_range (tuple): Zakres dla zmiennej 'theta_dot' [rad/s].
        force_range (tuple): Zakres dla wyjściowej zmiennej 'force_dir' [-1, 1].
        resolution (int): Liczba punktów w uniwersum (przestrzeni) każdej zmiennej.
        hyst (float): Wartość progu histerezy do dyskretyzacji akcji.
        edge_hard (float): Bezwzględna wartość pozycji 'x', przy której
                           wymuszana jest akcja powrotu (reguła "twardej krawędzi").
        x_dot_range (tuple): Zakres dla zmiennej 'x_dot' [m/s].
        x_tip_range (tuple): Zakres dla zmiennej 'x_tip' (pozycji czubka) [m].
    """
    theta_range: tuple = (-0.21, 0.21)
    theta_dot_range: tuple = (-3.0, 3.0)
    force_range: tuple = (-1.0, 1.0)
    resolution: int = 101

    hyst: float = 0.35
    edge_hard: float = 2.0

    x_dot_range: tuple = (-2.0, 2.0)
    x_tip_range: tuple = (-2.6, 2.6)


class CartPoleFuzzyController:
    """
    Regulator rozmyty (Mamdani) dla zadania CartPole.

    Implementuje logikę sterowania opartą na 4 wejściach i 1 wyjściu
    do stabilizacji odwróconego wahadła.

    Wejścia (Antecedents):
      - x_tip: pozycja pozioma czubka kija [m]
      - x_dot: prędkość wózka [m/s]
      - theta: kąt kija [rad]
      - theta_dot: prędkość kątowa [rad/s]

    Wyjścia (Consequents):
      - force_dir: Kierunek siły w zakresie [-1.0, 1.0].
    """

    def __init__(self, cfg: FuzzyConfiguration | None = None) -> None:
        """
                Inicjalizuje regulator rozmyty.

                Args:
                    cfg (FuzzyConfiguration | None): Obiekt konfiguracji.
                        Jeśli None, używa domyślnych wartości z FuzzyConfiguration.
                """
        self.cfg = cfg or FuzzyConfiguration()

        temp_env = gym.make("CartPole-v1")
        self.pole_length = temp_env.unwrapped.length
        temp_env.close()
        print(f"Długość kija (L) = {self.pole_length}")

        self.x_tip_u = np.linspace(*self.cfg.x_tip_range, self.cfg.resolution)
        self.x_dot_u = np.linspace(*self.cfg.x_dot_range, self.cfg.resolution)
        self.theta_u = np.linspace(*self.cfg.theta_range, self.cfg.resolution)
        self.theta_dot_u = np.linspace(*self.cfg.theta_dot_range, self.cfg.resolution)
        self.force_u = np.linspace(*self.cfg.force_range, self.cfg.resolution)

        self.x_tip = ctrl.Antecedent(self.x_tip_u, "x_tip")
        self.x_dot = ctrl.Antecedent(self.x_dot_u, "x_dot")
        self.theta = ctrl.Antecedent(self.theta_u, "theta")
        self.theta_dot = ctrl.Antecedent(self.theta_dot_u, "theta_dot")

        self.force_dir = ctrl.Consequent(self.force_u, "force_dir")

        self.define_mfs()

        rules = self._define_rules()

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)
        self.prev_action = 1


    def define_mfs(self) -> None:
        """Definiuje funkcje przynależności dla wejść i wyjścia."""

        xtip_min, xtip_max = self.cfg.x_tip_range
        xdmin, xdmax = self.cfg.x_dot_range
        tmin, tmax = self.cfg.theta_range
        tdmin, tdmax = self.cfg.theta_dot_range

        self.x_tip["L"] = fuzz.trimf(self.x_tip_u, [xtip_min, xtip_min, 0.0])
        self.x_tip["C"] = fuzz.trimf(self.x_tip_u, [xtip_min * 0.15, 0.0, xtip_max * 0.15])
        self.x_tip["R"] = fuzz.trimf(self.x_tip_u, [0.0, xtip_max, xtip_max])

        self.x_dot["L"] = fuzz.trimf(self.x_dot_u, [xdmin, xdmin, 0.0])
        self.x_dot["Z"] = fuzz.trimf(self.x_dot_u, [xdmin * 0.3, 0.0, xdmax * 0.3])
        self.x_dot["R"] = fuzz.trimf(self.x_dot_u, [0.0, xdmax, xdmax])

        self.theta["L"] = fuzz.trimf(self.theta_u, [tmin, tmin, 0.0])
        self.theta["Z"] = fuzz.trimf(self.theta_u, [tmin * 0.3, 0.0, tmax * 0.3])
        self.theta["R"] = fuzz.trimf(self.theta_u, [0.0, tmax, tmax])

        self.theta_dot["LF"] = fuzz.trimf(self.theta_dot_u, [tdmin, tdmin, 0.0])
        self.theta_dot["Z"] = fuzz.trimf(self.theta_dot_u, [tdmin * 0.3, 0.0, tdmax * 0.3])
        self.theta_dot["RF"] = fuzz.trimf(self.theta_dot_u, [0.0, tdmax, tdmax])

        self.force_dir["L"] = fuzz.trimf(self.force_u, [-1.0, -1.0, 0.0])
        self.force_dir["Z"] = fuzz.trimf(self.force_u, [-0.2, 0.0, 0.2])
        self.force_dir["R"] = fuzz.trimf(self.force_u, [0.0, 1.0, 1.0])

    def _define_rules(self) -> List[ctrl.Rule]:
        """Definiuje reguły bazujące na x_tip (pozycji czubka)."""
        rules = [
            ctrl.Rule(self.theta["R"] & self.theta_dot["RF"], self.force_dir["R"]),
            ctrl.Rule(self.theta["L"] & self.theta_dot["LF"], self.force_dir["L"]),
            ctrl.Rule(self.theta["R"] & self.theta_dot["Z"], self.force_dir["R"]),
            ctrl.Rule(self.theta["L"] & self.theta_dot["Z"], self.force_dir["L"]),
            ctrl.Rule(self.theta["Z"] & self.theta_dot["RF"], self.force_dir["R"]),
            ctrl.Rule(self.theta["Z"] & self.theta_dot["LF"], self.force_dir["L"]),

            ctrl.Rule(self.theta["Z"] & self.x_tip["R"] & self.x_dot["R"], self.force_dir["L"]),
            ctrl.Rule(self.theta["Z"] & self.x_tip["R"] & self.x_dot["L"], self.force_dir["Z"]),

            ctrl.Rule(self.theta["Z"] & self.x_tip["R"] & self.x_dot["Z"], self.force_dir["R"]),

            ctrl.Rule(self.theta["Z"] & self.x_tip["L"] & self.x_dot["L"], self.force_dir["R"]),
            ctrl.Rule(self.theta["Z"] & self.x_tip["L"] & self.x_dot["R"], self.force_dir["Z"]),

            ctrl.Rule(self.theta["Z"] & self.x_tip["L"] & self.x_dot["Z"], self.force_dir["L"]),

            ctrl.Rule(self.theta["Z"] & self.theta_dot["Z"] & self.x_tip["C"] & self.x_dot["Z"],
                      self.force_dir["Z"]),
        ]
        return rules


    def compute_force(self, x_tip: float, x_dot: float, theta: float, theta_dot: float) -> float:
        """
        Uruchamia symulację systemu rozmytego, aby obliczyć wyjściową siłę.

        Args:
            x_tip (float): Pozycja czubka kija.
            x_dot (float): Prędkość wózka.
            theta (float): Kąt kija.
            theta_dot (float): Prędkość kątowa kija.

        Returns:
            float: Obliczona wartość siły (force_dir) z zakresu [-1.0, 1.0].
                   Zwraca 0.0 w przypadku błędu symulacji.
        """

        x_tip = float(np.clip(x_tip, *self.cfg.x_tip_range))
        x_dot = float(np.clip(x_dot, *self.cfg.x_dot_range))
        theta = float(np.clip(theta, *self.cfg.theta_range))
        theta_dot = float(np.clip(theta_dot, *self.cfg.theta_dot_range))

        self.sim.input["x_tip"] = x_tip
        self.sim.input["x_dot"] = x_dot
        self.sim.input["theta"] = theta
        self.sim.input["theta_dot"] = theta_dot

        try:
            self.sim.compute()
        except Exception:
            return 0.0

        if "force_dir" not in self.sim.output:
            return 0.0

        return float(self.sim.output["force_dir"])

    def decide_action(self, x: float, x_dot: float, theta: float, theta_dot: float) -> int:
        """
        Podejmuje ostateczną, dyskretną decyzję o akcji (0 lub 1).

        Wykorzystuje reguły "twardej krawędzi" oraz przetwarza wyjście
        systemu rozmytego przez histerezę, aby wybrać akcję.

        Args:
            x (float): Surowa pozycja wózka (ze środowiska).
            x_dot (float): Surowa prędkość wózka.
            theta (float): Surowy kąt kija.
            theta_dot (float): Surowa prędkość kątowa kija.

        Returns:
            int: Dyskretna akcja dla środowiska (0=lewo, 1=prawo).
        """
        cfg = self.cfg

        if x > cfg.edge_hard and x_dot > 0.05:
            self.prev_action = 0
            return 0
        if x < -cfg.edge_hard and x_dot < -0.05:
            self.prev_action = 1
            return 1

        x_tip = x + self.pole_length * np.sin(theta)

        force = self.compute_force(x_tip, x_dot, theta, theta_dot)

        force = float(np.clip(force, -1.0, 1.0))

        H = cfg.hyst
        a = self.prev_action
        if a == 1:
            a = 0 if force < -H else 1
        else:
            a = 1 if force > +H else 0

        self.prev_action = a
        return a

def run_episodes(num_episodes: int = 10, max_steps: int = 500, render: bool = True) -> None:
    """
    Uruchamia symulację środowiska CartPole-v1 ze sterownikiem rozmytym.

    Args:
        num_episodes (int): Liczba epizodów do uruchomienia.
        max_steps (int): Maksymalna liczba kroków w pojedynczym epizodzie.
        render (bool): Czy renderować środowisko graficznie (tryb 'human').
    """

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent = CartPoleFuzzyController()
    returns = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total = 0.0

        for t in range(max_steps):
            x, x_dot, theta, theta_dot = obs

            action = agent.decide_action(x, x_dot, theta, theta_dot)

            obs, reward, terminated, truncated, info = env.step(action)
            total += reward

            if terminated or truncated:
                break

        returns.append(total)
        print(f"[Ep {ep:02d}] steps={int(total)} reward={total:.0f}")

    env.close()

    mean = float(np.mean(returns)) if returns else 0.0
    std = float(np.std(returns)) if returns else 0.0
    print(f"\nŚrednia liczba kroków z {num_episodes} ep.: {mean:.1f} ± {std:.1f}")


if __name__ == "__main__":

    run_episodes(num_episodes=5, max_steps=500, render=True)
