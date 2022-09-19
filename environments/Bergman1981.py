# Human Glucose - Insulin System Model by Bergman 1981
# A simple 3 equation model
# Chirath Hettiarachchi
# ANU, August 2020.

# States = [Plasma Glucose(G), Plasma Insulin Remote Compartment(X), Plasma Insulin(I)]
# Research Paper:


class Bergman1981:
    def __init__(self):
        self.Gb = 4.5  # mmol/L
        self.Xb = 15  # mU/L
        self.Ib = 15  # mU/L
        self.P1 = 0.028735  # min-1
        self.P2 = 0.028344  # min-1
        self.P3 = 5.035 * (10 ** -5)
        self.V1 = 12  # L
        self.n = float(5/54)  # min

    def step(self, cur_states, action):
        g = cur_states[0, 0]
        x = cur_states[0, 1]
        i = cur_states[0, 2]
        d = action[0, 1]
        u = action[0, 0]
        dgdt = -1 * self.P1 * (g - self.Gb) - (x - self.Xb) * g + d
        dxdt = -1 * self.P2 * (x - self.Xb) + self.P3 * (i - self.Ib)
        didt = -1 * self.n * i + u / self.V1
        return [dgdt, dxdt, didt]
