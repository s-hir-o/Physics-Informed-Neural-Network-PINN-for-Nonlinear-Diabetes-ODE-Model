import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# PARAMETERS
# =============================
Λ = 11/1000
β = 0.1478
η = 1.0573
α1 = 0.1346
α2 = 0.8653
ω = 0.31
ψ = 0.7634
δ = 0.6832
θ = 0.0098
ω1 = 0.0235
ω2 = 0.5
φ = 0.0235
γ = 0.4495
σ1 = 0.013
σ2 = 0.2160
σ3 = 0.1381
μ = 0.0128

N_max = Λ / μ  # ≈ 0.86

# =============================
# INITIAL CONDITIONS (FIXED)
# =============================
D0 = 0.05
TN0 = 0.01 * D0
TP0 = 0.02 * D0
R0  = 0.005 * D0
S0  = N_max - (D0 + TN0 + TP0 + R0)

# =============================
# TIME NORMALIZATION
# =============================
T_max = 50.0

def normalize_t(t):
    return t / T_max

def denormalize_dt(dt_scaled):
    return dt_scaled / T_max

# =============================
# MODEL
# =============================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5)
        )

    def forward(self, t):
        return torch.nn.functional.softplus(self.net(t))

model = PINN().to(device)

# =============================
# PHYSICS LOSS (FIXED)
# =============================
def physics_loss(t):
    t_norm = normalize_t(t)
    t_norm.requires_grad = True

    out = model(t_norm)
    S, D, TN, TP, R = torch.split(out, 1, dim=1)

    N = S + D + TN + TP + R + 1e-6

    def grad(y):
        dy_dt_scaled = torch.autograd.grad(
            y, t_norm,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        return denormalize_dt(dy_dt_scaled)

    dS = grad(S)
    dD = grad(D)
    dTN = grad(TN)
    dTP = grad(TP)
    dR = grad(R)
    dN = grad(N)

    eq1 = dS - (Λ - β*(η*α1 + α2)*S - μ*S)

    eq2 = dD - (
        β*(η*α1 + α2)*S + ω*R + ω1*TN + ω2*TP
        - ψ*TN*D/N - (δ+μ+σ1)*D
    )

    eq3 = dTN - (
        ψ*TN*D/N - (θ+μ+σ2+ω1+φ)*TN
    )

    eq4 = dTP - (
        δ*D + φ*TN - (γ+μ+σ3+ω2)*TP
    )

    eq5 = dR - (
        γ*TP + θ*TN - (μ+ω)*R
    )

    # TOTAL POPULATION EQUATION (CRITICAL)
    eq6 = dN - (Λ - μ*N - (σ1*D + σ2*TN + σ3*TP))

    return (
        5*eq1.pow(2).mean() +
        10*eq2.pow(2).mean() +
        5*eq3.pow(2).mean() +
        3*eq4.pow(2).mean() +
        3*eq5.pow(2).mean() +
        5*eq6.pow(2).mean()
    )

# =============================
# INITIAL CONDITION LOSS (FIXED)
# =============================
def initial_loss():
    t0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    pred = model(t0)[0]

    return ((pred[0]-S0)**2 +
            (pred[1]-D0)**2 +
            (pred[2]-TN0)**2 +
            (pred[3]-TP0)**2 +
            (pred[4]-R0)**2)

# =============================
# BOUNDEDNESS LOSS (FIXED)
# =============================
def population_loss(t):
    t_norm = normalize_t(t)
    out = model(t_norm)
    N = out.sum(dim=1)

    return torch.mean(torch.relu(N - N_max)**2)

# =============================
# TRAINING
# =============================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(8000):
    t = torch.linspace(0, T_max, 500).view(-1,1).to(device)

    loss = (
        20*physics_loss(t) +
        5*initial_loss() +
        10*population_loss(t)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"{epoch} loss={loss.item():.6f}")

# =============================
# PLOT
# =============================
t_test = torch.linspace(0, T_max, 500).view(-1,1).to(device)
pred = model(normalize_t(t_test))

S, D, TN, TP, R = torch.split(pred, 1, dim=1)

S = S.detach().cpu().numpy().flatten()
D = D.detach().cpu().numpy().flatten()
TN = TN.detach().cpu().numpy().flatten()
TP = TP.detach().cpu().numpy().flatten()
R = R.detach().cpu().numpy().flatten()

t_plot = t_test.cpu().numpy().flatten()

plt.figure(figsize=(10,6))
plt.plot(t_plot, S, label="S")
plt.plot(t_plot, D, label="D")
plt.plot(t_plot, TN, label="TN")
plt.plot(t_plot, TP, label="TP")
plt.plot(t_plot, R, label="R")

plt.xlabel("Time (Years)")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()