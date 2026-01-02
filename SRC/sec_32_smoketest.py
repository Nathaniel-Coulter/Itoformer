import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from itoformer.losses.no_arbitrage import (
    butterfly_convexity_loss,
    calendar_monotonicity_loss,
    hjm_drift_consistency_loss,
)

import torch

# 1) Butterfly: convex quad surface should have ~0 loss
K = torch.linspace(-2, 2, 41)                 # moneyness grid
T = torch.tensor([0.1, 0.5, 1.0])             # three tenors
iv = 0.2 + 0.05*K**2                          # convex in K
surf = iv.unsqueeze(0).repeat(len(T), 1)      # [T, K] same across T
loss_bfly = butterfly_convexity_loss(surf.unsqueeze(0))  # -> [B=1, T, K]
print("butterfly convexity (convex) ≈ 0 ->", float(loss_bfly))

# 2) Calendar: total variance increasing in T → ~0 loss
ivT = torch.stack([iv, iv*1.05, iv*1.10], dim=0)   # IV grows with T
loss_cal = calendar_monotonicity_loss(ivT.unsqueeze(0))
print("calendar monotonicity (increasing) ≈ 0 ->", float(loss_cal))

# 3) Calendar negative case: later tenor LOWER total variance → positive penalty
iv_bad = torch.stack([iv*1.10, iv, iv*0.90], dim=0)
print("calendar monotonicity (violations) > 0 ->", float(calendar_monotonicity_loss(iv_bad.unsqueeze(0))))

# 4) HJM drift placeholder: i.i.d. flat curve small penalty; trend larger
fwd_flat = torch.zeros(8, 64, 11) + 0.02     # [B=8, L=64, N=11]
print("hjm (flat) small ->", float(hjm_drift_consistency_loss(fwd_flat)))
trend = torch.linspace(0, 0.01, 64).view(1, 64, 1).repeat(8, 1, 11)
print("hjm (trend) bigger ->", float(hjm_drift_consistency_loss(trend)))
