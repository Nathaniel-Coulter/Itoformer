import torch
from itoformer.losses.no_arbitrage import (
    butterfly_convexity_loss,
    calendar_monotonicity_loss,
    hjm_drift_consistency_loss,
)

def test_butterfly_zero_on_convex():
    K = torch.linspace(-2, 2, 41)
    iv = 0.2 + 0.05*K**2
    surf = iv.unsqueeze(0).repeat(3, 1)
    assert float(butterfly_convexity_loss(surf.unsqueeze(0))) < 1e-8

def test_calendar_penalizes_violation():
    K = torch.linspace(-2, 2, 41)
    iv = 0.2 + 0.05*K**2
    iv_bad = torch.stack([iv*1.1, iv, iv*0.9], dim=0)
    assert float(calendar_monotonicity_loss(iv_bad.unsqueeze(0))) > 0.0

def test_hjm_trend_gt_flat():
    flat = torch.zeros(2, 16, 11) + 0.02
    trend = torch.linspace(0, 0.01, 16).view(1, 16, 1).repeat(2, 1, 11)
    assert hjm_drift_consistency_loss(trend) > hjm_drift_consistency_loss(flat)
