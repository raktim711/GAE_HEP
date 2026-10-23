import numpy as np
import torch

from gaehep.models.latent_flow import LatentFlow

torch.set_num_threads(1)


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 1, (n, 1)).astype(np.float32)
    z = np.c_[2.0 * c[:, 0] + 0.1 * rng.normal(size=n), rng.normal(size=n)].astype(np.float32)
    return z, c


def test_flow_scores_outliers_higher():
    z, _ = _toy(4000, 0)
    zv, _ = _toy(1000, 1)
    flow = LatentFlow(2, transforms=2, hidden=(32, 32)).fit(z, zv, epochs=6, batch_size=256)
    inl = flow.nll(zv[:200])
    out = flow.nll(zv[:200] + np.array([0.0, 6.0], dtype=np.float32))
    assert np.median(out) > np.median(inl) + 3
    assert len(flow.history["val_nll"]) >= 1


def test_conditional_flow_uses_context():
    z, c = _toy(4000, 0)
    zv, cv = _toy(1000, 1)
    cond = LatentFlow(2, transforms=2, hidden=(32, 32), context_dim=1).fit(
        z, zv, epochs=6, batch_size=256, c_train=c, c_val=cv
    )
    plain = LatentFlow(2, transforms=2, hidden=(32, 32)).fit(z, zv, epochs=6, batch_size=256)
    # z_0 is almost determined by the context, so the conditional density is much sharper
    assert cond.nll(zv, context=cv).mean() < plain.nll(zv).mean() - 0.5
    # same z with a mismatched context is unlikely under the conditional flow
    assert np.median(cond.nll(zv, context=-cv)) > np.median(cond.nll(zv, context=cv)) + 3
