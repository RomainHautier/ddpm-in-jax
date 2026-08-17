"""Data-parallel batched sampling for the grading/sweep harnesses.

Training always pmapped across all TPU chips; grading never did — every sampler call ran
jax.jit on ONE device while the others idled, which is why the setpoint sweep took days.
This helper shards the same per-chunk closures across all local devices.

Contract: `fn(xb, kk)` is exactly the closure the serial `batched()` helpers take — model
params closed over (they become broadcast constants of the pmapped executable), xb a batch of
inputs, kk a PRNG key. pbatched pmaps fn, processes the pool in chunks of n_dev*per_dev, pads
the tail chunk by repeating its last row (padding sliced off after), and folds the chunk key
by chunk start index like the serial convention, then splits it per device.

NOISE CAVEAT: per-device key splitting means the sampling-noise draws differ from the serial
helper's (and from historically graded cells). That difference is the documented eval-seed
noise floor (~±0.015 retention); anything smaller than it was never a finding. The driver
validates this against an already-graded cell before unattended use, and PSAMPLE=0 in the
environment restores the exact serial behavior.
"""
import os
import numpy as np, jax, jax.numpy as jnp

SERIAL = os.environ.get('PSAMPLE', '1') == '0'


def batched_serial(fn, x, seed, bs=16):
    """The original single-device helper, byte-identical semantics."""
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


def pbatched(fn, x, seed, per_dev=8):
    """Data-parallel version of batched_serial. Falls back to it under PSAMPLE=0 or on a
    single device."""
    nd = jax.local_device_count()
    if SERIAL or nd == 1:
        return batched_serial(fn, x, seed, bs=max(per_dev, 16) if nd == 1 else 16)
    pfn = jax.pmap(fn)
    B = nd * per_dev
    key = jax.random.PRNGKey(seed)
    out = []
    for i in range(0, len(x), B):
        xb = np.asarray(x[i:i + B])
        pad = B - len(xb)
        if pad:
            xb = np.concatenate([xb, np.repeat(xb[-1:], pad, axis=0)])
        ks = jax.random.split(jax.random.fold_in(key, i), nd)
        y = np.asarray(pfn(jnp.asarray(xb.reshape(nd, per_dev, *xb.shape[1:])), ks))
        y = y.reshape(B, *y.shape[2:])
        out.append(y[:B - pad] if pad else y)
    return np.concatenate(out)
