# import sys
# import types
# import numpy as np
# import time

# from pyrtid.utils import NDArrayFloat
# from pyrtid.forward import geochem_solver as G
# from quickpaver import RectilinearGrid

# # --- stub pyrtid so the module imports standalone ---
# for m in ["pyrtid", "pyrtid.forward", "pyrtid.forward.models", "pyrtid.utils"]:
#     sys.modules.setdefault(m, types.ModuleType(m))


# class ConstantConcentration:
#     def __init__(self, span):
#         self.span = span


# class GeochemicalParameters:
#     def __init__(s, kv=1.5, As=2.0, Ks=10.0, stocoef=1.0, exp=False):
#         s.kv, s.As, s.Ks, s.stocoef, s.use_explicit_formulation = (
#             kv,
#             As,
#             Ks,
#             stocoef,
#             exp,
#         )

# class TimeParameters:
#     def __init__(s, dt):
#         s.dt = dt


# class TransportModel:
#     pass


# class RectilinearGrid:
#     def __init__(s, nx, ny, nz):
#         s.nx, s.ny, s.nz = nx, ny, nz


# mod = sys.modules["pyrtid.forward.models"]
# for n, o in [
#     ("ConstantConcentration", ConstantConcentration),
#     ("GeochemicalParameters", GeochemicalParameters),
#     ("TimeParameters", TimeParameters),
#     ("TransportModel", TransportModel),
# ]:
#     setattr(mod, n, o)


# def make_model(nx, ny, nz, seed=0):
#     r = np.random.default_rng(seed)
#     tm = TransportModel()
#     C1 = r.uniform(0.1, 6.0, (nx, ny, nz))
#     C2 = r.uniform(0.1, 5.0, (nx, ny, nz))
#     S1 = r.uniform(0.1, 6.0, (nx, ny, nz))
#     S2 = np.zeros((nx, ny, nz))
#     tm.lmob = {0: np.stack([C1, C2]), 1: np.stack([C1, C2]).copy()}
#     tm.limmob = {0: np.stack([S1, S2]), 1: np.zeros((2, nx, ny, nz))}
#     tm.boundary_conditions = []
#     return tm


# p = GeochemicalParameters()
# print("=" * 72)
# print("1. RESIDUAL of the implicit extent equation (should be ~0)")
# print("=" * 72)
# for dt in [1e-6, 1e-3, 0.1, 1.0, 50.0, 1e4]:
#     tm = make_model(8, 8, 4)
#     C1, C2 = tm.lmob[1][0].copy(), tm.lmob[1][1].copy()
#     S1 = tm.limmob[0][0].copy()
#     xi = G.get_extent_implicit(C1, C2, S1, p, dt)
#     xmax = G.get_extent_max(C1, C2, S1, p)
#     res = xi - dt * p.kv * p.As * S1 * (C2 - p.stocoef * xi) * (1 - (C1 + xi) / p.Ks)
#     interior = (xi > 1e-14) & (
#         xi < xmax - 1e-14
#     )  # residual must vanish off the clip bounds
#     print(
#         f" dt={dt:8.1e} max|res|(interior)={np.max(np.abs(res[interior])) if interior
# .any() else 0:.3e}"
#         f"  clipped={np.count_nonzero(~interior):4d}/{xi.size}"
#     )

# print()
# print("=" * 72)
# print("2. CONSERVATION + POSITIVITY over 200 steps")
# print("=" * 72)
# for dt in [1e-3, 0.1, 5.0]:
#     tm = make_model(10, 10, 5)
#     tot0 = tm.lmob[0][0] + tm.limmob[0][0]
#     acid0 = tm.lmob[0][1] + tm.limmob[0][1]
#     neg = False
#     maxbal = 0.0
#     for step in range(200):
#         tm.lmob[1] = tm.lmob[0].copy()
#         tm.limmob[1] = np.zeros_like(tm.limmob[0])
#         G.solve_geochem_implicit(
#             RectilinearGrid(10, 10, 5), tm, p, TimeParameters(dt), 1
#         )
#         maxbal = max(maxbal, G.check_acid_balance(tm, 1))
#         neg |= bool(np.any(tm.lmob[1] < -1e-12) or np.any(tm.limmob[1] < -1e-12))
#         tm.lmob[0], tm.limmob[0] = tm.lmob[1].copy(), tm.limmob[1].copy()
#     metal = np.max(np.abs(tm.lmob[0][0] + tm.limmob[0][0] - tot0))
#     acid = np.max(np.abs(tm.lmob[0][1] + tm.limmob[0][1] - acid0))
#     print(
#         f" dt={dt:7.3f} | metal C1+S1 drift={metal:.2e} | acid C2+S2 drift={acid:.2e}"
#         f" | audit={maxbal:.2e} | any negative={neg}"
#     )

# print()
# print("=" * 72)
# print("3. EXPLICIT -> IMPLICIT consistency as dt->0 (1st order)")
# print("=" * 72)
# tm = make_model(6, 6, 3)
# C1, C2 = tm.lmob[1][0].copy(), tm.lmob[1][1].copy()
# S1 = tm.limmob[0][0].copy()
# prev = None
# for dt in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
#     xe = G.get_extent_explicit(C1, C2, S1, p, dt)
#     xi = G.get_extent_implicit(C1, C2, S1, p, dt)
#     rel = np.max(np.abs(xe - xi)) / max(np.max(np.abs(xi)), 1e-30)
#     r = f"{prev / rel:5.1f}x" if prev else "  --"
#     print(f" dt={dt:7.1e} rel diff={rel:.3e}  reduction={r}")
#     prev = rel

# print()
# print("=" * 72)
# print("4. BRENT fallback vs closed form")
# print("=" * 72)
# for dt in [1e-3, 0.5, 10.0]:
#     xb = G.get_extent_brent(C1, C2, S1, p, dt)
#     xi = G.get_extent_implicit(C1, C2, S1, p, dt)
#     print(f" dt={dt:6.2f} max|brent-closed|={np.max(np.abs(xb - xi)):.3e}")

# print()
# print("=" * 72)
# print("5. EDGE CASES")
# print("=" * 72)
# z = lambda v: np.array([v])
# cases = [
#     ("no mineral S1=0", z(1.0), z(1.0), z(0.0)),
#     ("no acid C2=0", z(1.0), z(0.0), z(1.0)),
#     ("supersaturated C1>Ks", z(15.0), z(1.0), z(1.0)),
#     ("exactly at Ks", z(10.0), z(1.0), z(1.0)),
#     ("all zero", z(0.0), z(0.0), z(0.0)),
#     ("huge acid", z(0.0), z(1e6), z(1.0)),
# ]
# for name, a, b, c in cases:
#     x = G.get_extent_implicit(a, b, c, p, 1.0)
#     print(f" {name:24s} xi={x[0]:12.6f}  finite={np.isfinite(x[0])}  >=0={x[0] >= 0}")

# print()
# print("=" * 72)
# print("6. SPEED vs legacy 4x4 Newton loop (64000 cells)")
# print("=" * 72)
# n = 40
# tm = make_model(n, n, n)
# C1, C2 = tm.lmob[1][0].copy(), tm.lmob[1][1].copy()
# S1 = tm.limmob[0][0].copy()
# t = time.perf_counter()
# G.get_extent_implicit(C1, C2, S1, p, 1e-3)
# t1 = time.perf_counter() - t
# print(f" closed form, whole grid : {t1 * 1e3:8.2f} ms")
# print(
#     f" legacy Newton loop      :
# ~4566 ms (measured earlier)  => ~{4566 / (t1 * 1e3):.0f}x faster"
# )
