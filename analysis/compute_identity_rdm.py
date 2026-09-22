"""Actor-identity control RDM.

18x18 RDM: D[i,j] = 0 if stimuli i and j are the same actor, 1 otherwise.
Diagonal = 0. Matches the canonical construction in
analysis/identity_partial_rsa.py (stim order = 6 actors x 3 emotions,
emotion-fastest, so actor = index // 3).

NOTE: within a single emotion the 6 stimuli are 6 *different* actors, so the
per-emotion 6x6 submatrix is all-ones (constant) and identity partialling is
degenerate there — meaningful only at the full 18-stimulus level.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("FER_ROOT",".")+"/analysis/motion_controls/rdms/identity_rdm.npy")

N_ACTORS = 6
N_EMO_PER_ACTOR = 3


def actor_identity_rdm(n_actors: int = N_ACTORS,
                       n_emotions_per_actor: int = N_EMO_PER_ACTOR) -> np.ndarray:
    """Canonical order is EMOTION-major (fear 0-5, happy 6-11, neutral 12-17),
    each block running over the same 6 actors, so actor = index % n_actors.
    (An actor-major `index // 3` would mislabel every stimulus but the first.)
    """
    n = n_actors * n_emotions_per_actor
    actor = np.arange(n) % n_actors                       # [0,1,..,5,0,1,..,5,...]
    rdm = (actor[:, None] != actor[None, :]).astype(np.float32)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def main() -> None:
    rdm = actor_identity_rdm()
    n_pairs = rdm.shape[0] * (rdm.shape[0] - 1) // 2
    iu = np.triu_indices(rdm.shape[0], k=1)
    same = int((rdm[iu] == 0).sum())
    diff = int((rdm[iu] == 1).sum())
    print(f"Identity RDM {rdm.shape}: {n_pairs} pairs "
          f"({same} same-actor, {diff} different-actor)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, rdm)
    print(f"[ok] saved -> {OUT}")


if __name__ == "__main__":
    main()
