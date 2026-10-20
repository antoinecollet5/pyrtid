# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide callbacks to monitor the iterations of iterative linear solvers."""


class Callback:
    """Implement callbacks for gmres or other solvers."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.comptor: int = 0

    def __call__(self, *args, **kwargs) -> None:
        """
        Increment the iteration counter (called at each solver iteration).

        Parameters
        ----------
        *args : Any
            Arguments passed by the solver (ignored).
        **kwargs : Any
            Keyword arguments passed by the solver (ignored).
        """
        self.comptor += 1

    def itercount(self) -> int:
        """
        Return the number of iterations counted so far.

        Returns
        -------
        int
            The number of calls since the last :meth:`clear`.
        """
        return self.comptor

    def clear(self) -> None:
        """Reset the iteration counter to zero."""
        self.comptor = 0
