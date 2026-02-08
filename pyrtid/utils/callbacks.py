# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET


class Callback:
    """Implement callbacks for gmres or other solvers."""

    def __init__(self) -> None:
        self.comptor: int = 0

    def __call__(self, *args, **kwargs) -> None:
        self.comptor += 1

    def itercount(self) -> int:
        return self.comptor

    def clear(self) -> None:
        self.comptor = 0
