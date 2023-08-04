"""
Provide a StrEnum class.

Note: it has been added in python 3.10 https://bugs.python.org/issue41816

@author: acollet
"""
from enum import Enum
from typing import List


class StrEnum(str, Enum):
    """Hashable string Enum.

    .. tip::
        Can be used as :class:`pandas.DataFrame` column names.

    """

    def __str__(self) -> str:
        """Return instance value."""
        return self.value

    def __hash__(self) -> int:
        """Return the hash of the value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Return if two instances are equal."""
        if not isinstance(other, StrEnum) and not isinstance(other, str):
            return False
        return self.value == other

    @classmethod
    def to_list(cls) -> List[str]:
        """Return all enums as a list."""
        return list(cls)
