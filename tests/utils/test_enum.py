"""
Test the StrEnum class

@author: acollet
"""

from pyrtid.utils.enum import StrEnum


class Colors(StrEnum):
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"


def test_eq() -> None:
    """Basic test."""
    assert Colors.BLUE == "blue"
    assert Colors.YELLOW == "yellow"
    assert Colors.RED == "red"
    assert not Colors.BLUE == 2


def test_str() -> None:
    str(Colors.BLUE)


def test_hashability() -> None:
    my_dict = {Colors.BLUE: 2, Colors.YELLOW: 5}
    assert my_dict[Colors.BLUE] == 2


def test_to_list() -> None:
    assert Colors.to_list() == ["blue", "yellow", "red"]
