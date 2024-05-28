from dataclasses import dataclass

from pyrtid.utils.dataclass import default_field


@dataclass
class MyDataClass:
    a_nondefault_arg: dict
    a_default_arg: dict = default_field({})


def test_default_field() -> None:
    instance1 = MyDataClass({})
    assert instance1.a_nondefault_arg == {}
    assert instance1.a_default_arg == {}

    instance1.a_default_arg["a"] = "b"

    instance2 = MyDataClass({})
    assert instance2.a_nondefault_arg == {}
    assert instance2.a_default_arg == {}
