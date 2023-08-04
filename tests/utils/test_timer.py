import pytest

from pyrtid.utils import time_it


@time_it(level=1)
def calc_cube(numbers):
    result = []
    for number in numbers:
        result.append(number * number * number)
    return result


def test_time_it_decorator_return_wrapped_function_results():
    @time_it()
    def calc_square(number):
        return number**2

    assert calc_square(3) == 9


def test_time_it_decorator_works_with_level_keywords():
    @time_it(level=True)
    def calc_square1(number):
        return number**2

    assert calc_square1(3) == 9

    @time_it(level=1)
    def calc_square2(number):
        return number**2

    assert calc_square2(3) == 9

    @time_it(level=2)
    def calc_square4(number):
        return number**2

    assert calc_square4(3) == 9

    @time_it(level=True)
    def calc_square5(number):
        return number**2

    assert calc_square5(3) == 9

    # Fail if level is negative
    @time_it(level=-1)
    def calc_square6(number):
        return number**2

    with pytest.raises(ValueError):
        calc_square6(3)


# def test_time_it_decorator_works_with_level_attribute_keywords():
#     @time_it(level=True)
#     def calc_square(number):
#         return number**2

#     assert calc_square(3) == 9

#     class Foo:
#         def __init__(self):
#             self.test_level = 4

#         @time_it(level_attribute="test_level")
#         def bar(self, word):
#             return word

#     f = Foo()
#     assert f.bar("kitty") == "kitty"
