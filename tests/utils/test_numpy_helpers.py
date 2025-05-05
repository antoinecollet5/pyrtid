import numpy as np
from pyrtid.utils import NDArrayFloat, np_cache


def test_np_cache() -> None:
    arr = np.ones((3, 3))

    class Counter:
        def __init__(self) -> None:
            self.counter: int = 0
            self.counterc: int = 0

    def call(_arr, self) -> NDArrayFloat:
        self.counter += 1
        return _arr

    @np_cache()  # array in fist position by default
    def call_c(_arr, self) -> NDArrayFloat:
        self.counterc += 1
        return np.asarray(_arr)

    @np_cache(pos=1)  # array in second position
    def call_d(self, _arr) -> NDArrayFloat:
        self.counterc += 1
        return np.asarray(_arr)

    counter = Counter()
    assert counter.counter == 0
    call(arr, counter)
    assert counter.counter == 1
    call(arr, counter)
    assert counter.counter == 2

    assert counter.counterc == 0
    call_c(arr, counter)
    assert counter.counterc == 1
    call_c(arr, counter)
    assert counter.counterc == 1

    assert counter.counterc == 1
    call_d(counter, arr)
    assert counter.counterc == 2
    call_d(counter, arr)
    assert counter.counterc == 2
