"""Provide some utilities for working with decorators."""


def conditional_decorator_func(dec, condition):
    """Apply a func decorator on condition."""

    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


class ConditionalClassDecorator:
    """Apply a class decorator on condition."""

    def __init__(self, dec, condition):
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)
