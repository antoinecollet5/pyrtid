# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:43:14 2020

@author: ancollet
"""

import time
from functools import wraps

from pyrtid.utils.logging import display_final_computation_time_message


def time_it(level=1):
    """Time the decorated function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initiate a timer to measure the function performance
            start_time = time.time()
            # Call the function that was passed as an argument
            result = func(*args, **kwargs)
            # Final message
            display_final_computation_time_message(start_time, level)
            return result

        return wrapper

    return decorator
