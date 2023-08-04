# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:44:53 2020

@author: ancollet
"""

import time


def display_initial_message(task_nb, message) -> None:
    """
    This function prints a message to the console and is to be called at the
    start of each task. It assures a clean and unified display.

    Parameters
    ----------
    task_nb : int
        the number of the task executed (to get a clean console display)
    message: str
        message to be displayed

    Returns
    -------
    No returns
    """

    # print the message
    print("\n{:02.0f} # {}".format(task_nb, message))


def display_intermediate_message(message, level) -> None:
    """
    This function prints a message to the console and is to be called within
    the tasks for intermediates outputs.
    It assures a clean and unified display.

    Parameters
    ----------
    message: str
        message to be displayed
    level : int
        the level of the message. Indicates the indentation to apply and the
        symbol to use ('# ', '* ', '> ', ...)

    Returns
    -------
    No returns
    """

    dico_symbols = {0: "", 1: "- ", 2: "* ", 3: "> ", 4: ". "}
    dico_indent = {0: 0, 1: 5, 2: 7, 3: 9, 4: 11}

    # print the message
    print(
        "{:>{width}}{}".format(dico_symbols[level], message, width=dico_indent[level])
    )


def display_final_computation_time_message(start_time, level=1):
    """
    This function prints a message to the console and is to be called at the
    end of each task. It assures a clean and unified display.

    Parameters
    ----------
    start_time : time()
        time when the task was started
    final: bool
        boolean indicating if it is a script end or a task end

    Returns
    -------
    No returns
    """

    duration = time.time() - start_time
    msg = "Done in : {:.3f}s seconds - ".format(duration)
    # Add
    if level == 0:
        print("\n#### {} ####".format(msg))
    else:
        print("{:{width}}# {}".format(" ", msg, width=1 + level * 2))


if __name__ == "__main__":
    from random import choice as rc

    # Final start time
    start_time = time.time()

    # Simulate several tasks
    for task_nb in range(0, 5, 1):
        # Sub start time
        sub_start_time = time.time()
        # Test the functions
        task_nb += 1
        msg = "title"
        display_initial_message(task_nb, msg)

        # Select a sub task number
        for sub_task in range(0, rc([0, 1, 2, 3, 4, 5])):
            # Add sub title
            # Test the functions
            level = 1
            msg = "subtitle of level " + str(level)
            display_intermediate_message(msg, level)

            # Select a sub task number
            for sub_sub_task in range(0, rc([0, 1, 2, 3, 4, 5])):
                # Add sub title
                # Test the functions
                level = 2
                msg = "subtitle of level " + str(level)
                display_intermediate_message(msg, level)

                # Select a sub task number
                for sub_sub_sub_task in range(0, rc([0, 1, 2, 3, 4, 5])):
                    # Add sub title
                    # Test the functions
                    level = 3
                    msg = "subtitle of level " + str(level)
                    display_intermediate_message(msg, level)

        # Display the final sub computation time
        display_final_computation_time_message(sub_start_time, final=False)
    # Display the total computation time
    display_final_computation_time_message(start_time, final=True)
