# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Regularization with distributions.

This is just some ideas for the future:
- Use a metric to compute the difference between two distributions (one
theoretical) and one given from the field to optimize.py

Several norm can be used: https://stats.stackexchange.com/questions/82076/similarity-measure-between-multiple-distributions

Normally, none is differentiable and so it must be approximated from finite differences.

Although I don"t know if that is possible since we are dealing with continuous and not
discrete variables.
Maybe we need to find something treating the problem in a continuous way ???
"""
