# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delivery Optimisation environment server components."""

try:
    from .delivery_optimisation_environment import DeliveryOptimisationEnv
except (ImportError, SystemError):
    from delivery_optimisation_environment import DeliveryOptimisationEnv

__all__ = ["DeliveryOptimisationEnv"]

