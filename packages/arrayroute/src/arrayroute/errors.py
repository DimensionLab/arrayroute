# SPDX-License-Identifier: Apache-2.0
"""Explicit capability failures that a read-policy comparison may skip."""


class UnsupportedReadPolicyError(ValueError):
    """The selected engine/layout cannot execute this policy."""
