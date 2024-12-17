"""
Implement general interface to create transformations
acting upon data instances.

Jannik Stebani 2023
"""
from woodnet.transformations.buildtools import (get_class, from_configuration,
                                                from_configurations)

__all__ = ['get_class', 'from_configuration', 'from_configurations']