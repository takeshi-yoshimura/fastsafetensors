# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Dict, Optional, Type

from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase
from ..st_types import Device
from .base import CopierInterface

CopierType = str
CopierConstructFunc = Callable[
    [SafeTensorsMetadata, Device, FrameworkOpBase], CopierInterface
]
CopierConstructorFactory = Callable[..., CopierConstructFunc]

_copier_registry: Dict[CopierType, CopierConstructorFactory] = {}
_copier_class_registry: Dict[CopierType, Type[CopierInterface]] = {}


def register_copier_constructor(
    copier_type: CopierType, copier_class: Optional[Type[CopierInterface]] = None
):
    """Register a factory for *copier_type*.

    *copier_class* is the ``CopierInterface`` subclass the factory builds.
    Registering it lets callers reach the copier's class-level policy (e.g.
    ``chunk_transient_multiplier``) before any instance exists; see
    ``copier_class_of``.
    """

    def decorator(factory_func: CopierConstructorFactory) -> CopierConstructorFactory:
        @functools.wraps(factory_func)
        def factory(*args: Any, **kwargs: Any) -> CopierConstructFunc:
            construct = factory_func(*args, **kwargs)
            # A factory may delegate to another copier's factory -- gds hands
            # off to nogds/unified when cuFile is unavailable. The delegate
            # runs first and tags the constructor it returns, so the innermost
            # factory wins and the tag names the copier that will actually be
            # built, not the one that was asked for.
            if getattr(construct, "copier_class", None) is None:
                try:
                    construct.copier_class = copier_class  # type: ignore[attr-defined]
                except AttributeError:
                    pass  # exotic callable that rejects attributes; tag is optional
            return construct

        _copier_registry[copier_type] = factory
        if copier_class is not None:
            _copier_class_registry[copier_type] = copier_class
        return factory

    return decorator


def get_copier_class(copier_type: CopierType) -> Type[CopierInterface]:
    """The ``CopierInterface`` subclass registered for *copier_type*.

    This is the statically requested copier. When a constructor is already in
    hand, prefer ``copier_class_of`` -- a factory may have fallen back to a
    different copier than the type name suggests.

    Falls back to ``CopierInterface`` itself when a factory was registered
    without its class, so class-level policy hits the interface defaults
    (which raise ``NotImplementedError``) rather than a wrong guess.
    """
    return _copier_class_registry.get(copier_type, CopierInterface)


def copier_class_of(construct: CopierConstructFunc) -> Type[CopierInterface]:
    """The ``CopierInterface`` subclass *construct* will actually build.

    Reads the tag applied by ``register_copier_constructor``, so it survives a
    factory delegating to another copier's factory. Unregistered constructors
    fall back to ``CopierInterface`` (whose class-level policy refuses).
    """
    return getattr(construct, "copier_class", None) or CopierInterface


def create_copier_constructor(
    copier_type: CopierType, device: Device, **kwargs
) -> CopierConstructFunc:
    if copier_type not in _copier_registry:
        raise KeyError(
            f"Copier type '{copier_type}' is not registered. "
            f"Available types: {list(_copier_registry.keys())}"
        )

    factory_func = _copier_registry[copier_type]
    return factory_func(device, **kwargs)
