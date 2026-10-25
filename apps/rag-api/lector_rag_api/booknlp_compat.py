"""Narrow runtime compatibility for BookNLP's legacy BERT checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

LEGACY_POSITION_IDS_KEY = "bert.embeddings.position_ids"


@contextmanager
def booknlp_state_dict_compatibility(
    module_class: type[Any] | None = None,
) -> Iterator[None]:
    """Ignore the obsolete BERT position_ids buffer for all BookNLP modules.

    Entity, coreference, and speaker checkpoints serialize this buffer, while
    current Transformers marks it non-persistent. Patch PyTorch's base Module
    only while BookNLP initializes. The original loader is still called with
    its existing strict setting, so every other mismatch remains an error.
    """
    if module_class is None:
        from torch.nn import Module

        module_class = Module

    original_loader = module_class.load_state_dict
    missing_class_attribute = object()
    original_attribute = module_class.__dict__.get(
        "load_state_dict", missing_class_attribute
    )

    def load_compatible_state_dict(
        instance: Any, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        candidate: Any = state_dict
        if (
            isinstance(state_dict, Mapping)
            and LEGACY_POSITION_IDS_KEY in state_dict
            and LEGACY_POSITION_IDS_KEY not in instance.state_dict()
        ):
            candidate = copy.copy(state_dict)
            del candidate[LEGACY_POSITION_IDS_KEY]
        return original_loader(instance, candidate, *args, **kwargs)

    module_class.load_state_dict = load_compatible_state_dict  # type: ignore[method-assign]
    try:
        yield
    finally:
        if original_attribute is missing_class_attribute:
            delattr(module_class, "load_state_dict")
        else:
            module_class.load_state_dict = original_attribute  # type: ignore[method-assign]
