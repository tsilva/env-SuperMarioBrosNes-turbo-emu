"""Game-owned action-table metadata for Super Mario Bros NES."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any


GAME_ID = "SuperMarioBros-Nes-v0"
RESERVED_ACTION_SET_NAMES = frozenset(
    {"all", "filtered", "discrete", "multi_discrete"}
)

ActionTable = Sequence[Sequence[str]]


@dataclass(frozen=True)
class CustomActionSpec:
    preset: str | None
    table: tuple[tuple[str, ...], ...]
    meanings: tuple[str, ...]
    masks: tuple[tuple[int, ...], ...]
    table_hash: str


def _metadata() -> Mapping[str, Any]:
    path = resources.files(__package__).joinpath("data", GAME_ID, "metadata.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not load packaged metadata.json: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("packaged metadata.json must contain a JSON object")
    return value


def normalize_action_table(
    table: Any,
    *,
    buttons: Sequence[str | None],
    context: str,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    str,
]:
    if isinstance(table, (str, bytes, bytearray)) or not isinstance(table, Sequence):
        raise ValueError(f"{context} must be a non-empty list of actions")
    if not table:
        raise ValueError(f"{context} must contain at least one action")
    button_to_index = {
        label: index for index, label in enumerate(buttons) if label is not None
    }
    normalized: list[tuple[str, ...]] = []
    meanings: list[str] = []
    masks: list[tuple[int, ...]] = []
    seen_masks: set[int] = set()
    for action_index, raw_action in enumerate(table):
        if isinstance(raw_action, (str, bytes, bytearray)) or not isinstance(
            raw_action, Sequence
        ):
            raise ValueError(f"{context} action {action_index} must be a list of button labels")
        labels: list[str] = []
        seen_labels: set[str] = set()
        mask = 0
        for label in raw_action:
            if not isinstance(label, str):
                raise ValueError(f"{context} action {action_index} labels must be strings")
            if label in seen_labels:
                raise ValueError(
                    f"{context} action {action_index} contains duplicate button {label!r}"
                )
            try:
                button_index = button_to_index[label]
            except KeyError as exc:
                valid = ", ".join(repr(name) for name in button_to_index)
                raise ValueError(
                    f"{context} action {action_index} contains unknown button {label!r}; "
                    f"valid labels: {valid}"
                ) from exc
            labels.append(label)
            seen_labels.add(label)
            mask |= 1 << button_index
        if mask in seen_masks:
            raise ValueError(f"{context} action {action_index} duplicates an earlier action")
        normalized.append(tuple(labels))
        meanings.append("noop" if not labels else "_".join(x.lower() for x in labels))
        masks.append((mask,))
        seen_masks.add(mask)
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    return (
        tuple(normalized),
        tuple(meanings),
        tuple(masks),
        hashlib.sha256(payload.encode("ascii")).hexdigest(),
    )


def load_action_tables(
    buttons: Sequence[str | None],
) -> tuple[
    dict[str, tuple[tuple[str, ...], ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    raw = _metadata().get("action_sets")
    if not isinstance(raw, Mapping):
        raise RuntimeError("packaged metadata.json action_sets must be an object")
    tables: dict[str, tuple[tuple[str, ...], ...]] = {}
    action_sets: dict[str, tuple[str, ...]] = {}
    buttons_by_meaning: dict[str, tuple[str, ...]] = {}
    folded_names: set[str] = set()
    for name, table in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("packaged action set names must be non-empty strings")
        folded = name.casefold()
        if folded in RESERVED_ACTION_SET_NAMES or folded in folded_names:
            raise RuntimeError(f"invalid or reserved packaged action set name {name!r}")
        normalized, meanings, _masks, _hash = normalize_action_table(
            table,
            buttons=buttons,
            context=f"action set {name!r}",
        )
        tables[name] = normalized
        action_sets[name] = meanings
        folded_names.add(folded)
        for meaning, labels in zip(meanings, normalized):
            previous = buttons_by_meaning.setdefault(meaning, labels)
            if previous != labels:
                raise RuntimeError(
                    f"action meaning {meaning!r} maps to conflicting button combinations"
                )
    return tables, action_sets, buttons_by_meaning


def resolve_custom_action(
    value: Any,
    *,
    buttons: Sequence[str | None],
    tables: Mapping[str, tuple[tuple[str, ...], ...]],
) -> CustomActionSpec:
    preset = None
    table = value
    if isinstance(value, str):
        matches = {name.casefold(): name for name in tables}
        try:
            preset = matches[value.strip().casefold()]
        except KeyError as exc:
            valid = sorted(RESERVED_ACTION_SET_NAMES | set(matches))
            raise ValueError(
                f"unknown use_restricted_actions value {value!r}; valid values: "
                + ", ".join(valid)
            ) from exc
        table = tables[preset]
    normalized, meanings, masks, table_hash = normalize_action_table(
        table,
        buttons=buttons,
        context=f"action set {preset!r}" if preset else "action table",
    )
    return CustomActionSpec(preset, normalized, meanings, masks, table_hash)


__all__ = [
    "ActionTable",
    "CustomActionSpec",
    "GAME_ID",
    "RESERVED_ACTION_SET_NAMES",
    "load_action_tables",
    "normalize_action_table",
    "resolve_custom_action",
]
