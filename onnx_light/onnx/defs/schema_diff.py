"""Structured diff between two :class:`~onnx_light.onnx.defs.OpSchema` versions.

This module provides :func:`compare_schemas` and the accompanying dataclasses
(:class:`SchemaDiff`, :class:`ParameterDiff`, :class:`AttributeDiff`,
:class:`ConstraintDiff`) for detecting differences — including **breaking
changes** — between two versions of the same ONNX operator schema.

Typical usage::

    from onnx_light.onnx import defs
    from onnx_light.onnx.defs.schema_diff import compare_schemas

    defs.register_onnx_operator_set_schema()
    old = defs.get_schema("Relu", 6)
    new = defs.get_schema("Relu", 14)
    diff = compare_schemas(old, new)
    print(diff)

A change is considered **breaking** when upgrading existing ONNX models from
the old schema version to the new one could alter observable behaviour without
any other modification.  Examples of breaking changes:

* Removing a previously required input or output.
* Adding a new required input (existing models do not supply it).
* Changing an attribute's type or making an optional attribute required.
* Narrowing a type constraint (removing previously allowed types).

Non-breaking examples:

* Adding a new optional input or output.
* Widening a type constraint (adding new allowed types).
* Adding an optional attribute with a sensible default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_OpSchema = _C.defs.OpSchema


def _attr_default_value_repr(attr: Any) -> str:
    """Returns a canonical string representation of an ``Attribute`` default value.

    :param attr: An ``OpSchema.Attribute`` object.
    :returns: A string encoding the attribute type and value suitable for
        equality comparison between two default values.
    :rtype: str
    """
    dv = attr._default_value
    at = dv.type
    # Import here to avoid a circular dependency at module load time.
    from ..onnx_proto._onnxpy import AttributeProto  # type: ignore[missing-module-attribute]

    if at == AttributeProto.UNDEFINED:
        return "UNDEFINED"
    if at == AttributeProto.FLOAT:
        return f"FLOAT:{dv.f}"
    if at == AttributeProto.INT:
        return f"INT:{dv.i}"
    if at == AttributeProto.STRING:
        return f"STRING:{dv.s!r}"
    if at == AttributeProto.FLOATS:
        return f"FLOATS:{list(dv.floats)}"
    if at == AttributeProto.INTS:
        return f"INTS:{list(dv.ints)}"
    if at == AttributeProto.STRINGS:
        return f"STRINGS:{list(dv.strings)}"
    # Tensor / Graph / etc. – fall back to the proto repr.
    return repr(dv)


@dataclass
class ParameterDiff:
    """Records a difference in a single input or output formal parameter.

    :param name: Name of the input or output parameter.
    :param kind: Type of difference: ``'added'``, ``'removed'``, or ``'changed'``.
    :param details: Human-readable description of what changed.
    :param is_breaking: ``True`` if the difference is a breaking change.
    """

    name: str
    kind: str
    details: list[str] = field(default_factory=list)
    is_breaking: bool = False

    def __str__(self) -> str:
        """Returns a one-line human-readable description of this parameter diff.

        :returns: A formatted string with an optional ``[BREAKING]`` prefix.
        :rtype: str
        """
        tag = "[BREAKING] " if self.is_breaking else ""
        return f"{tag}{self.kind} '{self.name}': {'; '.join(self.details)}"

    @classmethod
    def compare(
        cls, old_params: list[Any], new_params: list[Any], kind_label: str
    ) -> list[ParameterDiff]:
        """Compares two lists of formal parameters (inputs or outputs).

        :param old_params: Formal parameters from the old schema.
        :param new_params: Formal parameters from the new schema.
        :param kind_label: Label used for context (``'input'`` or ``'output'``).
        :returns: A list of :class:`ParameterDiff` items describing additions,
            removals, and changes.
        :rtype: list[ParameterDiff]
        """
        diffs: list[ParameterDiff] = []
        old_by_name = {p.name: (i, p) for i, p in enumerate(old_params)}
        new_by_name = {p.name: (i, p) for i, p in enumerate(new_params)}

        # Removed parameters.
        for name, (idx, _p_old) in old_by_name.items():
            if name not in new_by_name:
                diffs.append(
                    cls(
                        name=name,
                        kind="removed",
                        details=[f"was at position {idx}"],
                        is_breaking=True,
                    )
                )

        # Added parameters.
        for name, (idx, p_new) in new_by_name.items():
            if name not in old_by_name:
                is_optional = p_new.option == _OpSchema.FormalParameterOption.Optional
                diffs.append(
                    cls(
                        name=name,
                        kind="added",
                        details=[
                            f"at position {idx}",
                            f"option={p_new.option}",
                            f"type_str={p_new.type_str!r}",
                        ],
                        # Adding a non-optional parameter is breaking for outputs too because
                        # existing models do not produce the extra output value; for inputs a
                        # new required input breaks existing models that don't supply it.
                        is_breaking=not is_optional,
                    )
                )

        # Changed parameters (present in both).
        for name in old_by_name:
            if name not in new_by_name:
                continue
            idx_old, p_old = old_by_name[name]
            idx_new, p_new = new_by_name[name]
            changes: list[str] = []
            breaking = False

            if idx_old != idx_new:
                changes.append(f"position changed {idx_old} -> {idx_new}")
                breaking = True

            if p_old.option != p_new.option:
                changes.append(f"option changed {p_old.option} -> {p_new.option}")
                # Becoming required (losing Optional) is breaking.
                if p_new.option != _OpSchema.FormalParameterOption.Optional:
                    breaking = True

            if p_old.type_str != p_new.type_str:
                changes.append(f"type_str changed {p_old.type_str!r} -> {p_new.type_str!r}")
                # Type-variable renaming is not itself breaking, but a change from a
                # concrete type to another is.  We mark it breaking conservatively.
                breaking = True

            if changes:
                diffs.append(
                    cls(name=name, kind="changed", details=changes, is_breaking=breaking)
                )

        return diffs


@dataclass
class AttributeDiff:
    """Records a difference in a single operator attribute.

    :param name: Name of the attribute.
    :param kind: Type of difference: ``'added'``, ``'removed'``, or ``'changed'``.
    :param details: Human-readable description of what changed.
    :param is_breaking: ``True`` if the difference is a breaking change.
    """

    name: str
    kind: str
    details: list[str] = field(default_factory=list)
    is_breaking: bool = False

    def __str__(self) -> str:
        """Returns a one-line human-readable description of this attribute diff.

        :returns: A formatted string with an optional ``[BREAKING]`` prefix.
        :rtype: str
        """
        tag = "[BREAKING] " if self.is_breaking else ""
        return f"{tag}{self.kind} '{self.name}': {'; '.join(self.details)}"


@dataclass
class ConstraintDiff:
    """Records a difference in a type constraint.

    :param name: Type parameter string (e.g. ``'T'``, ``'T1'``).
    :param kind: Type of difference: ``'added'``, ``'removed'``, or ``'changed'``.
    :param added_types: Types present in the new schema but not in the old one.
    :param removed_types: Types present in the old schema but not in the new one.
    :param details: Human-readable description of what changed.
    :param is_breaking: ``True`` if the difference is a breaking change.
    """

    name: str
    kind: str
    added_types: list[str] = field(default_factory=list)
    removed_types: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    is_breaking: bool = False

    def __str__(self) -> str:
        """Returns a one-line human-readable description of this constraint diff.

        :returns: A formatted string with an optional ``[BREAKING]`` prefix.
        :rtype: str
        """
        tag = "[BREAKING] " if self.is_breaking else ""
        return f"{tag}{self.kind} '{self.name}': {'; '.join(self.details)}"


@dataclass
class SchemaDiff:
    """Summarizes the differences between two versions of an operator schema.

    Use :func:`compare_schemas` to create instances of this class.

    :param op_name: Operator name.
    :param domain: Operator domain.
    :param old_version: ``since_version`` of the old (reference) schema.
    :param new_version: ``since_version`` of the new schema.
    :param inputs: Differences in formal input parameters.
    :param outputs: Differences in formal output parameters.
    :param attributes: Differences in attributes.
    :param constraints: Differences in type constraints.
    :param is_breaking: ``True`` if any individual change is breaking.
    :param breaking_reasons: Human-readable list of reasons why the change is breaking.
    """

    op_name: str
    domain: str
    old_version: int
    new_version: int
    inputs: list[ParameterDiff] = field(default_factory=list)
    outputs: list[ParameterDiff] = field(default_factory=list)
    attributes: list[AttributeDiff] = field(default_factory=list)
    constraints: list[ConstraintDiff] = field(default_factory=list)
    is_breaking: bool = False
    breaking_reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Returns a human-readable summary of the schema diff.

        :returns: A multi-line string describing all differences.
        :rtype: str
        """
        lines: list[str] = [
            f"SchemaDiff: {self.op_name} (domain={self.domain!r})",
            f"  old version : {self.old_version}",
            f"  new version : {self.new_version}",
            f"  breaking    : {self.is_breaking}",
        ]
        if self.breaking_reasons:
            lines.append("  Breaking reasons:")
            for r in self.breaking_reasons:
                lines.append(f"    - {r}")
        if self.inputs:
            lines.append("  Inputs:")
            for d in self.inputs:
                lines.append(f"    {d}")
        if self.outputs:
            lines.append("  Outputs:")
            for d in self.outputs:
                lines.append(f"    {d}")
        if self.attributes:
            lines.append("  Attributes:")
            for d in self.attributes:
                lines.append(f"    {d}")
        if self.constraints:
            lines.append("  Type constraints:")
            for d in self.constraints:
                lines.append(f"    {d}")
        return "\n".join(lines)


def _compare_attributes(
    old_attrs: dict[str, Any], new_attrs: dict[str, Any]
) -> list[AttributeDiff]:
    """Compares two attribute dictionaries from :class:`OpSchema` objects.

    :param old_attrs: Attribute mapping from the old schema.
    :param new_attrs: Attribute mapping from the new schema.
    :returns: A list of :class:`AttributeDiff` items describing additions,
        removals, and changes.
    :rtype: list[AttributeDiff]
    """
    diffs: list[AttributeDiff] = []

    # Removed attributes.
    for name, a_old in old_attrs.items():
        if name not in new_attrs:
            diffs.append(
                AttributeDiff(
                    name=name,
                    kind="removed",
                    details=[f"type={a_old.type}", f"required={a_old.required}"],
                    is_breaking=True,
                )
            )

    # Added attributes.
    for name, a_new in new_attrs.items():
        if name not in old_attrs:
            # Adding a required attribute without a default is breaking because
            # existing models do not specify it.
            is_breaking = a_new.required
            details = [f"type={a_new.type}", f"required={a_new.required}"]
            if not a_new.required:
                details.append(f"default={_attr_default_value_repr(a_new)}")
            diffs.append(
                AttributeDiff(name=name, kind="added", details=details, is_breaking=is_breaking)
            )

    # Changed attributes (present in both).
    for name in old_attrs:
        if name not in new_attrs:
            continue
        a_old = old_attrs[name]
        a_new = new_attrs[name]
        changes: list[str] = []
        breaking = False

        if a_old.type != a_new.type:
            changes.append(f"type changed {a_old.type} -> {a_new.type}")
            breaking = True

        if a_old.required != a_new.required:
            changes.append(f"required changed {a_old.required} -> {a_new.required}")
            # Becoming required is breaking; becoming optional is not.
            if a_new.required:
                breaking = True

        old_dv_repr = _attr_default_value_repr(a_old)
        new_dv_repr = _attr_default_value_repr(a_new)
        if old_dv_repr != new_dv_repr:
            changes.append(f"default value changed {old_dv_repr} -> {new_dv_repr}")
            # A default value change alters the behaviour of models that rely on
            # the implicit default, so it is considered breaking.
            breaking = True

        if changes:
            diffs.append(
                AttributeDiff(name=name, kind="changed", details=changes, is_breaking=breaking)
            )

    return diffs


def _compare_constraints(
    old_constraints: list[Any], new_constraints: list[Any]
) -> list[ConstraintDiff]:
    """Compares two lists of type constraint parameters.

    :param old_constraints: Type constraints from the old schema.
    :param new_constraints: Type constraints from the new schema.
    :returns: A list of :class:`ConstraintDiff` items describing additions,
        removals, and changes.
    :rtype: list[ConstraintDiff]
    """
    diffs: list[ConstraintDiff] = []
    old_by_name = {tc.type_param_str: tc for tc in old_constraints}
    new_by_name = {tc.type_param_str: tc for tc in new_constraints}

    for name, _tc_old in old_by_name.items():
        if name not in new_by_name:
            diffs.append(
                ConstraintDiff(
                    name=name,
                    kind="removed",
                    details=["entire constraint removed"],
                    is_breaking=True,
                )
            )

    for name, tc_new in new_by_name.items():
        if name not in old_by_name:
            diffs.append(
                ConstraintDiff(
                    name=name,
                    kind="added",
                    added_types=list(tc_new.allowed_type_strs),
                    details=[f"added types: {sorted(tc_new.allowed_type_strs)}"],
                    is_breaking=False,
                )
            )

    for name in old_by_name:
        if name not in new_by_name:
            continue
        tc_old = old_by_name[name]
        tc_new = new_by_name[name]
        old_types = set(tc_old.allowed_type_strs)
        new_types = set(tc_new.allowed_type_strs)

        added = sorted(new_types - old_types)
        removed = sorted(old_types - new_types)

        if not added and not removed:
            continue

        details: list[str] = []
        if added:
            details.append(f"added types: {added}")
        if removed:
            details.append(f"removed types: {removed}")

        diffs.append(
            ConstraintDiff(
                name=name,
                kind="changed",
                added_types=added,
                removed_types=removed,
                details=details,
                # Removing previously supported types breaks existing models.
                is_breaking=bool(removed),
            )
        )

    return diffs


def compare_schemas(schema_old: Any, schema_new: Any) -> SchemaDiff:
    """Compares two :class:`OpSchema` objects for the same operator.

    The function inspects inputs, outputs, attributes, and type constraints and
    highlights what was added, removed, or changed.  It also determines whether
    the transition from *schema_old* to *schema_new* constitutes a **breaking
    change**, defined as a change that would alter the observable behaviour of
    an existing model when its operator is upgraded to the new schema version
    without any other modification.

    :param schema_old: The reference (typically older) schema.
    :param schema_new: The schema compared against (typically newer).
    :returns: A :class:`SchemaDiff` instance summarising all detected differences.
    :rtype: SchemaDiff

    Example::

        from onnx_light.onnx import defs
        from onnx_light.onnx.defs.schema_diff import compare_schemas
        defs.register_onnx_operator_set_schema()
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        print(diff)
    """
    input_diffs = ParameterDiff.compare(list(schema_old.inputs), list(schema_new.inputs), "input")
    output_diffs = ParameterDiff.compare(
        list(schema_old.outputs), list(schema_new.outputs), "output"
    )
    attr_diffs = _compare_attributes(schema_old.attributes, schema_new.attributes)
    constraint_diffs = _compare_constraints(
        list(schema_old.type_constraints), list(schema_new.type_constraints)
    )

    # Also compare min/max input/output arity.
    extra_input_diffs: list[ParameterDiff] = []
    extra_output_diffs: list[ParameterDiff] = []
    if (
        schema_old.min_input != schema_new.min_input
        or schema_old.max_input != schema_new.max_input
    ):
        old_max_str = (
            "∞" if _OpSchema.is_infinite(schema_old.max_input) else str(schema_old.max_input)
        )
        new_max_str = (
            "∞" if _OpSchema.is_infinite(schema_new.max_input) else str(schema_new.max_input)
        )
        breaking = (schema_new.min_input > schema_old.min_input) or (
            not _OpSchema.is_infinite(schema_new.max_input)
            and (
                _OpSchema.is_infinite(schema_old.max_input)
                or schema_new.max_input < schema_old.max_input
            )
        )
        extra_input_diffs.append(
            ParameterDiff(
                name="<arity>",
                kind="changed",
                details=[
                    f"min_input {schema_old.min_input} -> {schema_new.min_input}",
                    f"max_input {old_max_str} -> {new_max_str}",
                ],
                is_breaking=breaking,
            )
        )

    if (
        schema_old.min_output != schema_new.min_output
        or schema_old.max_output != schema_new.max_output
    ):
        old_max_str = (
            "∞" if _OpSchema.is_infinite(schema_old.max_output) else str(schema_old.max_output)
        )
        new_max_str = (
            "∞" if _OpSchema.is_infinite(schema_new.max_output) else str(schema_new.max_output)
        )
        breaking = (schema_new.min_output > schema_old.min_output) or (
            not _OpSchema.is_infinite(schema_new.max_output)
            and (
                _OpSchema.is_infinite(schema_old.max_output)
                or schema_new.max_output < schema_old.max_output
            )
        )
        extra_output_diffs.append(
            ParameterDiff(
                name="<arity>",
                kind="changed",
                details=[
                    f"min_output {schema_old.min_output} -> {schema_new.min_output}",
                    f"max_output {old_max_str} -> {new_max_str}",
                ],
                is_breaking=breaking,
            )
        )

    all_input_diffs = input_diffs + extra_input_diffs
    all_output_diffs = output_diffs + extra_output_diffs

    # Collect all individual breaking changes.
    breaking_reasons: list[str] = []
    for d in all_input_diffs:
        if d.is_breaking:
            breaking_reasons.append(f"input '{d.name}' ({d.kind}): {'; '.join(d.details)}")
    for d in all_output_diffs:
        if d.is_breaking:
            breaking_reasons.append(f"output '{d.name}' ({d.kind}): {'; '.join(d.details)}")
    for d in attr_diffs:
        if d.is_breaking:
            breaking_reasons.append(f"attribute '{d.name}' ({d.kind}): {'; '.join(d.details)}")
    for d in constraint_diffs:
        if d.is_breaking:
            breaking_reasons.append(
                f"type constraint '{d.name}' ({d.kind}): {'; '.join(d.details)}"
            )

    return SchemaDiff(
        op_name=schema_old.name,
        domain=schema_old.domain,
        old_version=schema_old.since_version,
        new_version=schema_new.since_version,
        inputs=all_input_diffs,
        outputs=all_output_diffs,
        attributes=attr_diffs,
        constraints=constraint_diffs,
        is_breaking=bool(breaking_reasons),
        breaking_reasons=breaking_reasons,
    )
