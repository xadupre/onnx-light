"""Structured diff between two :class:`~onnx_light.onnx.defs.OpSchema` versions.

This module provides :func:`compare_schemas` and the accompanying dataclasses
(:class:`SchemaDiff`, :class:`ParameterDiff`, :class:`AttributeDiff`,
:class:`ConstraintDiff`, :class:`DocDiff`) for detecting differences —
including **breaking changes** — between two versions of the same ONNX
operator schema.

Documentation strings are diffed at the **line** level (via
:class:`DocDiff`), using :mod:`difflib` (line-based edit distance and
unified diff) rather than character-level comparison, so the result reads
naturally both as plain text and inside Sphinx ``code-block:: diff``
directives.

Typical usage with a full :class:`~onnx_light.onnx.defs.OpSchema`:

.. runpython::
    :showcode:

    from onnx_light.onnx import defs
    from onnx_light.compatibility.schema_diff import compare_schemas

    defs.register_onnx_operator_set_schema()
    old = defs.get_schema("Relu", 6)
    new = defs.get_schema("Relu", 14)
    diff = compare_schemas(old, new)
    print(diff)

The same function also accepts the lightweight ``LightOpSchema`` objects
exposed by ``onnx_light`` (those have no attributes nor input/output
arity, so those sections of the diff are simply omitted):

.. runpython::
    :showcode:

    from collections import defaultdict
    from onnx_light.onnx_proto._onnxpy import onnx_op
    from onnx_light.compatibility.schema_diff import compare_schemas

    by_name = defaultdict(list)
    for s in onnx_op.GetAllOnnxOpSchemasWithHistory(True):
        by_name[(s.domain, s.name)].append(s)
    versions = sorted(by_name[("ai.onnx", "Add")], key=lambda s: s.since_version)
    old, new = versions[0], versions[-1]
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

import difflib
from dataclasses import dataclass, field
from typing import Any

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_OpSchema = _C.defs.OpSchema  # type: ignore


def _param_type_str(param: Any) -> str:
    """Returns the type string of a formal parameter.

    Supports both the full :class:`OpSchema.FormalParameter` (attribute
    ``type_str``) and the lightweight
    :class:`~onnx_light.onnx_proto._onnxpy.onnx_op.FormalParameter` (attribute
    ``type``) exposed by ``LightOpSchema``.
    """
    if hasattr(param, "type_str"):
        return param.type_str
    return getattr(param, "type", "")


def _param_option(param: Any) -> Any:
    """Returns the :class:`OpSchema.FormalParameterOption` of a formal parameter.

    For lightweight schemas which do not expose an ``option`` attribute, this
    returns ``OpSchema.FormalParameterOption.Single`` so that comparison logic
    treats them as ordinary required parameters.
    """
    return getattr(param, "option", _OpSchema.FormalParameterOption.Single)


def _type_to_str(t: Any) -> str:
    """Normalises a type entry of ``TypeConstraintParam.allowed_type_strs``.

    The full :class:`OpSchema` stores those entries as strings while the
    lightweight ``LightOpSchema`` exposes them as ``onnx_op.TensorType`` enum
    values.  This helper returns a string in both cases so the diff is sortable
    and comparable.
    """
    if isinstance(t, str):
        return t
    # LightOpSchema: TensorType enum value -> "tensor(...)" or "seq(...)" string.
    try:
        from ..onnx_proto._onnxpy import onnx_op as _onnx_op  # type: ignore

        return _onnx_op.ToTypeString(t)
    except Exception:
        return str(t)


def _attr_type_name(t: Any) -> str:
    """Returns the bare canonical name of an attribute type.

    Accepts ``onnx::AttributeProto::AttributeType`` enum values (full schema),
    ``onnx_op.AttributeType`` enum values (lightweight schema), integers, and
    strings, returning the bare ONNX enum name (e.g. ``"INTS"``).
    """
    if hasattr(t, "name") and not isinstance(t, str):
        return t.name
    return str(t)


def _attr_default_value_repr(attr: Any) -> str:
    """Returns a canonical string representation of an ``Attribute`` default value.

    Supports both full ``OpSchema.Attribute`` objects (which expose a proto
    ``_default_value``) and the lightweight ``AttributeParam`` adapter used by
    ``onnx_op.LightOpSchema`` (which exposes a pre-formatted
    ``default_value_repr`` string).

    :param attr: An ``OpSchema.Attribute`` or compatible adapter.
    :returns: A string encoding the attribute type and value suitable for
        equality comparison between two default values.
    :rtype: str
    """
    if hasattr(attr, "default_value_repr"):
        return attr.default_value_repr or "UNDEFINED"
    if not hasattr(attr, "_default_value"):
        return "UNDEFINED"
    dv = attr._default_value
    at = dv.type
    # Import here to avoid a circular dependency at module load time.
    from ..onnx_proto._onnxpy import AttributeProto  # type: ignore[attr-defined]

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
                is_optional = _param_option(p_new) == _OpSchema.FormalParameterOption.Optional
                diffs.append(
                    cls(
                        name=name,
                        kind="added",
                        details=[
                            f"at position {idx}",
                            f"option={_param_option(p_new)}",
                            f"type_str={_param_type_str(p_new)!r}",
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

            opt_old = _param_option(p_old)
            opt_new = _param_option(p_new)
            if opt_old != opt_new:
                changes.append(f"option changed {opt_old} -> {opt_new}")
                # Becoming required (losing Optional) is breaking.
                if opt_new != _OpSchema.FormalParameterOption.Optional:
                    breaking = True

            ts_old = _param_type_str(p_old)
            ts_new = _param_type_str(p_new)
            if ts_old != ts_new:
                changes.append(f"type_str changed {ts_old!r} -> {ts_new!r}")
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

    @classmethod
    def compare(cls, old_attrs: dict[str, Any], new_attrs: dict[str, Any]) -> list[AttributeDiff]:
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
                    cls(
                        name=name,
                        kind="removed",
                        details=[
                            f"type={_attr_type_name(a_old.type)}",
                            f"required={a_old.required}",
                        ],
                        is_breaking=True,
                    )
                )

        # Added attributes.
        for name, a_new in new_attrs.items():
            if name not in old_attrs:
                # Adding a required attribute without a default is breaking because
                # existing models do not specify it.
                is_breaking = a_new.required
                details = [f"type={_attr_type_name(a_new.type)}", f"required={a_new.required}"]
                if not a_new.required:
                    details.append(f"default={_attr_default_value_repr(a_new)}")
                diffs.append(
                    cls(name=name, kind="added", details=details, is_breaking=is_breaking)
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
                changes.append(
                    f"type changed {_attr_type_name(a_old.type)} -> "
                    f"{_attr_type_name(a_new.type)}"
                )
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
                    cls(name=name, kind="changed", details=changes, is_breaking=breaking)
                )

        return diffs


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

    @classmethod
    def compare(
        cls, old_constraints: list[Any], new_constraints: list[Any]
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
                    cls(
                        name=name,
                        kind="removed",
                        details=["entire constraint removed"],
                        is_breaking=True,
                    )
                )

        for name, tc_new in new_by_name.items():
            if name not in old_by_name:
                new_type_strs = [_type_to_str(t) for t in tc_new.allowed_type_strs]
                diffs.append(
                    cls(
                        name=name,
                        kind="added",
                        added_types=new_type_strs,
                        details=[f"added types: {sorted(new_type_strs)}"],
                        is_breaking=False,
                    )
                )

        for name in old_by_name:
            if name not in new_by_name:
                continue
            tc_old = old_by_name[name]
            tc_new = new_by_name[name]
            old_types = {_type_to_str(t) for t in tc_old.allowed_type_strs}
            new_types = {_type_to_str(t) for t in tc_new.allowed_type_strs}

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
                cls(
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


@dataclass
class DocDiff:
    """Records a difference between two operator documentation strings.

    Differences are computed at the **line** level using
    :func:`difflib.SequenceMatcher` and :func:`difflib.unified_diff`, rather
    than at the character level.  This matches how humans typically edit
    docstrings (line by line) and produces a much more readable diff.

    A documentation change is never considered breaking on its own, but it is
    still useful information when reviewing a new operator schema version.

    :param old_doc: Documentation string of the old schema (may be empty).
    :param new_doc: Documentation string of the new schema (may be empty).
    :param similarity: Line-level similarity ratio in ``[0.0, 1.0]`` as
        returned by :meth:`difflib.SequenceMatcher.ratio` applied to the
        list of lines.  A value of ``1.0`` means the two docs are identical.
    :param unified_diff: A list of lines produced by
        :func:`difflib.unified_diff` describing the line-level edits to
        transform ``old_doc`` into ``new_doc``.  Empty when the two docs
        are identical.
    :param added_lines: Number of inserted lines (``+`` lines in the unified
        diff, excluding the file header).
    :param removed_lines: Number of removed lines (``-`` lines in the unified
        diff, excluding the file header).
    """

    old_doc: str = ""
    new_doc: str = ""
    similarity: float = 1.0
    unified_diff: list[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0

    @property
    def changed(self) -> bool:
        """Returns ``True`` if the two documentation strings differ."""
        return self.old_doc != self.new_doc

    def __str__(self) -> str:
        """Returns a short summary of the documentation diff.

        :returns: A summary line followed by the unified-diff block (if any).
        :rtype: str
        """
        if not self.changed:
            return "doc unchanged"
        header = (
            f"doc changed (line similarity={self.similarity:.2f}, "
            f"+{self.added_lines}/-{self.removed_lines} lines)"
        )
        if not self.unified_diff:
            return header
        return header + "\n" + "\n".join(self.unified_diff)

    @classmethod
    def compare(
        cls,
        old_doc: str | None,
        new_doc: str | None,
        old_label: str = "old",
        new_label: str = "new",
        context_lines: int = 3,
    ) -> DocDiff:
        """Compares two documentation strings at the line level.

        The two strings are split into lines (preserving relative blank
        lines).  Similarity is computed with
        :meth:`difflib.SequenceMatcher.ratio` on the resulting lists, which
        is equivalent to a normalised line-level edit distance.  A unified
        diff is produced with :func:`difflib.unified_diff` so the result
        renders nicely both as plain text and inside RST ``code-block::
        diff`` directives.

        :param old_doc: Old documentation string (``None`` is treated as
            an empty string).
        :param new_doc: New documentation string (``None`` is treated as
            an empty string).
        :param old_label: Label used for the old side of the unified diff
            header.
        :param new_label: Label used for the new side of the unified diff
            header.
        :param context_lines: Number of context lines around each hunk in
            the unified diff.
        :returns: A :class:`DocDiff` summarising the differences.
        :rtype: DocDiff
        """
        old_text = old_doc or ""
        new_text = new_doc or ""

        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        similarity = difflib.SequenceMatcher(a=old_lines, b=new_lines).ratio()

        if old_text == new_text:
            return cls(
                old_doc=old_text,
                new_doc=new_text,
                similarity=1.0,
                unified_diff=[],
                added_lines=0,
                removed_lines=0,
            )

        ud = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=old_label,
                tofile=new_label,
                n=context_lines,
                lineterm="",
            )
        )
        added = sum(1 for line in ud if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in ud if line.startswith("-") and not line.startswith("---"))
        return cls(
            old_doc=old_text,
            new_doc=new_text,
            similarity=similarity,
            unified_diff=ud,
            added_lines=added,
            removed_lines=removed,
        )


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
    :param doc: Line-level diff of the operator documentation strings.
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
    doc: DocDiff = field(default_factory=DocDiff)
    is_breaking: bool = False
    breaking_reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Returns a plain text human-readable summary of the schema diff.

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
            for di in self.inputs:
                lines.append(f"    {di}")
        if self.outputs:
            lines.append("  Outputs:")
            for do in self.outputs:
                lines.append(f"    {do}")
        if self.attributes:
            lines.append("  Attributes:")
            for da in self.attributes:
                lines.append(f"    {da}")
        if self.constraints:
            lines.append("  Type constraints:")
            for dc in self.constraints:
                lines.append(f"    {dc}")
        if self.doc.changed:
            lines.append("  Documentation:")
            lines.append(
                f"    line similarity: {self.doc.similarity:.2f} "
                f"(+{self.doc.added_lines}/-{self.doc.removed_lines} lines)"
            )
            for ud_line in self.doc.unified_diff:
                lines.append(f"    {ud_line}")
        return "\n".join(lines)

    def to_rst(self) -> str:
        """Returns an RST-formatted summary of the schema diff.

        Produces reStructuredText markup suitable for inclusion in
        Sphinx documentation (e.g. via a ``.. runpython::`` directive
        with the ``:rst:`` flag, provided by the ``sphinx-runpython``
        extension).

        :returns: An RST string describing all differences.
        :rtype: str
        """
        breaking_label = "**yes**" if self.is_breaking else "no"
        lines: list[str] = [
            f"**SchemaDiff**: ``{self.op_name}`` (domain ``{self.domain!r}``)",
            "",
            f"* old version: {self.old_version}",
            f"* new version: {self.new_version}",
            f"* breaking: {breaking_label}",
        ]
        if self.breaking_reasons:
            lines.append("")
            lines.append("**Breaking reasons:**")
            lines.append("")
            for r in self.breaking_reasons:
                lines.append(f"* {r}")
        if self.inputs:
            lines.append("")
            lines.append("**Inputs:**")
            lines.append("")
            for di in self.inputs:
                lines.append(f"* {di}")
        if self.outputs:
            lines.append("")
            lines.append("**Outputs:**")
            lines.append("")
            for d in self.outputs:
                lines.append(f"* {d}")
        if self.attributes:
            lines.append("")
            lines.append("**Attributes:**")
            lines.append("")
            for da in self.attributes:
                lines.append(f"* {da}")
        if self.constraints:
            lines.append("")
            lines.append("**Type constraints:**")
            lines.append("")
            for dc in self.constraints:
                lines.append(f"* {dc}")
        if self.doc.changed:
            lines.append("")
            lines.append("**Documentation:**")
            lines.append("")
            lines.append(
                f"* line similarity: {self.doc.similarity:.2f} "
                f"(+{self.doc.added_lines}/-{self.doc.removed_lines} lines)"
            )
            if self.doc.unified_diff:
                lines.append("")
                lines.append(".. code-block:: diff")
                lines.append("")
                for ud_line in self.doc.unified_diff:
                    lines.append(f"    {ud_line}")
        return "\n".join(lines)


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

    .. runpython::
        :showcode:

        from onnx_light.onnx import defs
        from onnx_light.compatibility.schema_diff import compare_schemas
        defs.register_onnx_operator_set_schema()
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        print(diff)

    The function also accepts lightweight schemas exposed by ``onnx_light``
    (``onnx_proto._onnxpy.onnx_op.LightOpSchema``).  Those schemas do not
    expose attributes nor input/output arity, so those parts of the diff are
    simply omitted when both schemas lack them.

    .. runpython::
        :showcode:

        from collections import defaultdict
        from onnx_light.onnx_proto._onnxpy import onnx_op
        from onnx_light.compatibility.schema_diff import compare_schemas

        by_name = defaultdict(list)
        for s in onnx_op.GetAllOnnxOpSchemasWithHistory(True):
            by_name[(s.domain, s.name)].append(s)
        versions = sorted(by_name[("ai.onnx", "Add")], key=lambda s: s.since_version)
        old, new = versions[0], versions[-1]
        diff = compare_schemas(old, new)
        print(diff)
    """
    input_diffs = ParameterDiff.compare(list(schema_old.inputs), list(schema_new.inputs), "input")
    output_diffs = ParameterDiff.compare(
        list(schema_old.outputs), list(schema_new.outputs), "output"
    )
    old_attrs = getattr(schema_old, "attributes", None)
    new_attrs = getattr(schema_new, "attributes", None)
    # ``LightOpSchema.attributes`` is a list of ``AttributeParam`` records,
    # whereas the full ``OpSchema.attributes`` is a name-keyed mapping.
    # Normalise to a mapping so ``AttributeDiff.compare`` can consume either.
    if isinstance(old_attrs, list):
        old_attrs = {a.name: a for a in old_attrs}
    if isinstance(new_attrs, list):
        new_attrs = {a.name: a for a in new_attrs}
    if old_attrs is None and new_attrs is None:
        attr_diffs: list[AttributeDiff] = []
    else:
        attr_diffs = AttributeDiff.compare(old_attrs or {}, new_attrs or {})
    constraint_diffs = ConstraintDiff.compare(
        list(schema_old.type_constraints), list(schema_new.type_constraints)
    )

    # Also compare min/max input/output arity (only when both schemas expose it).
    extra_input_diffs: list[ParameterDiff] = []
    extra_output_diffs: list[ParameterDiff] = []
    has_input_arity = hasattr(schema_old, "min_input") and hasattr(schema_new, "min_input")
    has_output_arity = hasattr(schema_old, "min_output") and hasattr(schema_new, "min_output")
    if has_input_arity and (
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

    if has_output_arity and (
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
    for di in all_input_diffs:
        if di.is_breaking:
            breaking_reasons.append(f"input '{di.name}' ({di.kind}): {'; '.join(di.details)}")
    for do in all_output_diffs:
        if do.is_breaking:
            breaking_reasons.append(f"output '{do.name}' ({do.kind}): {'; '.join(do.details)}")
    for da in attr_diffs:
        if da.is_breaking:
            breaking_reasons.append(f"attribute '{da.name}' ({da.kind}): {'; '.join(da.details)}")
    for dc in constraint_diffs:
        if dc.is_breaking:
            breaking_reasons.append(
                f"type constraint '{dc.name}' ({dc.kind}): {'; '.join(dc.details)}"
            )

    doc_diff = DocDiff.compare(
        getattr(schema_old, "doc", "") or "",
        getattr(schema_new, "doc", "") or "",
        old_label=f"{schema_old.name} v{schema_old.since_version}",
        new_label=f"{schema_new.name} v{schema_new.since_version}",
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
        doc=doc_diff,
        is_breaking=bool(breaking_reasons),
        breaking_reasons=breaking_reasons,
    )
