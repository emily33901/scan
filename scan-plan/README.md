# Typed scan plans

`ReadSigned { offset, width }` reads an address-relative scalar using the target
byte order and sign extends it to `Value::Signed(i64)`. `SignedIntegerWidth`
supports `I8`, `I16`, `I32`, and `I64`. Scalars remain terminal values unless a
later operation explicitly accepts their type; address operations reject them.
Signed reads use the normal read count and byte budgets with `ReadPurpose::Signed`.

`X86ResolvePcRelative { displacement_offset, instruction_length }` resolves an
x64 RIP-relative signed disp32 using the end of the entire instruction as its
origin. For `83 3d f0 ff ff ff 01` (`cmp dword ptr [rip-16], 1`), pass offset 2
and length 7. The target is instruction start minus 9. The displacement need not
be the final field; a trailing immediate is included in the origin.

The resolver requires x86-64, an instruction length of 1 through 15 bytes, and a
four-byte displacement field wholly within that length. Metadata must come from
a validated decoder; this step does not validate instruction opcodes or operands.
It reads only the displacement's four bytes, charged against instruction, read
count, and read byte budgets. Instruction-end and target arithmetic are checked.

Schema 2 encodes each step as a table keyed by operation name. Steps without
operands take an empty table. For example:

```toml
schema_version = 2
result = "address"
steps = [
    { read_unsigned = { offset = 0, width = "u64" } },
    { unsigned_to_address = {} },
]
```

Scan's `MethodStep` compatibility API exposes both new actions. Portable `Plan`
evaluation and direct, scoped, and fault-contained memory adapters share the same
evaluator, typed results, limits, and error reporting.
