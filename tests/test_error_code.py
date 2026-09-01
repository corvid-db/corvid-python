"""The frozen error-code table (docs/SURFACE.tsv: the corvid::Error rows).

The fixtures prove the codes the suite can trigger (err:10/11/12/14/15/17);
the redb-internal fault variants have no public trigger (the engine's own
radar exempts them), so the table itself — asserted here verbatim — is the
proof that every engine ``Error`` variant maps to its frozen C-ABI code
(FFI.md §1.3: values are never renumbered). Code 19 (BUSY) is FFI-only:
compact exclusivity, with no engine ``Error`` variant behind it.
"""

import corvid

FROZEN = {
    "DATABASE": 1,
    "TRANSACTION": 2,
    "TABLE": 3,
    "STORAGE": 4,
    "COMMIT": 5,
    "SET_DURABILITY": 6,
    "COMPACTION": 7,
    "DECODE": 8,
    "CORRUPT_INDEX": 9,
    "RESERVED_COLLECTION": 10,
    "INVALID_NAME": 11,
    "INVALID_ARGUMENT": 12,
    "INCOMPATIBLE_FORMAT": 13,
    "EMPTY_INDEX_TRAINING": 14,
    "SCHEMA_VIOLATION": 15,
    "INVALID_DUMP": 16,
    "BACKUP_TARGET_EXISTS": 17,
    "IO": 18,
    "BUSY": 19,  # FFI-only: compact exclusivity
}


def test_error_code_table():
    actual = {name: getattr(corvid.ErrorCode, name) for name in FROZEN}
    assert actual == FROZEN, f"error-code table drifted: {actual}"
    # No extra variants beyond the frozen table.
    exported = {
        name
        for name in dir(corvid.ErrorCode)
        if not name.startswith("_") and isinstance(getattr(corvid.ErrorCode, name), int)
    }
    assert exported == set(FROZEN), f"unexpected ErrorCode members: {exported ^ set(FROZEN)}"
