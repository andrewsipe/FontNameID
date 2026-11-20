#!/usr/bin/env python3
"""
Unified interface for matching name records in TTX and binary fonts.

Handles the differences between string-based TTX attributes and
integer-based binary attributes transparently.

Usage:
    from FontCore.core_namerecord_matcher import NameRecordMatcher

    # TTX matching
    matcher = NameRecordMatcher.for_ttx(name_id=1, pid=3, eid=1, lang="0x409")
    for nr in name_table.findall("namerecord"):
        if matcher.matches_ttx(nr):
            process(nr)

    # Binary matching
    matcher = NameRecordMatcher.for_binary(name_id=1, pid=3, eid=1, lang=0x409)
    for record in font["name"].names:
        if matcher.matches_binary(record):
            process(record)

    # Unified iteration
    for record in matcher.iter_matches_ttx(name_table):
        process(record)
"""

from __future__ import annotations
from typing import Iterator, Optional, Any
from dataclasses import dataclass
from FontCore.core_logging_config import get_logger

logger = get_logger(__name__)

# Platform/encoding constants
PID_WIN = 3
EID_UNICODE_BMP = 1
LANG_EN_US_HEX = "0x409"
LANG_EN_US_INT = 0x0409


@dataclass(frozen=True)
class NameRecordCriteria:
    """
    Immutable criteria for matching name records.

    Stores criteria in both TTX (string) and binary (int) formats
    for transparent matching across both representations.
    """

    name_id: int
    platform_id: int
    plat_enc_id: int
    lang_id_hex: str  # TTX format: "0x409"
    lang_id_int: int  # Binary format: 0x0409

    @classmethod
    def create(
        cls,
        name_id: int,
        platform_id: int = PID_WIN,
        plat_enc_id: int = EID_UNICODE_BMP,
        lang_id: int | str = LANG_EN_US_INT,
    ) -> "NameRecordCriteria":
        """
        Create criteria from flexible input.

        Args:
            name_id: Name ID to match
            platform_id: Platform ID (default: 3 = Windows)
            plat_enc_id: Platform encoding ID (default: 1 = Unicode BMP)
            lang_id: Language ID as int (0x0409) or string ("0x409")

        Returns:
            Immutable criteria object

        Examples:
            >>> c = NameRecordCriteria.create(1, lang_id=0x0409)
            >>> c.lang_id_hex
            '0x409'
            >>> c = NameRecordCriteria.create(1, lang_id="0x409")
            >>> c.lang_id_int
            1033
        """
        # Normalize lang_id to both formats
        if isinstance(lang_id, str):
            lang_id_hex = lang_id
            try:
                lang_id_int = (
                    int(lang_id, 16) if lang_id.startswith("0x") else int(lang_id)
                )
            except ValueError:
                logger.warning(f"Invalid lang_id string '{lang_id}', using 0x409")
                lang_id_int = LANG_EN_US_INT
                lang_id_hex = LANG_EN_US_HEX
        else:
            lang_id_int = int(lang_id)
            lang_id_hex = f"0x{lang_id_int:x}"

        return cls(
            name_id=int(name_id),
            platform_id=int(platform_id),
            plat_enc_id=int(plat_enc_id),
            lang_id_hex=lang_id_hex,
            lang_id_int=lang_id_int,
        )

    def __str__(self) -> str:
        return (
            f"nameID={self.name_id}, "
            f"platformID={self.platform_id}, "
            f"platEncID={self.plat_enc_id}, "
            f"langID={self.lang_id_hex}"
        )


class NameRecordMatcher:
    """
    Unified interface for matching name records in TTX and binary fonts.

    Provides consistent matching logic across both representations,
    handling string vs. integer attribute differences transparently.
    """

    def __init__(self, criteria: NameRecordCriteria):
        """
        Initialize matcher with criteria.

        Args:
            criteria: Matching criteria
        """
        self.criteria = criteria

    @classmethod
    def for_ttx(
        cls,
        name_id: int,
        platform_id: int = PID_WIN,
        plat_enc_id: int = EID_UNICODE_BMP,
        lang_id: str = LANG_EN_US_HEX,
    ) -> "NameRecordMatcher":
        """
        Create matcher for TTX format (string attributes).

        Args:
            name_id: Name ID to match
            platform_id: Platform ID
            plat_enc_id: Platform encoding ID
            lang_id: Language ID as hex string

        Returns:
            Configured matcher instance

        Examples:
            >>> matcher = NameRecordMatcher.for_ttx(1, lang_id="0x409")
            >>> matcher.criteria.name_id
            1
        """
        criteria = NameRecordCriteria.create(name_id, platform_id, plat_enc_id, lang_id)
        return cls(criteria)

    @classmethod
    def for_binary(
        cls,
        name_id: int,
        platform_id: int = PID_WIN,
        plat_enc_id: int = EID_UNICODE_BMP,
        lang_id: int = LANG_EN_US_INT,
    ) -> "NameRecordMatcher":
        """
        Create matcher for binary format (integer attributes).

        Args:
            name_id: Name ID to match
            platform_id: Platform ID
            plat_enc_id: Platform encoding ID
            lang_id: Language ID as integer

        Returns:
            Configured matcher instance

        Examples:
            >>> matcher = NameRecordMatcher.for_binary(1, lang_id=0x0409)
            >>> matcher.criteria.lang_id_int
            1033
        """
        criteria = NameRecordCriteria.create(name_id, platform_id, plat_enc_id, lang_id)
        return cls(criteria)

    def matches_ttx(self, element: Any) -> bool:
        """
        Check if TTX namerecord element matches criteria.

        Args:
            element: XML element to check

        Returns:
            True if element matches all criteria

        Examples:
            >>> # Assuming element with nameID="1", platformID="3", etc.
            >>> matcher = NameRecordMatcher.for_ttx(1)
            >>> matcher.matches_ttx(element)
            True
        """
        try:
            return (
                element.get("nameID") == str(self.criteria.name_id)
                and element.get("platformID") == str(self.criteria.platform_id)
                and element.get("platEncID") == str(self.criteria.plat_enc_id)
                and element.get("langID") == self.criteria.lang_id_hex
            )
        except AttributeError as e:
            logger.debug(f"Invalid element for TTX matching: {e}")
            return False

    def matches_binary(self, record: Any) -> bool:
        """
        Check if binary name record matches criteria.

        Args:
            record: Binary name record to check

        Returns:
            True if record matches all criteria

        Examples:
            >>> # Assuming record with nameID=1, platformID=3, etc.
            >>> matcher = NameRecordMatcher.for_binary(1)
            >>> matcher.matches_binary(record)
            True
        """
        try:
            return (
                getattr(record, "nameID", None) == self.criteria.name_id
                and getattr(record, "platformID", None) == self.criteria.platform_id
                and getattr(record, "platEncID", None) == self.criteria.plat_enc_id
                and getattr(record, "langID", None) == self.criteria.lang_id_int
            )
        except Exception as e:
            logger.debug(f"Invalid record for binary matching: {e}")
            return False

    def iter_matches_ttx(self, name_table: Any) -> Iterator[Any]:
        """
        Iterate over matching TTX namerecord elements.

        Args:
            name_table: TTX name table element

        Yields:
            Matching namerecord elements

        Examples:
            >>> matcher = NameRecordMatcher.for_ttx(1)
            >>> for nr in matcher.iter_matches_ttx(name_table):
            ...     print(nr.text)
        """
        for element in name_table.findall("namerecord"):
            if self.matches_ttx(element):
                yield element

    def iter_matches_binary(self, name_table: Any) -> Iterator[Any]:
        """
        Iterate over matching binary name records.

        Args:
            name_table: Binary name table

        Yields:
            Matching name records

        Examples:
            >>> matcher = NameRecordMatcher.for_binary(1)
            >>> for record in matcher.iter_matches_binary(font["name"]):
            ...     print(record.toUnicode())
        """
        for record in name_table.names:
            if self.matches_binary(record):
                yield record

    def find_first_ttx(self, name_table: Any) -> Optional[Any]:
        """
        Find first matching TTX namerecord element.

        Args:
            name_table: TTX name table element

        Returns:
            First matching element or None

        Examples:
            >>> matcher = NameRecordMatcher.for_ttx(1)
            >>> nr = matcher.find_first_ttx(name_table)
            >>> if nr:
            ...     print(nr.text)
        """
        for element in self.iter_matches_ttx(name_table):
            return element
        return None

    def find_first_binary(self, name_table: Any) -> Optional[Any]:
        """
        Find first matching binary name record.

        Args:
            name_table: Binary name table

        Returns:
            First matching record or None

        Examples:
            >>> matcher = NameRecordMatcher.for_binary(1)
            >>> record = matcher.find_first_binary(font["name"])
            >>> if record:
            ...     print(record.toUnicode())
        """
        for record in self.iter_matches_binary(name_table):
            return record
        return None

    def count_matches_ttx(self, name_table: Any) -> int:
        """
        Count matching TTX namerecord elements.

        Args:
            name_table: TTX name table element

        Returns:
            Number of matching elements
        """
        return sum(1 for _ in self.iter_matches_ttx(name_table))

    def count_matches_binary(self, name_table: Any) -> int:
        """
        Count matching binary name records.

        Args:
            name_table: Binary name table

        Returns:
            Number of matching records
        """
        return sum(1 for _ in self.iter_matches_binary(name_table))


# Convenience functions for backward compatibility
def find_namerecord_ttx(
    name_table: Any,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> Optional[Any]:
    """
    Find first matching TTX namerecord (backward compatible wrapper).

    Examples:
        >>> nr = find_namerecord_ttx(name_table, 1)
    """
    matcher = NameRecordMatcher.for_ttx(name_id, pid, eid, lang)
    return matcher.find_first_ttx(name_table)


def find_namerecord_binary(
    name_table: Any,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: int = LANG_EN_US_INT,
) -> Optional[Any]:
    """
    Find first matching binary namerecord (new function).

    Examples:
        >>> record = find_namerecord_binary(font["name"], 1)
    """
    matcher = NameRecordMatcher.for_binary(name_id, pid, eid, lang)
    return matcher.find_first_binary(name_table)


__all__ = [
    "NameRecordCriteria",
    "NameRecordMatcher",
    "find_namerecord_ttx",
    "find_namerecord_binary",
    "PID_WIN",
    "EID_UNICODE_BMP",
    "LANG_EN_US_HEX",
    "LANG_EN_US_INT",
]
