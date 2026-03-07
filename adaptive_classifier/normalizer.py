"""Text normalization utilities.

Pre-processing messy POS/transaction text before classification
dramatically improves both embedding and LLM accuracy.
"""

from __future__ import annotations

import re
from typing import Callable


# Common abbreviations found in POS / transaction data
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    # Food
    "chz": "cheese",
    "brgr": "burger",
    "hmbrgr": "hamburger",
    "pza": "pizza",
    "pep": "pepperoni",
    "chkn": "chicken",
    "sndwch": "sandwich",
    "sm": "small",
    "md": "medium",
    "lg": "large",
    "xl": "extra large",
    "dbl": "double",
    "trpl": "triple",
    "fry": "fries",
    "bev": "beverage",
    "coff": "coffee",
    "esprs": "espresso",
    "org": "organic",
    "dec": "decaf",
    # Transactions / banking
    "txn": "transaction",
    "dep": "deposit",
    "wdl": "withdrawal",
    "xfer": "transfer",
    "pymt": "payment",
    "pmt": "payment",
    "chq": "cheque",
    "int": "interest",
    "svc": "service",
    "maint": "maintenance",
    "ins": "insurance",
    "groc": "grocery",
    "pharm": "pharmacy",
    "rest": "restaurant",
    "util": "utility",
    "tel": "telephone",
    "govt": "government",
    "mkt": "market",
    "apt": "apartment",
    "mtg": "mortgage",
    # Retail / inventory
    "qty": "quantity",
    "pcs": "pieces",
    "ea": "each",
    "pkg": "package",
    "cs": "case",
    "bx": "box",
    "ctn": "carton",
    "dz": "dozen",
    "pr": "pair",
    "set": "set",
    "assy": "assembly",
    "hw": "hardware",
    "sw": "software",
    "elec": "electronic",
    "furn": "furniture",
    "bkshf": "bookshelf",
    "tbl": "table",
    "chr": "chair",
    "cab": "cabinet",
}


def create_normalizer(
    abbreviations: dict[str, str] | None = None,
    lowercase: bool = True,
    strip_amounts: bool = True,
    strip_codes: bool = True,
    custom_fn: Callable[[str], str] | None = None,
) -> Callable[[str], str]:
    """Create a normalizer pipeline.

    Args:
        abbreviations: Custom abbreviations dict. Merges with defaults if provided.
        lowercase: Convert to lowercase.
        strip_amounts: Remove dollar amounts, quantities.
        strip_codes: Remove transaction codes, reference numbers.
        custom_fn: Additional custom normalization applied last.

    Returns:
        A normalizer function: str -> str
    """
    abbrevs = {**DEFAULT_ABBREVIATIONS}
    if abbreviations:
        abbrevs.update(abbreviations)

    # Pre-compile the abbreviation pattern
    # Sort by length (longest first) to handle overlapping abbreviations
    sorted_abbrevs = sorted(abbrevs.keys(), key=len, reverse=True)
    abbrev_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(a) for a in sorted_abbrevs) + r')\b',
        re.IGNORECASE,
    )

    def normalize(text: str) -> str:
        result = text.strip()

        if lowercase:
            result = result.lower()

        if strip_amounts:
            # Remove dollar amounts: $12.34, $1,234.56
            result = re.sub(r'\$[\d,]+\.?\d*', '', result)
            # Remove trailing quantities: x2, x 3, qty:5
            result = re.sub(r'\bx\s*\d+\b', '', result)
            result = re.sub(r'\bqty[:\s]*\d+\b', '', result)

        if strip_codes:
            # Remove transaction/reference codes: REF#123456, TXN-789
            result = re.sub(r'\b[A-Z]{2,}[#\-]\d+\b', '', result, flags=re.IGNORECASE)
            # Remove pure numeric codes (4+ digits)
            result = re.sub(r'\b\d{4,}\b', '', result)

        # Expand abbreviations
        def _expand(match):
            return abbrevs.get(match.group(0).lower(), match.group(0))
        result = abbrev_pattern.sub(_expand, result)

        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        if custom_fn:
            result = custom_fn(result)

        return result

    return normalize
