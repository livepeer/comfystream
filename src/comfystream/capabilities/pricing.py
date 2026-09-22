"""Committed-catalog sell price for fal single-shot capabilities.

Advertised Livepeer price is always ``unit=fixed``. No live estimates.
Sell price = unit_price * expected_units * (1 + upcharge_bps / 10000).
An explicit ``price_info`` override wins.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

SELL_PRICE_QUANTUM = Decimal("0.000000000001")
PRICE_UNITS = frozenset({"fixed", "hour", "720p"})


class PricingError(ValueError):
    """Raised when catalog or overlay pricing cannot be resolved."""


def _positive_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise PricingError(f"{field} must be a positive number")
    try:
        quantity = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise PricingError(f"{field} must be a positive number") from error
    if quantity <= 0:
        raise PricingError(f"{field} must be a positive number")
    return quantity


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PricingError(f"{field} must be a non-negative integer")
    return value


def explicit_price_info(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PricingError(f"{field} is unresolved")
    price = value.get("price")
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        raise PricingError(f"{field}.price must be positive")
    currency = value.get("currency")
    unit = value.get("unit")
    if currency != "usd":
        raise PricingError(f"{field}.currency must be usd")
    if unit not in PRICE_UNITS:
        raise PricingError(f"{field}.unit must be one of {', '.join(sorted(PRICE_UNITS))}")
    return {"currency": currency, "price": float(price), "unit": unit}


def catalog_unit_price(route_document: dict[str, Any], capability: str) -> Decimal:
    raw = route_document.get("pricing")
    if not isinstance(raw, dict):
        raise PricingError(f"route model {capability} pricing must be an object")
    return _positive_decimal(raw.get("unit_price"), f"{capability}.pricing.unit_price")


def derived_price_info(
    *,
    unit_price: Decimal,
    expected_units: Decimal,
    upcharge_bps: int,
) -> dict[str, Any]:
    sell = (unit_price * expected_units * Decimal(10000 + upcharge_bps)) / Decimal(10000)
    quantized = sell.quantize(SELL_PRICE_QUANTUM, rounding=ROUND_HALF_UP)
    price = float(quantized)
    if not math.isfinite(price) or price <= 0:
        raise PricingError("derived sell price must be a positive finite number")
    return {"currency": "usd", "price": price, "unit": "fixed"}


def resolve_price_info(
    *,
    capability: str,
    route_document: dict[str, Any],
    selection: dict[str, Any],
    default_upcharge_bps: int,
    default_expected_units: Decimal,
) -> dict[str, Any]:
    explicit = selection.get("price_info")
    if isinstance(explicit, dict) and explicit.get("price") is not None:
        return explicit_price_info(explicit, f"{capability}.price_info")
    unit_price = catalog_unit_price(route_document, capability)
    upcharge_bps = default_upcharge_bps
    if "upcharge_bps" in selection:
        upcharge_bps = _non_negative_int(selection.get("upcharge_bps"), f"{capability}.upcharge_bps")
    expected_units = default_expected_units
    if "expected_units" in selection:
        expected_units = _positive_decimal(
            selection.get("expected_units"), f"{capability}.expected_units"
        )
    return derived_price_info(
        unit_price=unit_price,
        expected_units=expected_units,
        upcharge_bps=upcharge_bps,
    )
