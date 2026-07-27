"""Create or update NeuralTrend's Stripe Customer Portal configuration.

Run this once per Stripe mode (test or live), then copy the printed
STRIPE_BILLING_PORTAL_CONFIGURATION_ID value into the matching Render service.
"""

from __future__ import annotations

import os
import sys

import stripe


SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
MONTHLY_PRICE_ID = (
    os.environ.get("STRIPE_PRO_MONTHLY_PRICE_ID")
    or os.environ.get("STRIPE_PRO_PRICE_ID")
)
ANNUAL_PRICE_ID = os.environ.get("STRIPE_PRO_ANNUAL_PRICE_ID")
EXISTING_CONFIGURATION_ID = os.environ.get(
    "STRIPE_BILLING_PORTAL_CONFIGURATION_ID"
)
BASE_URL = os.environ.get("BASE_URL", "https://neuraltrend.org").rstrip("/")
CONFIG_MARKER = "neuraltrend_monthly_annual_v1"


def to_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "_to_dict_recursive"):
        return value._to_dict_recursive()
    if hasattr(value, "to_dict_recursive"):
        return value.to_dict_recursive()
    return dict(value)


def require_environment() -> None:
    missing = []

    if not SECRET_KEY:
        missing.append("STRIPE_SECRET_KEY")
    if not MONTHLY_PRICE_ID:
        missing.append("STRIPE_PRO_MONTHLY_PRICE_ID or STRIPE_PRO_PRICE_ID")
    if not ANNUAL_PRICE_ID:
        missing.append("STRIPE_PRO_ANNUAL_PRICE_ID")

    if missing:
        raise RuntimeError("Missing environment value(s): " + ", ".join(missing))


def build_features(product_id: str) -> dict:
    return {
        "customer_update": {
            "enabled": True,
            "allowed_updates": ["email", "name", "address"],
        },
        "invoice_history": {"enabled": True},
        "payment_method_update": {"enabled": True},
        "subscription_cancel": {
            "enabled": True,
            "mode": "at_period_end",
            "proration_behavior": "none",
            "cancellation_reason": {
                "enabled": True,
                "options": [
                    "too_expensive",
                    "missing_features",
                    "unused",
                    "switched_service",
                    "other",
                ],
            },
        },
        "subscription_update": {
            "enabled": True,
            "default_allowed_updates": ["price"],
            "proration_behavior": "always_invoice",
            "products": [
                {
                    "product": product_id,
                    "prices": [MONTHLY_PRICE_ID, ANNUAL_PRICE_ID],
                }
            ],
        },
    }


def find_existing_configuration():
    configurations = stripe.billing_portal.Configuration.list(limit=100)
    configurations = to_dict(configurations)

    for configuration in configurations.get("data", []):
        metadata = configuration.get("metadata") or {}
        if metadata.get("neuraltrend_configuration") == CONFIG_MARKER:
            return configuration

    return None


def main() -> int:
    require_environment()
    stripe.api_key = SECRET_KEY

    monthly_price = to_dict(stripe.Price.retrieve(MONTHLY_PRICE_ID))
    annual_price = to_dict(stripe.Price.retrieve(ANNUAL_PRICE_ID))

    monthly_product = monthly_price.get("product")
    annual_product = annual_price.get("product")

    if not monthly_product or monthly_product != annual_product:
        raise RuntimeError(
            "Monthly and annual prices must belong to the same Stripe Product."
        )

    features = build_features(monthly_product)
    configuration = None

    if EXISTING_CONFIGURATION_ID:
        configuration = to_dict(
            stripe.billing_portal.Configuration.retrieve(
                EXISTING_CONFIGURATION_ID
            )
        )
    else:
        configuration = find_existing_configuration()

    parameters = {
        "name": "NeuralTrend Pro monthly and annual billing",
        "default_return_url": f"{BASE_URL}/subscription",
        "business_profile": {
            "headline": "Manage your NeuralTrend Pro subscription",
            "privacy_policy_url": f"{BASE_URL}/privacy",
            "terms_of_service_url": f"{BASE_URL}/terms",
        },
        "features": features,
        "metadata": {
            "neuraltrend_configuration": CONFIG_MARKER,
        },
    }

    if configuration:
        configuration = stripe.billing_portal.Configuration.modify(
            configuration["id"],
            **parameters,
        )
        action = "updated"
    else:
        configuration = stripe.billing_portal.Configuration.create(**parameters)
        action = "created"

    configuration = to_dict(configuration)
    configuration_id = configuration.get("id")

    if not configuration_id:
        raise RuntimeError("Stripe did not return a portal configuration ID.")

    print(f"Portal configuration {action}: {configuration_id}")
    print()
    print("Add this environment variable to the matching Render service:")
    print(f"STRIPE_BILLING_PORTAL_CONFIGURATION_ID={configuration_id}")
    print()
    print("Run the script separately with test and live Stripe keys.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except stripe.error.StripeError as exc:
        print(f"Stripe error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1)
