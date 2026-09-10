#!/usr/bin/env python3
"""Generate a base32 TOTP secret for NeuralTrend administrator MFA.

Store the secret in the deployment environment, never in Git. For one admin:
  ADMIN_TOTP_SECRET=<secret>
For multiple admins use ADMIN_TOTP_SECRETS as a JSON object keyed by email.
"""
from __future__ import annotations

import argparse
import base64
import secrets
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Administrator email")
    args = parser.parse_args()

    secret = base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")
    label = quote(f"NeuralTrend:{args.email}", safe="")
    issuer = quote("NeuralTrend", safe="")
    uri = f"otpauth://totp/{label}?secret={secret}&issuer={issuer}&digits=6&period=30"

    print("ADMIN_TOTP_SECRET=" + secret)
    print("Authenticator setup URI:")
    print(uri)
    print("Keep this secret private. Do not commit it to the repository.")


if __name__ == "__main__":
    main()
