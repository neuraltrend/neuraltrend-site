# NeuralTrend automated testing and deployment verification

## Local or CI test suite

Install production and development dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov=models --cov=tools --cov-report=term-missing
```

The test suite creates a temporary SQLite database and temporary Forward Record
storage before importing the application. Email sending is suppressed, Stripe
calls are mocked, rate limits are disabled, and no production CSV is modified.

## Render prelaunch check

From the Render web-service Shell, after migrations and environment variables
are in place:

```bash
python tools/prelaunch_check.py
```

This checks configuration presence, database connectivity/schema, persistent
Forward Record storage, and BTC market-data availability. It does not call
Stripe or send email.

## Read-only production smoke test

From any machine with the repository dependencies installed:

```bash
python tools/production_smoke_test.py https://neuraltrend.org
```

Add the heavier Signal Overview calculation only when needed:

```bash
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

## GitHub Actions

`.github/workflows/tests.yml` runs Python compilation and the pytest suite on
every push and pull request. A failed workflow should block deployment until the
failure is understood.
