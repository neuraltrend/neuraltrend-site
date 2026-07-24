from flask_login import UserMixin
from datetime import datetime
from extensions import db

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    # One-time password-reset state. Only a SHA-256 digest of the random
    # reset nonce is stored; requesting a new link replaces the old digest.
    password_reset_token_hash = db.Column(
        db.String(64),
        nullable=True,
        index=True,
    )

    password_reset_requested_at = db.Column(
        db.DateTime,
        nullable=True,
    )

    password_changed_at = db.Column(
        db.DateTime,
        nullable=True,
    )

    # Incrementing this value invalidates every older authenticated session.
    auth_version = db.Column(
        db.Integer,
        nullable=False,
        default=1,
    )

    is_verified = db.Column(db.Boolean, default=False)

    failed_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)

    subscription_type = db.Column(
        db.String(50),
        default="free"
    )

    subscription_status = db.Column(
        db.String(50),
        default="inactive"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    last_login = db.Column(
        db.DateTime,
        nullable=True
    )

    stripe_customer_id = db.Column(
        db.String(255),
        nullable=True,
        unique=True,
        index=True
    )

    stripe_subscription_id = db.Column(
        db.String(255),
        nullable=True,
        unique=True,
        index=True
    )

    pending_checkout_session_id = db.Column(
        db.String(255),
        nullable=True,
        unique=True,
        index=True
    )

    checkout_attempt_id = db.Column(
        db.String(64),
        nullable=True,
        unique=True,
        index=True
    )

    checkout_attempt_started_at = db.Column(
        db.DateTime,
        nullable=True
    )

    subscription_updated_at = db.Column(
        db.DateTime,
        nullable=True
    )

    stripe_last_event_id = db.Column(
        db.String(255),
        nullable=True
    )

    stripe_last_event_created_at = db.Column(
        db.DateTime,
        nullable=True
    )

    def __repr__(self):
        return f"<User {self.email}>"

class StripeWebhookEvent(db.Model):
    __tablename__ = "stripe_webhook_events"

    id = db.Column(db.Integer, primary_key=True)

    stripe_event_id = db.Column(
        db.String(255),
        nullable=False,
        unique=True
    )

    event_type = db.Column(db.String(255), nullable=False, index=True)
    stripe_object_id = db.Column(db.String(255), nullable=True, index=True)
    livemode = db.Column(db.Boolean, nullable=False)
    event_created_at = db.Column(db.DateTime, nullable=True)

    processing_status = db.Column(
        db.String(30),
        nullable=False,
        default="processing",
        index=True
    )

    processing_started_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    received_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    processed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.String(1000), nullable=True)

    def __repr__(self):
        return (
            f"<StripeWebhookEvent {self.stripe_event_id} "
            f"{self.processing_status}>"
        )


class LiveSimulation(db.Model):
    __tablename__ = "live_simulations"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "live_simulations",
            lazy=True,
            cascade="all, delete-orphan"
        )
    )

    name = db.Column(db.String(120), nullable=False)

    ticker = db.Column(db.String(30), nullable=False, index=True)
    asset_type = db.Column(db.String(20), nullable=False)  # crypto or stock

    initial_cash = db.Column(db.Float, nullable=False)
    cash_balance = db.Column(db.Float, nullable=False)

    position_quantity = db.Column(db.Float, nullable=False, default=0.0)

    position_size_pct = db.Column(db.Float, nullable=False, default=100.0)
    transaction_cost_rate = db.Column(db.Float, nullable=False, default=0.0)

    benchmark_quantity = db.Column(db.Float, nullable=False, default=0.0)
    benchmark_cash_balance = db.Column(db.Float, nullable=False, default=0.0)

    start_date = db.Column(db.Date, nullable=False)
    last_processed_date = db.Column(db.Date, nullable=True)

    status = db.Column(db.String(20), nullable=False, default="active")

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    trades = db.relationship(
        "LiveSimulationTrade",
        backref="simulation",
        lazy=True,
        cascade="all, delete-orphan"
    )

    equity_points = db.relationship(
        "LiveSimulationEquity",
        backref="simulation",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "initial_cash": self.initial_cash,
            "cash_balance": self.cash_balance,
            "position_quantity": self.position_quantity,
            "position_size_pct": self.position_size_pct,
            "transaction_cost_rate": self.transaction_cost_rate,
            "benchmark_quantity": self.benchmark_quantity,
            "benchmark_cash_balance": self.benchmark_cash_balance,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "last_processed_date": self.last_processed_date.isoformat() if self.last_processed_date else None,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LiveSimulationTrade(db.Model):
    __tablename__ = "live_simulation_trades"

    id = db.Column(db.Integer, primary_key=True)

    simulation_id = db.Column(
        db.Integer,
        db.ForeignKey("live_simulations.id"),
        nullable=False,
        index=True
    )

    trade_date = db.Column(db.Date, nullable=False, index=True)
    ticker = db.Column(db.String(30), nullable=False)

    signal = db.Column(db.Integer, nullable=False)  # 1 = BUY, -1 = SELL
    price = db.Column(db.Float, nullable=False)

    quantity = db.Column(db.Float, nullable=False)
    gross_amount = db.Column(db.Float, nullable=False)
    transaction_cost = db.Column(db.Float, nullable=False)

    cash_after = db.Column(db.Float, nullable=False)
    position_after = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint(
            "simulation_id",
            "trade_date",
            name="uq_live_simulation_trade_date"
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "simulation_id": self.simulation_id,
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "ticker": self.ticker,
            "signal": self.signal,
            "price": self.price,
            "quantity": self.quantity,
            "gross_amount": self.gross_amount,
            "transaction_cost": self.transaction_cost,
            "cash_after": self.cash_after,
            "position_after": self.position_after,
        }


class LiveSimulationEquity(db.Model):
    __tablename__ = "live_simulation_equity"

    id = db.Column(db.Integer, primary_key=True)

    simulation_id = db.Column(
        db.Integer,
        db.ForeignKey("live_simulations.id"),
        nullable=False,
        index=True
    )

    equity_date = db.Column(db.Date, nullable=False, index=True)
    ticker = db.Column(db.String(30), nullable=False)

    signal = db.Column(db.Integer, nullable=False)
    close_price = db.Column(db.Float, nullable=False)

    cash_balance = db.Column(db.Float, nullable=False)
    position_quantity = db.Column(db.Float, nullable=False)

    strategy_value = db.Column(db.Float, nullable=False)
    benchmark_value = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint(
            "simulation_id",
            "equity_date",
            name="uq_live_simulation_equity_date"
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "simulation_id": self.simulation_id,
            "equity_date": self.equity_date.isoformat() if self.equity_date else None,
            "ticker": self.ticker,
            "signal": self.signal,
            "close_price": self.close_price,
            "cash_balance": self.cash_balance,
            "position_quantity": self.position_quantity,
            "strategy_value": self.strategy_value,
            "benchmark_value": self.benchmark_value,
        }


class WatchlistItem(db.Model):
    __tablename__ = "watchlist_items"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "watchlist_items",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    ticker = db.Column(db.String(30), nullable=False, index=True)

    email_alert_enabled = db.Column(
        db.Boolean,
        nullable=False,
        default=False,
    )

    # Baseline used by the scheduled alert worker. Enabling alerts records the
    # currently published signal first, so users receive only future changes.
    last_observed_signal = db.Column(db.SmallInteger, nullable=True)
    last_observed_signal_date = db.Column(db.Date, nullable=True)

    # Fingerprint of the published Date / Close / epoch_signal row last seen by
    # the alert worker. This allows a same-date signal revision to be detected
    # without exposing or hashing proprietary feature columns.
    last_observed_row_fingerprint = db.Column(db.String(64), nullable=True)

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        db.UniqueConstraint(
            "user_id",
            "ticker",
            name="uq_watchlist_user_ticker",
        ),
        db.CheckConstraint(
            "last_observed_signal IS NULL OR last_observed_signal IN (-1, 0, 1)",
            name="ck_watchlist_last_observed_signal",
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "email_alert_enabled": bool(self.email_alert_enabled),
            "last_observed_signal": self.last_observed_signal,
            "last_observed_signal_date": (
                self.last_observed_signal_date.isoformat()
                if self.last_observed_signal_date else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SignalAlertDelivery(db.Model):
    __tablename__ = "signal_alert_deliveries"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    watchlist_item_id = db.Column(
        db.Integer,
        db.ForeignKey("watchlist_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    ticker = db.Column(db.String(30), nullable=False, index=True)
    signal_date = db.Column(db.Date, nullable=False, index=True)
    previous_signal = db.Column(db.SmallInteger, nullable=False)
    current_signal = db.Column(db.SmallInteger, nullable=False)

    # change: one newly dated row changed the signal
    # revision: an already observed signal date was later revised
    # catch_up: two or more newly published rows were processed together
    event_type = db.Column(
        db.String(20),
        nullable=False,
        default="change",
        index=True,
    )
    observation_start_date = db.Column(db.Date, nullable=False)
    observation_end_date = db.Column(db.Date, nullable=False)
    change_count = db.Column(db.Integer, nullable=False, default=1)
    source_row_fingerprint = db.Column(db.String(64), nullable=True)
    event_summary = db.Column(db.Text, nullable=True)

    # Deterministic SHA-256 key prevents two cron workers from claiming the
    # same user/ticker/date/signal transition concurrently.
    event_key = db.Column(
        db.String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    processing_status = db.Column(
        db.String(20),
        nullable=False,
        default="processing",
        index=True,
    )
    processing_started_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    attempted_at = db.Column(db.DateTime, nullable=True)
    sent_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.String(1000), nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    watchlist_item = db.relationship(
        "WatchlistItem",
        backref=db.backref(
            "alert_deliveries",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "signal_alert_deliveries",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    __table_args__ = (
        db.CheckConstraint(
            "previous_signal IN (-1, 0, 1)",
            name="ck_signal_alert_previous_signal",
        ),
        db.CheckConstraint(
            "current_signal IN (-1, 0, 1)",
            name="ck_signal_alert_current_signal",
        ),
        db.CheckConstraint(
            "event_type IN ('change', 'revision', 'catch_up')",
            name="ck_signal_alert_event_type",
        ),
        db.CheckConstraint(
            "change_count >= 1",
            name="ck_signal_alert_change_count",
        ),
        db.CheckConstraint(
            "((event_type IN ('change', 'revision') AND previous_signal <> current_signal) "
            "OR event_type = 'catch_up')",
            name="ck_signal_alert_event_consistency",
        ),
        db.CheckConstraint(
            "processing_status IN ('processing', 'sent', 'failed')",
            name="ck_signal_alert_processing_status",
        ),
    )
