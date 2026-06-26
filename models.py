from flask_login import UserMixin
from datetime import datetime
from extensions import db

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

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
        nullable=True
    )

    def __repr__(self):
        return f"<User {self.email}>"

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
