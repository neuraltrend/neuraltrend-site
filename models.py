from flask_login import UserMixin
from datetime import datetime

from app import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(db.String(255), unique=True, nullable=False)

    password_hash = db.Column(db.String(255), nullable=False)

    is_verified = db.Column(db.Boolean, default=False)

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
