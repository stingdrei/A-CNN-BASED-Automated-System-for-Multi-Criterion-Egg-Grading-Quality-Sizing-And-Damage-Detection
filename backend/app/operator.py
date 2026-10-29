"""Single-operator context for the auth-free MVP API.

The MVP is designed for one local inspection station. Persistence still uses
the existing ``users`` table so predictions keep their foreign key, but every
request resolves to one deterministic operator row instead of a bearer token.
"""

from fastapi import Depends
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User

LOCAL_OPERATOR_EMAIL = "operator@localhost"
LOCAL_OPERATOR_USERNAME = "operator"
LOCAL_OPERATOR_PASSWORD_HASH = "!auth-removed-for-mvp"


def get_operator(db: Session = Depends(get_db)) -> User:
    """Return the local MVP operator, creating it on first use."""
    operator = (
        db.query(User).filter(User.email == LOCAL_OPERATOR_EMAIL).first()
    )
    if operator is not None:
        return operator

    operator = User(
        email=LOCAL_OPERATOR_EMAIL,
        username=LOCAL_OPERATOR_USERNAME,
        password_hash=LOCAL_OPERATOR_PASSWORD_HASH,
        is_active=True,
        mm_per_pixel=settings.MM_PER_PIXEL,
    )
    db.add(operator)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        operator = (
            db.query(User).filter(User.email == LOCAL_OPERATOR_EMAIL).first()
        )
        if operator is None:
            raise
        return operator

    db.refresh(operator)
    return operator
