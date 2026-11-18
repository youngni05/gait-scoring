from uuid import UUID

import jwt
from jwt.exceptions import InvalidTokenError

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.config import JWT_SECRET_KEY


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
        )

        user_id = UUID(payload["sub"])

    except (InvalidTokenError, ValueError, KeyError, TypeError):
        raise credentials_exception

    user = db.get(User, user_id)

    if user is None:
        raise credentials_exception

    return user
