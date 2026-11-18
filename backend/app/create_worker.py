from getpass import getpass

from sqlalchemy import select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password


def main():
    login_id = input("Login ID: ").strip()

    if not login_id or len(login_id) > 100:
        print("Invalid login ID")
        return

    with SessionLocal() as db:
        existing = db.scalar(
            select(User).where(User.login_id == login_id)
        )

        if existing:
            print("Login ID already exists")
            return

        password = getpass("Password (12+ characters): ")

        if len(password) < 12:
            print("Password must be at least 12 characters")
            return

        user = User(
            login_id=login_id,
            password_hash=hash_password(password),
        )

        db.add(user)
        db.commit()

        print("Worker account created successfully")


if __name__ == "__main__":
    main()