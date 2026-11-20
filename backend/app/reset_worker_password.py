"""Local administrator CLI. Never accepts passwords as command-line arguments."""
import argparse
import warnings
from getpass import GetPassWarning, getpass

from sqlalchemy import select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset an existing worker password locally.")
    parser.add_argument("login_id", help="Existing login ID")
    args = parser.parse_args()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.login_id == args.login_id))
        if user is None:
            print("Account not found. No changes made.")
            return 1
        try:
            # Fail instead of falling back to visible input when no terminal is available.
            with warnings.catch_warnings():
                warnings.simplefilter("error", GetPassWarning)
                password = getpass("New password (12+ characters): ")
                confirmation = getpass("Confirm new password: ")
        except (GetPassWarning, EOFError, KeyboardInterrupt):
            print("\nCancelled or secure terminal unavailable. No changes made.")
            return 1
        if len(password) < 12:
            print("Password must be at least 12 characters. No changes made.")
            return 1
        if password != confirmation:
            print("Passwords do not match. No changes made.")
            return 1
        user.password_hash = hash_password(password)
        db.commit()
    print("Password updated. Sign in with your new password. Existing records are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
