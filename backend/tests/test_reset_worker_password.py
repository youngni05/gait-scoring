import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from getpass import GetPassWarning

from app import reset_worker_password as cli
from app.security import verify_password


class ResetWorkerPasswordTests(unittest.TestCase):
    def run_cli(self, values, found=True):
        user = SimpleNamespace(id='original-id', password_hash='original-hash')
        db = MagicMock()
        db.scalar.return_value = user if found else None
        with patch.object(cli, 'SessionLocal') as factory, patch.object(cli, 'getpass', side_effect=values), patch('sys.argv', ['reset', 'worker01']), patch('builtins.print'):
            factory.return_value.__enter__.return_value = db
            result = cli.main()
        return result, user, db

    def test_only_password_changes_after_matching_confirmation(self):
        result, user, db = self.run_cli(['synthetic-password-123', 'synthetic-password-123'])
        self.assertEqual(result, 0)
        self.assertEqual(user.id, 'original-id')
        self.assertTrue(verify_password('synthetic-password-123', user.password_hash))
        db.commit.assert_called_once()

    def test_invalid_or_cancelled_input_does_not_commit(self):
        for values in (['short', 'short'], ['synthetic-password-123', 'different-password-456'], [EOFError()], [KeyboardInterrupt()], [GetPassWarning()]):
            with self.subTest(values=type(values[0]).__name__):
                result, user, db = self.run_cli(values)
                self.assertEqual(result, 1)
                self.assertEqual(user.password_hash, 'original-hash')
                db.commit.assert_not_called()

    def test_missing_account_is_not_created(self):
        result, _, db = self.run_cli([], found=False)
        self.assertEqual(result, 1)
        db.add.assert_not_called()
        db.commit.assert_not_called()
