"""The release operator dispatches builds without compiling locally."""
import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mario_release', ROOT / 'scripts/release.py')
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseOperatorTests(unittest.TestCase):
    def test_locks_do_not_compile(self):
        with patch.object(release, 'run') as run:
            release.refresh_locks()
        self.assertEqual([c.args[0] for c in run.call_args_list], [
            ['uv', 'lock', '--check'], ['cargo', 'metadata', '--locked', '--no-deps']
        ])

    def test_validation_uses_exact_remote_sha(self):
        sha = 'a' * 40
        with patch.object(release, 'upstream_ref', return_value='origin/main'), patch.object(
            release, 'capture', return_value=sha
        ), patch.object(release, 'run') as run:
            release.validate()
        self.assertEqual(run.call_args_list[-1].args[0], [
            'gh', 'workflow', 'run', 'release.yml', '--ref', 'main', '-f', f'ref={sha}'
        ])

    def test_validation_never_prepares_release_metadata(self):
        with patch.object(release, 'parse_args', return_value=argparse.Namespace(validate=True)), patch.object(
            release, 'validate'
        ) as validate, patch.object(release, 'ensure_clean') as clean:
            release.main()
        validate.assert_called_once()
        clean.assert_not_called()

    def test_metadata_helper_uses_operator_python(self):
        self.assertEqual(release.PYTHON, Path(sys.executable))


if __name__ == '__main__':
    unittest.main()
