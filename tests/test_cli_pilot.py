from pathlib import Path

import pytest

from jag_tree.cli import main
from jag_tree.sandbox import TrustedLocalSandbox


CONFIG = str(Path(__file__).resolve().parents[1] / 'configs/pilots/jag_screen16.yaml')


def test_cli_connects_explicit_local_pilot(monkeypatch, tmp_path):
    import jag_tree.trainer as trainer

    def run(config, **kwargs):
        assert isinstance(kwargs['sandbox'], TrustedLocalSandbox)
        assert kwargs['backend'].allow_pilot_predictor is True
        return {'status': 'INCOMPLETE'}

    monkeypatch.setattr(trainer, 'run_experiment', run)
    assert main(['run', CONFIG, '--output-root', str(tmp_path), '--seed', '17',
                 '--sandbox', 'trusted-local', '--allow-pilot-predictor']) == 0


@pytest.mark.parametrize('options', [
    ['--sandbox', 'trusted-local'], ['--allow-pilot-predictor'],
])
def test_pilot_options_reject_formal_before_execution(monkeypatch, tmp_path, capsys, options):
    import jag_tree.cli as cli
    from types import SimpleNamespace

    monkeypatch.setattr(cli, 'load_experiment', lambda _: SimpleNamespace(data={'formal': True}))
    monkeypatch.setattr(cli, 'validate_experiment', lambda _: None)
    assert main(['run', CONFIG, '--output-root', str(tmp_path), '--seed', '17', *options]) == 2
    assert 'formal' in capsys.readouterr().err
