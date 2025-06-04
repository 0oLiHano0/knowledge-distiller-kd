import sys
import types
import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import typer

import kd_tool.core.cli_main as cli_main
from kd_tool.core.errors import KDToolError  # 确保唯一import

runner = CliRunner()

# ====== Fixtures ======
@pytest.fixture
def fake_config_file(tmp_path):
    config = tmp_path / "kd_config.yaml"
    config.write_text("test: 1")
    return config

@pytest.fixture
def fake_input_file(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("data")
    return f

@pytest.fixture
def fake_input_dir(tmp_path):
    d = tmp_path / "input_dir"
    d.mkdir()
    (d / "a.txt").write_text("a")
    return d

# ====== run命令测试 ======

def test_run_success(fake_input_file, fake_config_file):
    with patch("kd_tool.core.cli_main.ApplicationBuilder") as MockBuilder:
        mock_app = MagicMock()
        mock_logger = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.success = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()
        mock_logger.exception = MagicMock()
        mock_app.logger = mock_logger
        mock_app.run_default_pipeline = MagicMock(side_effect=lambda *a, **kw: print('PIPELINE CALLED'))
        MockBuilder.return_value.build.return_value = mock_app
        with pytest.raises(typer.Exit):
            cli_main.run([fake_input_file], config_file=fake_config_file)

def test_run_config_not_found(fake_input_file, tmp_path):
    # 不存在的配置文件
    not_exist = tmp_path / "no.yaml"
    result = runner.invoke(
        cli_main.app,
        ["run", str(fake_input_file), "-c", str(not_exist)]
    )
    assert result.exit_code == 1
    assert "无法找到配置文件" in result.output

def test_run_default_config_found(fake_input_file, fake_config_file, tmp_path, monkeypatch):
    # 在默认目录创建配置文件
    default_dir = tmp_path / ".kd_tool"
    default_dir.mkdir()
    default_config = default_dir / "kd_config.yaml"
    default_config.write_text("test: 1")
    monkeypatch.chdir(tmp_path)
    # patch DEFAULT_CONFIG_PATHS
    with patch.object(cli_main, "DEFAULT_CONFIG_PATHS", [tmp_path / "kd_config.yaml", default_config]):
        with patch("kd_tool.core.cli_main.ApplicationBuilder") as MockBuilder:
            mock_app = MagicMock()
            mock_logger = MagicMock()
            mock_logger.info = MagicMock()
            mock_logger.success = MagicMock()
            mock_logger.debug = MagicMock()
            mock_logger.warning = MagicMock()
            mock_logger.error = MagicMock()
            mock_logger.exception = MagicMock()
            mock_app.logger = mock_logger
            mock_app.run_default_pipeline = MagicMock(side_effect=lambda *a, **kw: print('PIPELINE CALLED'))
            MockBuilder.return_value.build.return_value = mock_app
            with pytest.raises(typer.Exit):
                cli_main.run([fake_input_file], config_file=None)

def test_run_logger_init_error(fake_input_file, fake_config_file):
    # ApplicationBuilder抛出KDToolError
    err = KDToolError("fail", original_exception=ValueError("bad"))
    def raise_kdtoolerror(*a, **kw):
        print("raise KDToolError", type(err), id(type(err)), id(KDToolError), flush=True)  # 临时调试
        raise err
    with patch("kd_tool.core.cli_main.ApplicationBuilder", side_effect=raise_kdtoolerror):
        result = runner.invoke(
            cli_main.app,
            ["run", str(fake_input_file), "-c", str(fake_config_file)]
        )
        assert result.exit_code == 2
        assert "Logger未能成功初始化" in result.output

def test_run_build_unknown_error(fake_input_file, fake_config_file):
    # ApplicationBuilder抛出其他异常
    with patch("kd_tool.core.cli_main.ApplicationBuilder", side_effect=RuntimeError("fail")):
        result = runner.invoke(
            cli_main.app,
            ["run", str(fake_input_file), "-c", str(fake_config_file)]
        )
        assert result.exit_code == 3
        assert "应用程序构建时发生未知错误" in result.output

def test_run_pipeline_kdtoolerror(fake_input_file, fake_config_file):
    with patch("kd_tool.core.cli_main.ApplicationBuilder") as MockBuilder:
        mock_app = MagicMock()
        mock_logger = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.success = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()
        mock_logger.exception = MagicMock()
        mock_app.logger = mock_logger
        err = KDToolError("pipeline fail")
        def raise_kdtoolerror(*a, **kw):
            print("raise KDToolError", type(err), id(type(err)), id(KDToolError), flush=True)  # 临时调试
            raise err
        mock_app.run_default_pipeline = MagicMock(side_effect=raise_kdtoolerror)
        mock_app.run_default_pipeline.return_value = None
        MockBuilder.return_value.build.return_value = mock_app
        result = runner.invoke(
            cli_main.app,
            ["run", str(fake_input_file), "-c", str(fake_config_file)]
        )
        assert result.exit_code == 4
        assert "流水线执行完成，但存在问题" in result.output

def test_run_pipeline_unknown_error(fake_input_file, fake_config_file):
    # run_default_pipeline抛出Exception
    with patch("kd_tool.core.cli_main.ApplicationBuilder") as MockBuilder:
        mock_app = MagicMock()
        mock_logger = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.success = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()
        mock_logger.exception = MagicMock()
        mock_app.logger = mock_logger
        mock_app.run_default_pipeline = MagicMock(side_effect=RuntimeError("fail"))
        mock_app.run_default_pipeline.return_value = None
        MockBuilder.return_value.build.return_value = mock_app
        result = runner.invoke(
            cli_main.app,
            ["run", str(fake_input_file), "-c", str(fake_config_file)]
        )
        assert result.exit_code == 5
        assert "流水线执行过程中发生未捕获的严重错误" in result.output

# ====== init命令测试 ======
def test_init_command():
    result = runner.invoke(cli_main.app, ["init"])
    assert result.exit_code == 126
    assert "尚未实现" in result.output

# ====== show_config命令测试 ======
def test_show_config_command():
    result = runner.invoke(cli_main.app, ["show-config"])
    assert result.exit_code == 126
    assert "尚未实现" in result.output

# ====== main函数测试 ======
def test_main_entry(monkeypatch):
    called = {}
    def fake_app():
        called["ok"] = True
    monkeypatch.setattr(cli_main, "app", fake_app)
    cli_main.main()
    assert called["ok"] 