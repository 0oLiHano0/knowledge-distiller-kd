"""
=================================================
c01.cli_main.py.md - KD_Tool 命令行入口 (v4.8.1)
=================================================

**模块功能**:

- **核心职责**: 提供 `kd_tool` 的命令行接口 (CLI)。
- **技术选型**: 使用 `Typer`。
- **v4.8.1 核心变更**:
    - **[指令] 必须** 正确调用 `ApplicationBuilder` 和 `Application.run_default_pipeline`。
    - **[指令] 必须** 实现健壮的顶层异常捕获，特别是 `KDToolError`。
    - **[指令] 必须** 使用 `typer.secho` 提供清晰的成功/错误反馈。
    - **[指令] 必须** 使用 `typer.Exit(code=...)` 设置明确的退出代码。
    - **[指令] 必须** 将 `Path` 对象转换为字符串传递给 `Application`。
    - **[指令] 必须** 在初始化失败时提供备用错误输出（因为 Logger 可能尚未可用）。

---
"""
import typer
from typing import List, Optional
from pathlib import Path
import sys
import traceback
from kd_tool.core.application_builder import ApplicationBuilder # kd_tool/core/application_builder.py 应用构建器
from kd_tool.core.errors import KDToolError # kd_tool/core/errors.py 错误
from kd_tool.logging.protocols import LoggerProtocol
from pydantic import BaseModel
app = typer.Typer(name='kd_tool', help=
    """
    KD_Tool (Knowledge Distiller) v4.0 - 本地化运行的源信息治理工具。
    """,
    no_args_is_help=True)
DEFAULT_CONFIG_FILENAME = 'kd_config.yaml'
DEFAULT_CONFIG_PATHS = [Path.cwd() / DEFAULT_CONFIG_FILENAME, Path.home() /
    '.kd_tool' / DEFAULT_CONFIG_FILENAME]


@app.command(help='运行默认的知识蒸馏处理流水线。')
def run(input_paths: List[Path]=typer.Argument(..., help=
    '要处理的一个或多个输入文件或目录路径。', exists=True, file_okay=True, dir_okay=True,
    readable=True, resolve_path=True), config_file: Optional[Path]=typer.
    Option(None, '--config', '-c', help=
    f"指定要使用的配置文件路径。 (默认查找: {', '.join(map(str, DEFAULT_CONFIG_PATHS))})",
    exists=True, file_okay=True, dir_okay=False, readable=True,
    resolve_path=True)):
    """
    WHY: 命令行入口。
    WHAT: 解析参数，调用ApplicationBuilder和Application。
    HOW: 只做参数解析和顶层异常处理。
    """
    typer.echo('🚀 欢迎使用 KD_Tool v4.0！')
    typer.echo('-' * 30)
    actual_config_path: Optional[Path] = config_file
    if actual_config_path is None:
        typer.echo(f"🔍 未指定配置文件，尝试在默认位置查找 '{DEFAULT_CONFIG_FILENAME}'...")
        for path in DEFAULT_CONFIG_PATHS:
            if path.is_file():
                actual_config_path = path
                typer.secho(f'✅ 在 {actual_config_path} 找到配置文件。', fg=typer.
                    colors.GREEN)
                break
    if actual_config_path is None:
        typer.secho(
            f"❌ 错误: 无法找到配置文件 '{DEFAULT_CONFIG_FILENAME}'。请使用 -c 或 --config 参数指定，或在默认位置创建。"
            , fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    logger: Optional[LoggerProtocol] = None
    try:
        typer.echo(f"🔧 使用配置文件 '{actual_config_path}' 初始化应用程序...")
        builder = ApplicationBuilder(str(actual_config_path))
        typer.echo('🏗️ 正在构建应用程序核心组件...')
        application = builder.build()
        logger = application.logger
        if logger:
            logger.info('⚡️ 应用程序构建完成，准备启动流水线...')
    except KDToolError as e:
        typer.secho(f'❌ 严重错误: Logger未能成功初始化。', fg=typer.colors.RED, err=True)
        if e.original_exception:
            typer.secho('--- 原始错误详情 ---', fg=typer.colors.YELLOW, err=True)
            traceback.print_exception(type(e.original_exception), e.
                original_exception, e.original_exception.__traceback__,
                file=sys.stderr)
            typer.secho('----------------------', fg=typer.colors.YELLOW,
                err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f'❌ 严重错误: 应用程序构建时发生未知错误: {e}', fg=typer.colors.RED, err
            =True)
        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=3)
    try:
        str_input_paths = [str(p) for p in input_paths]
        application.run_default_pipeline(str_input_paths)
        typer.secho('🎉 流水线执行成功完成！详情请查看日志。', fg=typer.colors.BRIGHT_GREEN)
        raise typer.Exit(code=0)
    except KDToolError as e:
        typer.secho(f'🔶 警告: 流水线执行完成，但存在问题: {e}', fg=typer.colors.YELLOW,
            err=True)
        typer.secho('   请仔细检查日志文件获取详细错误信息。', fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=4)
    except Exception as e:
        typer.secho(f'❌ 严重错误: 流水线执行过程中发生未捕获的严重错误: {e}', fg=typer.colors.RED,
            err=True)
        typer.secho('   请立即检查日志文件获取详细的堆栈跟踪信息。', fg=typer.colors.RED, err=True)
        raise typer.Exit(code=5)


@app.command(help='（未来实现）初始化数据库和配置。')
def init():
    typer.echo('此功能尚未实现。')
    raise typer.Exit(code=126)


@app.command(help='（未来实现）显示当前的配置。')
def show_config(config_file: Optional[Path]=typer.Option(None, '--config',
    '-c', help='指定配置文件路径。')):
    typer.echo('此功能尚未实现。')
    raise typer.Exit(code=126)


def main():
    """
    **[指令]** Python 脚本的主入口点。**必须** 仅用于调用 Typer 应用。
    """
    app()


if __name__ == '__main__':
    """
    **[指令]** 使得此脚本可以直接运行。
    """
    main()
