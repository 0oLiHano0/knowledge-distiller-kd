# kd_tool/core/cli_main.py (v4.8.1 - Context 修复与错误处理增强版)
# -*- coding: utf-8 -*-

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
import traceback # <-- [指令] 导入 traceback 用于在 Logger 不可用时打印堆栈

# 导入核心构建器和错误
from .application_builder import ApplicationBuilder
from .errors import KDToolError # <-- [指令] 导入 KDToolError
# [指令] 不再需要导入 AppConfig 或 load_config，这些由 Builder 处理

# ==============================================================================
# Typer 应用实例
# ==============================================================================
# [规范] Typer 应用实例定义保持不变。
app = typer.Typer(
    name="kd_tool",
    help="""
    KD_Tool (Knowledge Distiller) v4.0 - 本地化运行的源信息治理工具。
    """,
    no_args_is_help=True
)

# ==============================================================================
# 默认配置文件路径
# ==============================================================================
# [规范] 默认配置路径定义保持不变。
DEFAULT_CONFIG_FILENAME = "kd_config.yaml"
DEFAULT_CONFIG_PATHS = [
    Path.cwd() / DEFAULT_CONFIG_FILENAME,
    Path.home() / ".kd_tool" / DEFAULT_CONFIG_FILENAME,
]

# ==============================================================================
# 'run' 命令
# ==============================================================================
@app.command(help="运行默认的知识蒸馏处理流水线。")
def run(
    input_paths: List[Path] = typer.Argument(
        ...,
        help="要处理的一个或多个输入文件或目录路径。",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True # <-- [指令] 建议 Typer 自动解析为绝对路径
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help=f"指定要使用的配置文件路径。 (默认查找: {', '.join(map(str, DEFAULT_CONFIG_PATHS))})",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True # <-- [指令] 建议 Typer 自动解析为绝对路径
    )
):
    """
    **[指令]** `kd_tool` 的主执行命令。**必须** 严格执行以下步骤：
    1.  定位配置文件。
    2.  实例化并构建 `Application`，处理初始化期间的错误。
    3.  调用 `Application.run_default_pipeline`。
    4.  捕获运行期间的错误，并提供用户反馈和正确的退出代码。
    """

    typer.echo("🚀 欢迎使用 KD_Tool v4.0！")
    typer.echo("-" * 30)

    # --- [指令] 1. 定位配置文件 ---
    actual_config_path: Optional[Path] = config_file
    if actual_config_path is None:
        typer.echo(f"🔍 未指定配置文件，尝试在默认位置查找 '{DEFAULT_CONFIG_FILENAME}'...")
        for path in DEFAULT_CONFIG_PATHS:
            if path.is_file():
                actual_config_path = path
                typer.secho(f"✅ 在 {actual_config_path} 找到配置文件。", fg=typer.colors.GREEN)
                break

    if actual_config_path is None:
        typer.secho(
            f"❌ 错误: 无法找到配置文件 '{DEFAULT_CONFIG_FILENAME}'。请使用 -c 或 --config 参数指定，或在默认位置创建。",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(code=1)

    # --- [指令] 2 & 3. 初始化和构建 Application ---
    application = None # 先声明
    logger = None      # 先声明
    try:
        typer.echo(f"🔧 使用配置文件 '{actual_config_path}' 初始化应用程序...")
        builder = ApplicationBuilder(str(actual_config_path))
        typer.echo("🏗️ 正在构建应用程序核心组件...")
        application = builder.build()
        logger = application.logger # [指令] 必须 在构建成功后获取 Logger
        logger.info("⚡️ 应用程序构建完成，准备启动流水线...")

    except KDToolError as e:
        # [指令] 必须 捕获我们自己的初始化/构建错误。
        typer.secho(f"❌ 严重错误: 应用程序构建失败: {e}", fg=typer.colors.RED, err=True)
        if e.original_exception:
            # 如果 Logger 未成功初始化，我们无法用它记录，但可以打印原始堆栈。
            typer.secho("--- 原始错误详情 ---", fg=typer.colors.YELLOW, err=True)
            traceback.print_exception(type(e.original_exception), e.original_exception, e.original_exception.__traceback__, file=sys.stderr)
            typer.secho("----------------------", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2) # [指令] 必须 以错误码退出
    except Exception as e:
        # [指令] 必须 捕获所有其他意外的初始化/构建错误。
        typer.secho(f"❌ 严重错误: 应用程序构建时发生未知错误: {e}", fg=typer.colors.RED, err=True)
        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=3) # [指令] 必须 以错误码退出

    # --- [指令] 4. 运行流水线 ---
    try:
        # [指令] 必须 将 Path 对象转换为字符串列表。
        str_input_paths = [str(p) for p in input_paths]
        application.run_default_pipeline(str_input_paths)

        # [指令] 如果 run_default_pipeline 成功执行 *且未抛出异常*，则表示没有 *严重* 错误。
        #        Application 内部已经记录了成功日志。我们在此处打印最终成功消息并正常退出。
        typer.secho("🎉 流水线执行成功完成！详情请查看日志。", fg=typer.colors.BRIGHT_GREEN)
        raise typer.Exit(code=0) # [指令] 必须 以成功码 0 退出。

    except KDToolError as e:
        # [指令] 必须 捕获 Application.run 抛出的 KDToolError。
        #        这通常表示流水线 *完成* 但 *包含错误*，或者发生了受控的严重错误。
        #        Logger 已经在 Application 内部记录了细节。
        typer.secho(f"🔶 警告: 流水线执行完成，但存在问题: {e}", fg=typer.colors.YELLOW, err=True)
        typer.secho("   请仔细检查日志文件获取详细错误信息。", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=4) # [指令] 必须 以表示“完成但有错”的错误码退出。
    except Exception as e:
        # [指令] 必须 捕获 Application.run 抛出的任何其他 *未预料* 的严重错误。
        #        Logger 可能已经记录，但我们在这里提供最终的 CLI 反馈。
        typer.secho(f"❌ 严重错误: 流水线执行过程中发生未捕获的严重错误: {e}", fg=typer.colors.RED, err=True)
        # 在这里，Logger 应该是可用的，所以它应该已经记录了堆栈。
        # 我们可以在这里再次打印以确保万无一失，但通常依赖 Logger。
        typer.secho("   请立即检查日志文件获取详细的堆栈跟踪信息。", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=5) # [指令] 必须 以表示“严重失败”的错误码退出。

# ==============================================================================
# 其他命令 (占位符)
# ==============================================================================
# [规范] 占位符命令保持不变。
@app.command(help="（未来实现）初始化数据库和配置。")
def init():
    typer.echo("此功能尚未实现。")
    raise typer.Exit(code=126) # 使用不同的错误码

@app.command(help="（未来实现）显示当前的配置。")
def show_config(
     config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="指定配置文件路径。")
):
    typer.echo("此功能尚未实现。")
    raise typer.Exit(code=126)

# ==============================================================================
# 运行入口
# ==============================================================================
def main():
    """
    **[指令]** Python 脚本的主入口点。**必须** 仅用于调用 Typer 应用。
    """
    app()

if __name__ == "__main__":
    """
    **[指令]** 使得此脚本可以直接运行。
    """
    main()