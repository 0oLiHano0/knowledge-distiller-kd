# knowledge_distiller_kd/cli.py
"""
Command Line Interface (CLI) entry point for the Knowledge Distiller tool.

This script handles command-line argument parsing, initializes the core components
(Engine, Storage, UI), and starts the interactive user interface.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# 使用相对导入来引用同包内的模块
from .core import constants
from .core.engine import KnowledgeDistillerEngine
from .core.error_handler import ConfigurationError, handle_error
from .core.utils import logger, setup_logger # 使用 utils 中配置好的 logger
from .storage.file_storage import FileStorage
from .ui.cli_interface import CliInterface

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the KD Tool.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="知识蒸馏工具 KD Tool: 查找并处理 Markdown 文件中的重复内容块。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 启动交互模式 (推荐)
  python -m knowledge_distiller_kd.cli

  # 启动时指定输入目录
  python -m knowledge_distiller_kd.cli -i ./my_markdown_files

  # 指定所有路径并设置阈值
  python -m knowledge_distiller_kd.cli -i ./input -o ./output -d ./dec/decisions.json -t 0.75

  # 跳过语义分析
  python -m knowledge_distiller_kd.cli -i ./input --skip-semantic

  # 设置日志级别为 DEBUG
  python -m knowledge_distiller_kd.cli -i ./input --log-level DEBUG
"""
    )

    parser.add_argument(
        "-i", "--input-dir", type=str, default=None,
        help="输入文件夹路径 (包含 Markdown 文件)。如果未提供，将在交互模式中手动设置。"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=constants.DEFAULT_OUTPUT_DIR,
        help=f"保存去重后文件的输出文件夹路径 (默认: '{constants.DEFAULT_OUTPUT_DIR}')"
    )
    parser.add_argument(
        "-d", "--decision-file", type=str, default=constants.DEFAULT_DECISION_FILE,
        help=f"加载和保存决策的 JSON 文件路径 (默认: '{constants.DEFAULT_DECISION_FILE}')"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=constants.DEFAULT_SIMILARITY_THRESHOLD,
        help=f"语义相似度阈值 (0.0-1.0, 默认: {constants.DEFAULT_SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="跳过语义相似度分析 (仅执行 MD5 分析)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="设置日志记录级别 (默认: INFO)"
    )

    args = parser.parse_args()

    # 验证阈值范围
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("相似度阈值必须在 0.0 到 1.0 之间。")

    return args

def main() -> None:
    """
    Main function: Parses arguments, initializes components, and runs the CLI.
    """
    args = parse_args()

    # 设置日志级别
    log_level = constants.LOG_LEVEL_MAP.get(args.log_level.upper(), logging.INFO)
    try:
        setup_logger(log_level) # 确保调用方式正确
    except Exception as log_e:
        print(f"错误：无法配置日志记录器 - {log_e}", file=sys.stderr)

    logger.info("KD Tool started.")
    logger.debug(f"Parsed arguments: {args}")

    try:
        # 1. 初始化存储层 (Storage)
        storage = FileStorage()
        logger.info("FileStorage initialized.")

        # 2. 初始化核心引擎 (Engine)
        # 直接将参数传递给引擎的构造函数
        logger.info("Initializing KnowledgeDistillerEngine...")
        engine = KnowledgeDistillerEngine(
            storage=storage,
            input_dir=args.input_dir,         # 直接传递路径字符串或 None
            output_dir=args.output_dir,       # 直接传递路径字符串
            decision_file=args.decision_file, # 直接传递路径字符串
            skip_semantic=args.skip_semantic,
            similarity_threshold=args.threshold
        )
        logger.info("KnowledgeDistillerEngine initialized successfully.")

        # 3. 初始化用户界面 (UI)
        # 将引擎实例传递给 UI
        ui = CliInterface(engine=engine)
        logger.info("CliInterface initialized.")

        # 4. (可选) 如果通过命令行提供了输入目录，则预先设置
        # 注意：Engine 的 __init__ 现在也会尝试处理初始 input_dir，
        # 但为了明确流程和可能的日志/错误处理，保留这里的显式调用可能更好。
        # 或者可以依赖 __init__ 的处理，并移除这里的 set_input_dir 调用。
        # 我们暂时保留它以保持与之前逻辑的一致性。
        if args.input_dir and engine.input_dir is None: # 检查引擎初始化时是否成功设置
             logger.warning(f"Engine failed to initialize with input_dir '{args.input_dir}'. Attempting to set again.")
             success = engine.set_input_dir(args.input_dir) # Engine 内部处理 Path()
             if not success:
                  logger.error(f"Failed to set initial input directory '{args.input_dir}' provided via argument after init failure.")
                  # 决定是否退出或继续
        elif args.input_dir and engine.input_dir:
            logger.info(f"Input directory '{args.input_dir}' provided via arguments and successfully set during engine initialization.")


        # 5. 运行交互式 UI
        logger.info("Starting interactive UI loop...")
        ui.run() # CliInterface 的主循环

    except ConfigurationError as e:
        logger.critical(f"Tool initialization or configuration failed: {e}", exc_info=False)
        print(f"\n[致命错误] 工具配置失败: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[*] 用户中断操作，程序退出。", file=sys.stderr)
        logger.warning("Operation interrupted by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        # 捕获其他未预料的全局错误
        logger.critical(f"An unexpected error occurred during execution: {e}", exc_info=True)
        print("\n[致命错误] 程序运行过程中发生未预期错误:", file=sys.stderr)
        traceback.print_exc() # 打印详细的错误堆栈信息
        sys.exit(1)

    logger.info("KD Tool finished gracefully.")
    sys.exit(0)

if __name__ == "__main__":
    main()