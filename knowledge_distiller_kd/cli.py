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
from .storage.orm_storage import ORMStorage
from .ui.cli_interface import CliInterface
from .core.factories import create_app_config, create_storage, create_logger, create_engine

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

  # 仅执行预过滤步骤
  python -m knowledge_distiller_kd.cli -i ./input --pre-filter

  # 跳过预过滤步骤
  python -m knowledge_distiller_kd.cli -i ./input --skip-prefilter

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
        "--pre-filter", action="store_true",
        help="仅执行预过滤阶段，打印统计后退出"
    )
    parser.add_argument(
        "--skip-prefilter", action="store_true",
        help="跳过预过滤阶段，直接进入后续分析流程"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="设置日志记录级别 (默认: INFO)"
    )
    
    # 添加非交互式模式参数
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="非交互式模式，自动运行分析而不启动交互式 UI"
    )

    args = parser.parse_args()

    # 验证阈值范围
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("相似度阈值必须在 0.0 到 1.0 之间。")
    
    # 验证预过滤参数
    if args.pre_filter and args.skip_prefilter:
        parser.error("参数冲突：不能同时使用 --pre-filter 和 --skip-prefilter。")

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
        # 1. 使用工厂创建依赖项
        config = create_app_config()
        storage = create_storage(config)
        logger_instance = create_logger(config)
        
        # 2. 使用工厂创建引擎
        engine = create_engine(storage, config, logger_instance)
        
        # 设置引擎的其他参数
        if args.input_dir:
            success = engine.set_input_dir(args.input_dir)
            if not success:
                logger.error(f"Failed to set initial input directory '{args.input_dir}' provided via argument.")
                print(f"[错误] 无法设置输入目录 '{args.input_dir}'", file=sys.stderr)
                sys.exit(1)
                
        # 更新引擎的其他配置
        if args.skip_semantic:
            engine.set_skip_semantic(True)
        if args.skip_prefilter:
            engine.skip_prefilter = True
        if args.threshold != constants.DEFAULT_SIMILARITY_THRESHOLD:
            engine.set_similarity_threshold(args.threshold)
        
        # 3. 参数冲突检测放在最前面（已经在parse_args中实现了，这里不需要重复检查）
        # 4. 处理 --pre-filter 参数（仅执行预过滤步骤）
        if args.pre_filter:
            if not args.input_dir:
                logger.error("使用 --pre-filter 时必须指定输入目录")
                print("错误：使用 --pre-filter 时必须指定输入目录 (-i/--input-dir)", file=sys.stderr)
                sys.exit(1)
            
            # 运行预过滤并打印统计信息
            total_files, unique_files, duplicate_groups = engine.run_prefilter_only()
            
            # 计算重复文件数量（每组中第一个文件不算重复，其余都算重复）
            filtered_count = 0
            for group in duplicate_groups:
                # 每组中除第一个文件外，其余都算作重复
                filtered_count += max(0, len(group) - 1)
                
            print(f"[Prefilter] Scanned {total_files} files, filtered {filtered_count} duplicates → {len(unique_files)} files remain.")
            sys.exit(0)  # 正常退出程序
        
        # 5. 处理 --non-interactive 参数（非交互式模式）
        if args.non_interactive:
            if not args.input_dir:
                logger.error("使用 --non-interactive 时必须指定输入目录")
                print("错误：使用 --non-interactive 时必须指定输入目录 (-i/--input-dir)", file=sys.stderr)
                sys.exit(1)
                
            # 自动运行分析
            print(f"\n[*] 非交互式模式：自动运行分析...")
            analysis_successful = engine.run_analysis()
            
            if analysis_successful:
                # 获取分析结果统计
                md5_duplicates = engine.get_md5_duplicates()
                md5_duplicate_pairs = sum(len(group) - 1 for group in md5_duplicates)
                print(f"MD5 duplicates found: {md5_duplicate_pairs} pairs")
                
                if not args.skip_semantic and engine._model_loaded_successfully():
                    semantic_duplicates = engine.get_semantic_duplicates()
                    print(f"Semantic duplicates found: {len(semantic_duplicates)} pairs")
                
                print(f"\n[*] 分析完成。")
            else:
                print(f"\n[错误] 分析失败。")
                sys.exit(1)
                
            sys.exit(0)  # 正常退出程序
        
        # 6. 初始化用户界面 (UI)
        # 将引擎实例传递给 UI
        ui = CliInterface(engine=engine)
        logger.info("CliInterface initialized.")
            
        # 7. 运行交互式 UI
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