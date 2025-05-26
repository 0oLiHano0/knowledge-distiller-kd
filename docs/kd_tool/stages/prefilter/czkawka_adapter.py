import subprocess
import json
import platform
from pathlib import Path
from typing import List
from loguru import Logger
from kd_tool.stages.prefilter.settings_models import CzkawkaSettings
from kd_tool.stages.prefilter.adapter_interface import CzkawkaAdapterInterface
from kd_tool.stages.prefilter.dtos import CzkawkaScanOutputDTO, CzkawkaDuplicateResultDTO
from kd_tool.stages.prefilter.errors import CzkawkaExecutionError, CzkawkaParseError
VENDOR_PATH = Path(__file__).parent.parent.parent.parent / 'vendor' / 'czkawka'
PLATFORM_MAP = {'linux': 'czkawka_cli_linux', 'windows': 'czkawka_cli.exe',
    'darwin': 'czkawka_cli_macos'}


class CzkawkaAdapter(CzkawkaAdapterInterface):
    """与 Czkawka CLI 工具交互的具体实现。"""

    def __init__(self, settings: CzkawkaSettings, logger: Logger):
        self._logger = logger.bind(component='CzkawkaAdapter')
        self._settings = settings
        self._executable_path = self._resolve_executable_path()
        self._logger.info(
            f'CzkawkaAdapter initialized. Using executable: {self._executable_path}'
            )

    def _resolve_executable_path(self) ->Path:
        """
        确定 Czkawka CLI 的可执行文件路径。
        (简化版，实际需要更复杂的平台/架构检测和 PATH 查找)
        """
        self._logger.debug('Resolving Czkawka executable path...')
        configured_path = self._settings.executable_path
        if configured_path.is_file() and configured_path.stat().st_mode & 73:
            self._logger.info(
                f'Using configured executable path: {configured_path}')
            return configured_path
        else:
            self._logger.error(
                f"Configured Czkawka path '{configured_path}' is not a valid executable file."
                )
            raise CzkawkaExecutionError(command=str(configured_path),
                return_code=-1, error_output=
                f"Path '{configured_path}' not found or not executable.")

    def _build_command(self) ->List[str]:
        """根据配置构建 Czkawka CLI 命令。"""
        cmd = [str(self._executable_path)]
        cmd.extend(['-d'] + [str(p) for p in self._settings.
            directories_to_scan])
        cmd.append(self._settings.scan_mode)
        if self._settings.allowed_extensions:
            extensions_str = ','.join(self._settings.allowed_extensions)
            cmd.extend(['-e', extensions_str])
            self._logger.debug(f'Filtering extensions: {extensions_str}')
        if self._settings.min_file_size is not None:
            cmd.extend(['-m', str(self._settings.min_file_size)])
            self._logger.debug(
                f'Setting min file size: {self._settings.min_file_size}')
        cmd.append('--output-json')
        cmd.extend(self._settings.extra_args)
        self._logger.debug(f"Built Czkawka command: {' '.join(cmd)}")
        return cmd

    def scan_and_find_duplicates(self) ->CzkawkaScanOutputDTO:
        """运行 Czkawka 并解析输出。"""
        command = self._build_command()
        try:
            self._logger.info(f"Executing Czkawka: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=
                True, check=False, encoding='utf-8')
            if process.returncode != 0:
                self._logger.error(
                    f'Czkawka execution failed with code {process.returncode}')
                self._logger.error(f'Stderr: {process.stderr}')
                raise CzkawkaExecutionError(command=' '.join(command),
                    return_code=process.returncode, error_output=process.stderr
                    )
            self._logger.success('Czkawka execution finished successfully.')
            raw_json_output = process.stdout
            return self._parse_czkawka_output(raw_json_output)
        except FileNotFoundError:
            self._logger.exception(
                f'Czkawka executable not found at: {self._executable_path}')
            raise CzkawkaExecutionError(command=' '.join(command),
                return_code=-1, error_output='Executable not found.')
        except Exception as e:
            self._logger.exception(
                'An unexpected error occurred during Czkawka execution.')
            raise CzkawkaExecutionError(command=' '.join(command),
                return_code=-1, error_output=str(e)) from e

    def _parse_czkawka_output(self, json_output: str) ->CzkawkaScanOutputDTO:
        """
        解析 Czkawka 的 JSON 输出。
        【这是设计的核心难点和假设点】
        """
        self._logger.debug('Parsing Czkawka JSON output...')
        try:
            data = json.loads(json_output)
            if 'duplicates' not in data:
                raise CzkawkaParseError("JSON output missing 'duplicates' key."
                    , json_output)
            duplicate_groups = [CzkawkaDuplicateResultDTO(original_file=
                Path(group['original']), duplicates=[Path(d) for d in group
                ['duplicates']], size_bytes=group['size']) for group in
                data['duplicates']]
            if 'scanned_files' in data:
                all_files = [Path(f) for f in data['scanned_files']]
            else:
                self._logger.warning(
                    "Czkawka output did not contain 'scanned_files'. Attempting to infer..."
                    )
                all_files_set = set()
                for group in duplicate_groups:
                    all_files_set.add(group.original_file)
                    all_files_set.update(group.duplicates)
                if not all_files_set:
                    raise CzkawkaParseError(
                        'Could not determine any scanned files from Czkawka output.'
                        , json_output)
                all_files = list(all_files_set)
            self._logger.success(
                f'Successfully parsed {len(all_files)} files and {len(duplicate_groups)} groups.'
                )
            return CzkawkaScanOutputDTO(all_scanned_files=all_files,
                duplicate_groups=duplicate_groups)
        except json.JSONDecodeError as e:
            self._logger.error(f'Failed to decode Czkawka JSON output: {e}')
            raise CzkawkaParseError(f'JSON decode error: {e}', json_output
                ) from e
        except Exception as e:
            self._logger.exception(
                'An unexpected error occurred during Czkawka output parsing.')
            raise CzkawkaParseError(f'Unexpected parsing error: {e}',
                json_output) from e
