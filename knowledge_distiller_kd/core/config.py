"""
配置管理模块。

该模块使用Pydantic实现配置的加载、验证和类型安全。配置可以从环境变量或.env文件加载，
并提供合理的默认值。

主要组件:
- StorageConfig: 存储相关配置（数据库URL等）
- LoggingConfig: 日志相关配置（日志文件路径、级别等）
- EngineConfig: 引擎相关配置（相似度阈值等）
- AppConfig: 聚合以上所有配置的主配置类
"""

from typing import Optional
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class StorageConfig(BaseModel):
    """存储相关配置"""
    database_url: str = "sqlite+aiosqlite:///./instance/kd_default.sqlite"
    db_dir: str = "data"
    db_name: str = "kd_tool.db"

class LoggingConfig(BaseModel):
    """日志相关配置"""
    log_file_path: str = "logs/kd_tool.log"
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"
    log_serialize_json: bool = True
    log_dir: str = "logs"
    log_name: str = "kd_tool.log"

    @field_validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别是否有效"""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是以下之一: {", ".join(valid_levels)}')
        return v.upper()

class EngineConfig(BaseModel):
    """引擎相关配置"""
    similarity_threshold: float = 0.85
    czkawka_path: Optional[str] = None
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    cache_dir: str = "cache"
    cache_base_dir: str = ".kd_cache"

class AppConfig(BaseSettings):
    """应用主配置，聚合所有子配置"""
    # Storage配置
    database_url: str = "sqlite+aiosqlite:///./instance/kd_default.sqlite"
    db_dir: str = "data"
    db_name: str = "kd_tool.db"
    
    # Logging配置
    log_file_path: str = "logs/kd_tool.log"
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"
    log_serialize_json: bool = True
    log_dir: str = "logs"
    log_name: str = "kd_tool.log"
    
    # Engine配置
    similarity_threshold: float = 0.85
    czkawka_path: Optional[str] = None
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    cache_dir: str = "cache"
    cache_base_dir: str = ".kd_cache"

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='',
        extra='ignore'
    )

    @property
    def storage(self) -> StorageConfig:
        """获取存储配置"""
        return StorageConfig(
            database_url=self.database_url,
            db_dir=self.db_dir,
            db_name=self.db_name
        )

    @property
    def logging(self) -> LoggingConfig:
        """获取日志配置"""
        return LoggingConfig(
            log_file_path=self.log_file_path,
            log_level=self.log_level,
            log_rotation=self.log_rotation,
            log_retention=self.log_retention,
            log_serialize_json=self.log_serialize_json,
            log_dir=self.log_dir,
            log_name=self.log_name
        )

    @property
    def engine(self) -> EngineConfig:
        """获取引擎配置"""
        return EngineConfig(
            similarity_threshold=self.similarity_threshold,
            czkawka_path=self.czkawka_path,
            semantic_model=self.semantic_model,
            batch_size=self.batch_size,
            cache_dir=self.cache_dir,
            cache_base_dir=self.cache_base_dir
        )

    @field_validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别是否有效"""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是以下之一: {", ".join(valid_levels)}')
        return v.upper()

# 全局配置实例
_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """获取全局配置实例（单例模式）

    Returns:
        AppConfig: 配置实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance 