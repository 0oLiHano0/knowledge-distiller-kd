from pydantic import BaseModel, Field

class LoggingSettingsDTO(BaseModel):
    level: str = Field(default="INFO", description="日志级别")
    # 其它参数... 