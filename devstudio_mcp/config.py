"""
DevStudio MCP Configuration Management
Copyright (C) 2024 Nihit Gupta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For commercial licensing options, contact: nihitgupta.ng@outlook.com
"""

import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration settings."""

    name: str = "devstudio-mcp"
    version: str = "1.0.0"
    port: int = Field(3000, ge=1000, le=65535)
    host: str = "localhost"
    debug: bool = False

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host configuration."""
        if not v.strip():
            raise ValueError("Host cannot be empty")
        return v.strip()


class RecordingConfig(BaseModel):
    """Recording configuration settings."""

    output_dir: str = str(Path.home() / "devstudio" / "recordings")
    max_duration: int = Field(3600, ge=30, le=7200)  # 30 seconds to 2 hours
    formats: List[str] = ["mp4", "webm"]
    quality: Literal["low", "medium", "high"] = "medium"
    fps: int = Field(30, ge=10, le=60)
    audio_enabled: bool = True

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate and create output directory."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except Exception as e:
            raise ValueError(f"Cannot create output directory {v}: {e}")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v: List[str]) -> List[str]:
        """Validate recording formats."""
        supported_formats = {"mp4", "webm", "avi", "mov"}
        invalid_formats = set(v) - supported_formats
        if invalid_formats:
            raise ValueError(f"Unsupported formats: {invalid_formats}")
        return v


class StorageConfig(BaseModel):
    """Storage configuration settings."""

    type: Literal["local", "s3", "gcs"] = "local"
    bucket: Optional[str] = None
    region: Optional[str] = None
    max_file_size: int = Field(100 * 1024 * 1024, ge=1024)  # 100MB default, min 1KB

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, v: Optional[str], info) -> Optional[str]:
        """Validate bucket name for cloud storage."""
        values = info.data if info.data else {}
        storage_type = values.get("type")
        if storage_type in ["s3", "gcs"] and not v:
            raise ValueError(f"Bucket name required for {storage_type} storage")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    file_enabled: bool = True
    file_path: str = "./logs/devstudio_mcp.log"
    max_file_size: int = Field(10 * 1024 * 1024, ge=1024)  # 10MB
    backup_count: int = Field(5, ge=1, le=20)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate and create log directory."""
        path = Path(v)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except Exception as e:
            raise ValueError(f"Cannot create log directory for {v}: {e}")


class Settings(BaseSettings):
    """Main application settings (Phase 1: Recording Only)."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Phase 2/3: AI configuration (optional for future features)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables (Phase 1: Recording Only)."""
        return cls(
            server=ServerConfig(
                name=os.getenv("SERVER_NAME", "devstudio-mcp"),
                version=os.getenv("SERVER_VERSION", "1.0.0"),
                port=int(os.getenv("PORT", "3000")),
                host=os.getenv("HOST", "localhost"),
                debug=os.getenv("DEBUG", "false").lower() == "true",
            ),
            recording=RecordingConfig(
                output_dir=os.getenv("RECORDING_OUTPUT_DIR", str(Path.home() / "devstudio" / "recordings")),
                max_duration=int(os.getenv("MAX_RECORDING_DURATION", "3600")),
                formats=os.getenv("RECORDING_FORMATS", "mp4,webm").split(","),
                quality=os.getenv("RECORDING_QUALITY", "medium"),
                fps=int(os.getenv("RECORDING_FPS", "30")),
                audio_enabled=os.getenv("RECORDING_AUDIO", "true").lower() == "true",
            ),
            storage=StorageConfig(
                type=os.getenv("STORAGE_TYPE", "local"),
                bucket=os.getenv("STORAGE_BUCKET"),
                region=os.getenv("STORAGE_REGION"),
                max_file_size=int(os.getenv("STORAGE_MAX_FILE_SIZE", str(100 * 1024 * 1024))),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "json"),
                file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
                file_path=os.getenv("LOG_FILE_PATH", "./logs/devstudio_mcp.log"),
                max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
                backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            ),
        )

    def validate_configuration(self) -> None:
        """Validate the complete configuration (Phase 1: Recording Only)."""
        # Phase 1: No AI provider validation needed (recording only)

        # Validate storage configuration
        if self.storage.type != "local":
            if not self.storage.bucket:
                raise ValueError(f"Bucket required for {self.storage.type} storage")
            if not self.storage.region:
                raise ValueError(f"Region required for {self.storage.type} storage")

        # Validate paths exist and are writable
        try:
            Path(self.recording.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.logging.file_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create required directories: {e}")


# Global configuration instance
try:
    config = Settings.from_env()
    config.validate_configuration()
except Exception as e:
    # Fallback to basic configuration for development
    print(f"Warning: Configuration validation failed: {e}")
    config = Settings()
