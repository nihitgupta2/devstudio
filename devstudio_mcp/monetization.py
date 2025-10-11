"""
Monetization and licensing system for DevStudio MCP.

Implements tier-based access control and usage tracking
following MCP marketplace best practices.
"""

import asyncio
import datetime
import os
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings
from .utils.exceptions import AuthenticationError, ValidationError, handle_mcp_error
from .utils.logger import setup_logger

logger = setup_logger()


class SubscriptionTier(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class FeatureAccess(BaseModel):
    """Feature access configuration per tier."""
    # Cloud features
    recordings_per_month: int
    ai_transcription: bool
    premium_models: bool
    team_features: bool
    custom_branding: bool
    priority_support: bool
    api_rate_limit: int  # requests per hour

    # Local features (can be monetized too!)
    max_recording_duration: int  # minutes
    max_resolution: str  # "720p", "1080p", "4K"
    multi_monitor_capture: bool
    audio_quality: str  # "basic", "professional"
    advanced_export_formats: bool
    batch_processing: bool
    local_ai_models: bool  # Local whisper, etc.
    watermark_removal: bool


class UserLicense(BaseModel):
    """User license and subscription information."""
    user_id: str
    email: str
    tier: SubscriptionTier
    expires_at: Optional[datetime.datetime] = None
    usage_this_month: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class LicenseManager:
    """Manages user licensing and feature access."""

    def __init__(self, settings: Settings):
        """Initialize license manager."""
        self.settings = settings
        self.logger = logger

        # Define tier capabilities
        self.tier_features = {
            SubscriptionTier.FREE: FeatureAccess(
                recordings_per_month=10,
                ai_transcription=False,  # Local only
                premium_models=False,
                team_features=False,
                custom_branding=False,
                priority_support=False,
                api_rate_limit=100,
                max_recording_duration=5,  # 5 minutes
                max_resolution="720p",
                multi_monitor_capture=False,
                audio_quality="basic",
                advanced_export_formats=False,
                batch_processing=False,
                local_ai_models=False,
                watermark_removal=False
            ),
            SubscriptionTier.PRO: FeatureAccess(
                recordings_per_month=-1,  # Unlimited
                ai_transcription=True,
                premium_models=True,
                team_features=False,
                custom_branding=True,
                priority_support=True,
                api_rate_limit=1000,
                max_recording_duration=60,  # 1 hour
                max_resolution="1080p",
                multi_monitor_capture=True,
                audio_quality="professional",
                advanced_export_formats=True,
                batch_processing=True,
                local_ai_models=True,
                watermark_removal=True
            ),
            SubscriptionTier.TEAM: FeatureAccess(
                recordings_per_month=-1,
                ai_transcription=True,
                premium_models=True,
                team_features=True,
                custom_branding=True,
                priority_support=True,
                api_rate_limit=5000,
                max_recording_duration=120,  # 2 hours
                max_resolution="1080p",
                multi_monitor_capture=True,
                audio_quality="professional",
                advanced_export_formats=True,
                batch_processing=True,
                local_ai_models=True,
                watermark_removal=True
            ),
            SubscriptionTier.ENTERPRISE: FeatureAccess(
                recordings_per_month=-1,
                ai_transcription=True,
                premium_models=True,
                team_features=True,
                custom_branding=True,
                priority_support=True,
                api_rate_limit=10000,
                max_recording_duration=300,  # 5 hours
                max_resolution="4K",
                multi_monitor_capture=True,
                audio_quality="professional",
                advanced_export_formats=True,
                batch_processing=True,
                local_ai_models=True,
                watermark_removal=True
            )
        }

        # Load user license (in production, this would be from a database)
        self.current_license = self._load_user_license()

    def _load_user_license(self) -> UserLicense:
        """Load user license from local storage or environment."""
        # Check for license key in environment
        license_key = os.getenv("DEVSTUDIO_LICENSE_KEY")

        if not license_key:
            # Default to free tier for MVP
            return UserLicense(
                user_id="local_user",
                email="user@local.dev",
                tier=SubscriptionTier.FREE
            )

        # In production, validate license key with licensing server
        # For MVP, parse basic license info
        try:
            # Simple license format: tier_email_timestamp
            parts = license_key.split("_")
            if len(parts) >= 2:
                tier = SubscriptionTier(parts[0].lower())
                email = parts[1]

                return UserLicense(
                    user_id=email,
                    email=email,
                    tier=tier,
                    expires_at=datetime.datetime.now() + datetime.timedelta(days=30)
                )
        except Exception as e:
            self.logger.warning(f"Invalid license key: {e}")

        # Fallback to free tier
        return UserLicense(
            user_id="local_user",
            email="user@local.dev",
            tier=SubscriptionTier.FREE
        )

    def check_feature_access(self, feature: str) -> bool:
        """Check if current license has access to feature."""
        features = self.tier_features[self.current_license.tier]

        if feature == "ai_transcription":
            return features.ai_transcription
        elif feature == "premium_models":
            return features.premium_models
        elif feature == "team_features":
            return features.team_features
        elif feature == "custom_branding":
            return features.custom_branding
        elif feature == "priority_support":
            return features.priority_support

        return False

    def check_usage_limit(self, operation: str) -> bool:
        """Check if user has remaining usage quota."""
        features = self.tier_features[self.current_license.tier]

        if operation == "recording":
            monthly_limit = features.recordings_per_month
            if monthly_limit == -1:  # Unlimited
                return True

            current_usage = self.current_license.usage_this_month.get("recordings", 0)
            return current_usage < monthly_limit

        return True

    def track_usage(self, operation: str) -> None:
        """Track usage for billing and limits."""
        if operation not in self.current_license.usage_this_month:
            self.current_license.usage_this_month[operation] = 0

        self.current_license.usage_this_month[operation] += 1

        # Reset monthly usage if new month
        now = datetime.datetime.now()
        if (now - self.current_license.created_at).days >= 30:
            self.current_license.usage_this_month = {}
            self.current_license.created_at = now

    def get_tier_info(self) -> Dict[str, Any]:
        """Get current tier information and limits."""
        features = self.tier_features[self.current_license.tier]

        return {
            "tier": self.current_license.tier.value,
            "email": self.current_license.email,
            "expires_at": self.current_license.expires_at.isoformat() if self.current_license.expires_at else None,
            "features": {
                "recordings_per_month": features.recordings_per_month,
                "ai_transcription": features.ai_transcription,
                "premium_models": features.premium_models,
                "team_features": features.team_features,
                "custom_branding": features.custom_branding,
                "priority_support": features.priority_support,
                "api_rate_limit": features.api_rate_limit
            },
            "usage_this_month": self.current_license.usage_this_month,
            "upgrade_url": "https://devstudio.com/upgrade"
        }

    def require_feature(self, feature: str) -> None:
        """Require specific feature access, raise error if not available."""
        if not self.check_feature_access(feature):
            tier_name = self.current_license.tier.value.title()
            raise AuthenticationError(
                f"Feature '{feature}' requires Pro tier or higher. Current tier: {tier_name}. "
                f"Upgrade at https://devstudio.com/upgrade"
            )

    def require_usage_quota(self, operation: str) -> None:
        """Require usage quota, raise error if exceeded."""
        if not self.check_usage_limit(operation):
            features = self.tier_features[self.current_license.tier]
            limit = features.recordings_per_month

            raise ValidationError(
                f"Monthly {operation} limit exceeded ({limit}). "
                f"Upgrade to Pro for unlimited usage: https://devstudio.com/upgrade"
            )


# Decorators for feature gating
def require_pro_tier(func):
    """Decorator to require Pro tier or higher."""
    async def wrapper():
        if hasattr(func, '__self__') and hasattr(func.__self__, 'license_manager'):
            license_manager = func.__self__.license_manager
        else:
            # Fallback - get global license manager
            from . import license_manager as global_license_manager
            license_manager = global_license_manager

        if license_manager.current_license.tier == SubscriptionTier.FREE:
            raise AuthenticationError(
                "This feature requires Pro tier or higher. "
                "Upgrade at https://devstudio.com/upgrade"
            )

        return await func()
    return wrapper


def track_usage(operation: str):
    """Decorator to track usage for an operation."""
    def decorator(func):
        async def wrapper():
            # Track usage before execution
            if hasattr(func, '__self__') and hasattr(func.__self__, 'license_manager'):
                license_manager = func.__self__.license_manager
                license_manager.require_usage_quota(operation)
                license_manager.track_usage(operation)

            return await func()
        return wrapper
        return decorator
    return decorator


# Global license manager instance
license_manager = None


def get_monetization_tools(settings: Settings) -> Dict[str, Any]:
    """Get monetization and licensing tools for MCP registration."""
    global license_manager
    license_manager = LicenseManager(settings)

    @handle_mcp_error
    async def get_license_info() -> Dict[str, Any]:
        """
        Get current license and subscription information.

        Returns:
            License tier, features, usage, and upgrade information
        """
        return license_manager.get_tier_info()

    @handle_mcp_error
    async def check_feature_access(feature: str) -> Dict[str, Any]:
        """
        Check if current license has access to a specific feature.

        Args:
            feature: Feature name to check

        Returns:
            Access status and upgrade information if needed
        """
        has_access = license_manager.check_feature_access(feature)

        return {
            "feature": feature,
            "has_access": has_access,
            "current_tier": license_manager.current_license.tier.value,
            "upgrade_url": "https://devstudio.com/upgrade" if not has_access else None
        }

    @handle_mcp_error
    async def get_usage_stats() -> Dict[str, Any]:
        """
        Get current usage statistics and limits.

        Returns:
            Usage data and remaining quotas
        """
        tier_info = license_manager.get_tier_info()

        return {
            "current_usage": tier_info["usage_this_month"],
            "limits": {
                "recordings_per_month": tier_info["features"]["recordings_per_month"]
            },
            "tier": tier_info["tier"]
        }

    return {
        "get_license_info": get_license_info,
        "check_feature_access": check_feature_access,
        "get_usage_stats": get_usage_stats
    }
