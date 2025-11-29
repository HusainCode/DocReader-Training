"""
Model versioning system for ML training artifacts.

Provides automatic version management, S3 storage, and metadata tracking
for trained models with production-grade error handling.

Features:
- Auto-incrementing version numbers (v1, v2, v3, ...)
- Structured S3 storage: models/{model_name}/{version}/
- Metadata tracking (config, metrics, timestamp)
- Version discovery from S3
- Model download and restoration
- Comprehensive error handling and logging
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import torch
import boto3
from botocore.exceptions import ClientError

from .upload_s3 import S3Uploader
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelVersionManager:
    """
    Manages model versioning and S3 storage for ML training artifacts.

    Handles automatic version incrementing, metadata storage, and
    production-grade model lifecycle management.

    Example:
        >>> manager = ModelVersionManager(bucket_name="my-ml-models")
        >>> version_info = manager.save_model_to_s3(
        ...     model=my_model,
        ...     model_name="docreader_ocr",
        ...     metrics={"accuracy": 0.95, "loss": 0.05},
        ...     config={"lr": 0.001, "epochs": 50}
        ... )
        >>> print(f"Saved as {version_info['version']}")

        >>> loaded_model = manager.load_model_from_s3(
        ...     model_name="docreader_ocr",
        ...     version="v3"
        ... )
    """

    def __init__(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        model_format: str = "pt",  # "pt" for PyTorch, "onnx", "pkl", etc.
    ):
        """
        Initialize model version manager.

        Args:
            bucket_name: S3 bucket name for model storage
            region: AWS region (default: from AWS config)
            model_format: Model file format extension ("pt", "onnx", "pkl")
        """
        self.bucket_name = bucket_name
        self.model_format = model_format

        # Initialize S3 uploader
        self.uploader = S3Uploader(bucket_name=bucket_name, region=region)

        # Direct S3 client for listing operations
        self.s3_client = self.uploader.s3_client

        logger.info(f"ModelVersionManager initialized for bucket: {bucket_name}")

    def _list_versions(self, model_name: str) -> List[str]:
        """
        List all existing versions for a model in S3.

        Args:
            model_name: Name of the model

        Returns:
            List of version strings (e.g., ["v1", "v2", "v3"])
        """
        prefix = f"models/{model_name}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter="/"
            )

            # Extract version folders from CommonPrefixes
            versions = []
            if "CommonPrefixes" in response:
                for prefix_info in response["CommonPrefixes"]:
                    # Extract version from path like "models/model_name/v1/"
                    version_folder = prefix_info["Prefix"].rstrip("/").split("/")[-1]
                    if version_folder.startswith("v") and version_folder[1:].isdigit():
                        versions.append(version_folder)

            # Sort versions by number (v1, v2, v3, ...)
            versions.sort(key=lambda v: int(v[1:]))

            logger.debug(f"Found {len(versions)} versions for {model_name}: {versions}")
            return versions

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise ValueError(f"Bucket does not exist: {self.bucket_name}")
            else:
                logger.error(f"Failed to list versions: {e}")
                raise RuntimeError(f"Failed to list model versions: {e}")

    def _get_next_version(self, model_name: str) -> str:
        """
        Determine the next version number for a model.

        Args:
            model_name: Name of the model

        Returns:
            Next version string (e.g., "v1", "v2", "v3")
        """
        existing_versions = self._list_versions(model_name)

        if not existing_versions:
            next_version = "v1"
        else:
            # Get latest version number and increment
            latest_version = existing_versions[-1]
            latest_number = int(latest_version[1:])
            next_version = f"v{latest_number + 1}"

        logger.info(f"Next version for {model_name}: {next_version}")
        return next_version

    def _create_metadata(
        self,
        model_name: str,
        version: str,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for model version.

        Args:
            model_name: Name of the model
            version: Version string (e.g., "v1")
            metrics: Training metrics (loss, accuracy, etc.)
            config: Training configuration (hyperparameters)
            extra_info: Additional metadata

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "format": self.model_format,
            "metrics": metrics or {},
            "config": config or {},
        }

        if extra_info:
            metadata["extra"] = extra_info

        return metadata

    def save_model_to_s3(
        self,
        model: Any,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save model to S3 with automatic versioning and metadata.

        Uploads both the model file and a metadata JSON file containing
        training config, metrics, and timestamp.

        Args:
            model: Model object (e.g., PyTorch model, scikit-learn model)
            model_name: Name of the model (e.g., "docreader_ocr")
            metrics: Training metrics dictionary (e.g., {"accuracy": 0.95, "loss": 0.05})
            config: Training configuration (e.g., {"lr": 0.001, "epochs": 50})
            version: Optional explicit version (default: auto-increment)
            extra_info: Additional metadata to store

        Returns:
            Dictionary with version info:
                - version: Version string (e.g., "v3")
                - model_s3_uri: S3 URI for model file
                - metadata_s3_uri: S3 URI for metadata file
                - model_size_mb: Model file size in MB
                - upload_time_seconds: Total upload time

        Raises:
            ValueError: If model_name is invalid or model is None
            RuntimeError: If save or upload fails

        Example:
            >>> manager = ModelVersionManager(bucket_name="my-models")
            >>> info = manager.save_model_to_s3(
            ...     model=trained_model,
            ...     model_name="text_classifier",
            ...     metrics={"accuracy": 0.94, "f1": 0.92},
            ...     config={"lr": 0.001, "batch_size": 32}
            ... )
            >>> print(f"Model saved as {info['version']}")
        """
        if model is None:
            raise ValueError("Model cannot be None")

        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"Invalid model_name: {model_name}")

        # Determine version
        if version is None:
            version = self._get_next_version(model_name)
        else:
            # Validate explicit version format
            if not version.startswith("v") or not version[1:].isdigit():
                raise ValueError(f"Version must be in format 'v1', 'v2', etc. Got: {version}")

        logger.info(f"Saving {model_name} as {version}")

        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save model file
            model_filename = f"model.{self.model_format}"
            model_path = temp_path / model_filename

            try:
                if self.model_format == "pt":
                    # PyTorch model
                    if isinstance(model, torch.nn.Module):
                        torch.save(model.state_dict(), model_path)
                    else:
                        # Assume it's already a state dict or other serializable object
                        torch.save(model, model_path)
                elif self.model_format == "pkl":
                    # Pickle-based model (scikit-learn, etc.)
                    import pickle
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                else:
                    raise ValueError(f"Unsupported model format: {self.model_format}")

            except Exception as e:
                logger.error(f"Failed to save model locally: {e}")
                raise RuntimeError(f"Model serialization failed: {e}")

            # Get model file size
            model_size_bytes = model_path.stat().st_size
            model_size_mb = model_size_bytes / (1024 * 1024)

            # Create metadata
            metadata = self._create_metadata(
                model_name=model_name,
                version=version,
                metrics=metrics,
                config=config,
                extra_info=extra_info,
            )

            # Add model file info to metadata
            metadata["model_filename"] = model_filename
            metadata["model_size_bytes"] = model_size_bytes

            # Save metadata file
            metadata_filename = "metadata.json"
            metadata_path = temp_path / metadata_filename

            try:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to create metadata file: {e}")
                raise RuntimeError(f"Metadata creation failed: {e}")

            # Upload to S3
            s3_base_key = f"models/{model_name}/{version}"

            # Upload model file
            model_s3_key = f"{s3_base_key}/{model_filename}"
            model_upload_info = self.uploader.upload_file(
                local_file_path=str(model_path),
                s3_key=model_s3_key,
                metadata={
                    "model_name": model_name,
                    "version": version,
                    "type": "model",
                },
            )

            # Upload metadata file
            metadata_s3_key = f"{s3_base_key}/{metadata_filename}"
            metadata_upload_info = self.uploader.upload_file(
                local_file_path=str(metadata_path),
                s3_key=metadata_s3_key,
                metadata={
                    "model_name": model_name,
                    "version": version,
                    "type": "metadata",
                },
            )

            logger.info(
                f"Successfully saved {model_name} {version} "
                f"({model_size_mb:.2f} MB) to S3"
            )

            return {
                "version": version,
                "model_s3_uri": model_upload_info["s3_uri"],
                "metadata_s3_uri": metadata_upload_info["s3_uri"],
                "model_size_mb": model_size_mb,
                "upload_time_seconds": (
                    model_upload_info["upload_time_seconds"] +
                    metadata_upload_info["upload_time_seconds"]
                ),
            }

    def load_model_from_s3(
        self,
        model_name: str,
        version: Optional[str] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model and metadata from S3.

        Args:
            model_name: Name of the model
            version: Version to load (default: latest version)
            device: Device for PyTorch models ("cpu", "cuda", etc.)

        Returns:
            Dictionary containing:
                - model: Loaded model object
                - metadata: Model metadata (config, metrics, timestamp)
                - version: Version string

        Raises:
            ValueError: If model_name is invalid or version doesn't exist
            RuntimeError: If download or load fails

        Example:
            >>> manager = ModelVersionManager(bucket_name="my-models")
            >>> result = manager.load_model_from_s3(
            ...     model_name="text_classifier",
            ...     version="v3"
            ... )
            >>> model = result["model"]
            >>> config = result["metadata"]["config"]
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"Invalid model_name: {model_name}")

        # Determine version to load
        if version is None:
            # Load latest version
            existing_versions = self._list_versions(model_name)
            if not existing_versions:
                raise ValueError(f"No versions found for model: {model_name}")
            version = existing_versions[-1]
            logger.info(f"Loading latest version: {version}")
        else:
            # Validate version exists
            existing_versions = self._list_versions(model_name)
            if version not in existing_versions:
                raise ValueError(
                    f"Version {version} not found for {model_name}. "
                    f"Available: {existing_versions}"
                )

        logger.info(f"Loading {model_name} {version} from S3")

        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            s3_base_key = f"models/{model_name}/{version}"

            # Download metadata first
            metadata_s3_key = f"{s3_base_key}/metadata.json"
            metadata_local_path = temp_path / "metadata.json"

            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    metadata_s3_key,
                    str(metadata_local_path),
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ValueError(
                        f"Metadata not found for {model_name} {version}. "
                        f"S3 key: {metadata_s3_key}"
                    )
                else:
                    logger.error(f"Failed to download metadata: {e}")
                    raise RuntimeError(f"Metadata download failed: {e}")

            # Load metadata
            try:
                with open(metadata_local_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse metadata: {e}")
                raise RuntimeError(f"Metadata parsing failed: {e}")

            # Download model file
            model_filename = metadata.get("model_filename", f"model.{self.model_format}")
            model_s3_key = f"{s3_base_key}/{model_filename}"
            model_local_path = temp_path / model_filename

            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    model_s3_key,
                    str(model_local_path),
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ValueError(
                        f"Model file not found for {model_name} {version}. "
                        f"S3 key: {model_s3_key}"
                    )
                else:
                    logger.error(f"Failed to download model: {e}")
                    raise RuntimeError(f"Model download failed: {e}")

            # Load model
            try:
                if self.model_format == "pt":
                    # PyTorch model
                    model = torch.load(model_local_path, map_location=device)
                elif self.model_format == "pkl":
                    # Pickle-based model
                    import pickle
                    with open(model_local_path, "rb") as f:
                        model = pickle.load(f)
                else:
                    raise ValueError(f"Unsupported model format: {self.model_format}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Model deserialization failed: {e}")

            logger.info(f"Successfully loaded {model_name} {version} from S3")

            return {
                "model": model,
                "metadata": metadata,
                "version": version,
            }

    def list_all_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model with metadata.

        Args:
            model_name: Name of the model

        Returns:
            List of dictionaries with version info:
                - version: Version string
                - timestamp: Upload timestamp
                - metrics: Training metrics (if available)

        Example:
            >>> manager = ModelVersionManager(bucket_name="my-models")
            >>> versions = manager.list_all_versions("text_classifier")
            >>> for v in versions:
            ...     print(f"{v['version']}: accuracy={v['metrics'].get('accuracy')}")
        """
        versions = self._list_versions(model_name)

        version_info_list = []

        for version in versions:
            # Try to fetch metadata for each version
            try:
                metadata_s3_key = f"models/{model_name}/{version}/metadata.json"

                # Download metadata to temporary file
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as temp_file:
                    self.s3_client.download_file(
                        self.bucket_name,
                        metadata_s3_key,
                        temp_file.name,
                    )

                    with open(temp_file.name, "r") as f:
                        metadata = json.load(f)

                    version_info_list.append({
                        "version": version,
                        "timestamp": metadata.get("timestamp"),
                        "metrics": metadata.get("metrics", {}),
                        "config": metadata.get("config", {}),
                        "size_mb": metadata.get("model_size_bytes", 0) / (1024 * 1024),
                    })

            except Exception as e:
                logger.warning(f"Failed to load metadata for {version}: {e}")
                # Add version with minimal info
                version_info_list.append({
                    "version": version,
                    "timestamp": None,
                    "metrics": {},
                })

        return version_info_list

    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version from S3.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if successful, False otherwise

        Warning:
            This operation is irreversible. Use with caution.
        """
        logger.warning(f"Deleting {model_name} {version} from S3")

        # List all objects in the version folder
        s3_prefix = f"models/{model_name}/{version}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_prefix,
            )

            if "Contents" not in response:
                logger.warning(f"No objects found for {model_name} {version}")
                return False

            # Delete all objects
            objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": objects_to_delete},
            )

            logger.info(f"Deleted {len(objects_to_delete)} objects for {model_name} {version}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete version: {e}")
            return False


# Convenience functions for simple usage

def save_model_to_s3(
    model: Any,
    model_name: str,
    bucket_name: str,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    model_format: str = "pt",
) -> Dict[str, Any]:
    """
    Convenience function to save a model to S3 with automatic versioning.

    Args:
        model: Model object to save
        model_name: Name of the model
        bucket_name: S3 bucket name
        metrics: Training metrics (accuracy, loss, etc.)
        config: Training configuration (hyperparameters)
        model_format: Model file format ("pt", "pkl", etc.)

    Returns:
        Version info dictionary

    Example:
        >>> info = save_model_to_s3(
        ...     model=trained_model,
        ...     model_name="docreader_ocr",
        ...     bucket_name="my-ml-models",
        ...     metrics={"accuracy": 0.95, "loss": 0.05},
        ...     config={"lr": 0.001, "epochs": 50}
        ... )
        >>> print(f"Saved as {info['version']}")
    """
    manager = ModelVersionManager(bucket_name=bucket_name, model_format=model_format)
    return manager.save_model_to_s3(
        model=model,
        model_name=model_name,
        metrics=metrics,
        config=config,
    )


def load_model_from_s3(
    model_name: str,
    bucket_name: str,
    version: Optional[str] = None,
    model_format: str = "pt",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Convenience function to load a model from S3.

    Args:
        model_name: Name of the model
        bucket_name: S3 bucket name
        version: Version to load (default: latest)
        model_format: Model file format ("pt", "pkl", etc.)
        device: Device for PyTorch models ("cpu", "cuda")

    Returns:
        Dictionary with model, metadata, and version

    Example:
        >>> result = load_model_from_s3(
        ...     model_name="docreader_ocr",
        ...     bucket_name="my-ml-models",
        ...     version="v3"
        ... )
        >>> model = result["model"]
        >>> metrics = result["metadata"]["metrics"]
    """
    manager = ModelVersionManager(bucket_name=bucket_name, model_format=model_format)
    return manager.load_model_from_s3(
        model_name=model_name,
        version=version,
        device=device,
    )
