"""
S3 upload utilities for ML model artifacts.

Production-ready S3 upload with retry logic, progress tracking,
metadata tagging, and comprehensive error handling.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config

from ..utils.logger import get_logger

logger = get_logger(__name__)


class S3Uploader:
    """
    S3 uploader for ML training artifacts.

    Features:
    - Automatic retry with exponential backoff
    - Progress tracking for large files
    - Metadata tagging (version, timestamp, experiment)
    - Multipart upload for large files
    - Comprehensive error handling
    - IAM role support
    """

    def __init__(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: S3 bucket name
            region: AWS region (default: from AWS config)
            max_retries: Maximum number of upload retries
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.bucket_name = bucket_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure boto3 with retries
        config = Config(
            region_name=region,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
        )

        try:
            # Use IAM role credentials (recommended for production)
            self.s3_client = boto3.client("s3", config=config)
            logger.info(f"S3 client initialized for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

        # Verify bucket exists
        self._verify_bucket()

    def _verify_bucket(self) -> None:
        """Verify that the S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Verified access to bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"Bucket does not exist: {self.bucket_name}")
            elif error_code == "403":
                logger.error(f"Access denied to bucket: {self.bucket_name}")
            else:
                logger.error(f"Error accessing bucket: {e}")
            raise

    def upload_file(
        self,
        local_file_path: str,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to S3 with retry logic and progress tracking.

        Args:
            local_file_path: Path to local file
            s3_key: S3 object key (path in bucket)
            metadata: Custom metadata tags (e.g., version, experiment_id)
            progress_callback: Optional callback(bytes_transferred) for progress
            extra_args: Additional upload arguments (e.g., ServerSideEncryption)

        Returns:
            Dictionary with upload metadata:
                - s3_uri: Full S3 URI
                - s3_url: Public URL (if bucket is public)
                - version_id: S3 version ID (if versioning enabled)
                - etag: S3 ETag
                - file_size: File size in bytes
                - upload_time: Upload duration in seconds

        Raises:
            FileNotFoundError: If local file doesn't exist
            ValueError: If file is empty or path is invalid
            RuntimeError: If upload fails after retries
        """
        start_time = time.time()

        # Validate local file
        local_path = Path(local_file_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        if not local_path.is_file():
            raise ValueError(f"Path is not a file: {local_file_path}")

        file_size = local_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {local_file_path}")

        logger.info(
            f"Uploading {local_file_path} ({file_size / 1024 / 1024:.2f} MB) "
            f"to s3://{self.bucket_name}/{s3_key}"
        )

        # Prepare metadata
        upload_metadata = metadata or {}
        upload_metadata.update(
            {
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "original_filename": local_path.name,
                "file_size_bytes": str(file_size),
            }
        )

        # Prepare upload arguments
        upload_args = extra_args or {}
        upload_args["Metadata"] = upload_metadata

        # Add server-side encryption by default
        if "ServerSideEncryption" not in upload_args:
            upload_args["ServerSideEncryption"] = "AES256"

        # Setup progress tracking
        if progress_callback is None:

            def default_progress(bytes_transferred):
                percent = (bytes_transferred / file_size) * 100
                logger.debug(f"Upload progress: {percent:.1f}%")

            progress_callback = default_progress

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Use upload_file for automatic multipart handling
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=upload_args,
                    Callback=progress_callback,
                )

                # Get object metadata
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name, Key=s3_key
                )

                upload_duration = time.time() - start_time

                result = {
                    "s3_uri": f"s3://{self.bucket_name}/{s3_key}",
                    "s3_url": f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}",
                    "bucket": self.bucket_name,
                    "key": s3_key,
                    "version_id": response.get("VersionId"),
                    "etag": response.get("ETag", "").strip('"'),
                    "file_size": file_size,
                    "upload_time_seconds": upload_duration,
                }

                logger.info(
                    f"Successfully uploaded to {result['s3_uri']} "
                    f"in {upload_duration:.2f}s"
                )

                return result

            except NoCredentialsError:
                logger.error("AWS credentials not found. Check IAM role or credentials.")
                raise RuntimeError(
                    "AWS credentials not configured. Use IAM role or configure credentials."
                )

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                logger.warning(
                    f"Upload attempt {attempt + 1}/{self.max_retries} failed: {error_code}"
                )
                last_exception = e

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Upload failed after {self.max_retries} attempts: {e}")

            except Exception as e:
                logger.error(f"Unexpected error during upload: {e}")
                last_exception = e

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries failed
        raise RuntimeError(
            f"Failed to upload {local_file_path} after {self.max_retries} attempts: "
            f"{last_exception}"
        )

    def upload_model(
        self,
        model_path: str,
        model_name: str,
        version: str,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload ML model with automatic metadata tagging.

        Args:
            model_path: Path to model file (.pth, .onnx, etc.)
            model_name: Model name (e.g., 'docreader_ocr')
            version: Model version (e.g., 'v1.0.0', 'experiment_123')
            experiment_id: Optional experiment tracking ID

        Returns:
            Upload metadata dictionary
        """
        # Generate S3 key with versioning structure
        model_extension = Path(model_path).suffix
        s3_key = f"models/{model_name}/{version}/model{model_extension}"

        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "model_version": version,
            "model_type": model_extension.lstrip("."),
        }

        if experiment_id:
            metadata["experiment_id"] = experiment_id

        logger.info(f"Uploading model: {model_name} v{version}")

        return self.upload_file(
            local_file_path=model_path, s3_key=s3_key, metadata=metadata
        )


# Convenience function for simple uploads
def upload_to_s3(
    local_file_path: str,
    bucket_name: str,
    s3_key: str,
    metadata: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Simple S3 upload function for backward compatibility.

    Args:
        local_file_path: Path to local file
        bucket_name: S3 bucket name
        s3_key: S3 object key
        metadata: Optional metadata tags

    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        uploader = S3Uploader(bucket_name=bucket_name)
        uploader.upload_file(local_file_path=local_file_path, s3_key=s3_key, metadata=metadata)
        return True
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False