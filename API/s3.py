import os
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv
from boto3.s3.transfer import TransferConfig

load_dotenv(".env.api")


class S3Client:
    def __init__(self):
        endpoint = os.getenv("S3_URL", "").strip()
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=os.getenv("AWS_DEFAULT_REGION", "ru1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=Config(
                signature_version="s3v4",
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                s3={
                    "addressing_style": "path",
                    "payload_signing_enabled": False,
                },
            ),
        )
        self.bucket = os.getenv("BUCKET")
        if not endpoint or not self.bucket:
            raise RuntimeError("S3_URL and BUCKET must be set in .env.api")

    def upload_file(self, file_path: str, key: str | None = None) -> str:
        path = Path(file_path)
        object_key = key or path.name

        self.s3.upload_file(
            str(path),
            self.bucket,
            object_key,
            Config=TransferConfig(use_threads=False),
        )
        return self.public_url(object_key)

    def download_file(self, key: str, file_path: str):
        self.s3.download_file(self.bucket, key, file_path)

    def delete_file(self, key: str):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def list_files(self, prefix=""):
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        return [obj["Key"] for obj in response.get("Contents", [])]

    def public_url(self, key: str) -> str:
        endpoint = self.s3.meta.endpoint_url.rstrip("/")
        return f"{endpoint}/{self.bucket}/{key.lstrip('/')}"


s3 = S3Client()
