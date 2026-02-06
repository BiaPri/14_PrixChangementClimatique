"""Download all objects from an S3 folder (prefix) into a local directory."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str


def parse_s3_uri(s3_uri: str) -> S3Location:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    _, _, rest = s3_uri.partition("s3://")
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    return S3Location(bucket=bucket, prefix=prefix)


def normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix.lstrip("/")


def iter_s3_objects(
    client,
    bucket: str,
    prefix: str,
) -> Iterable[str]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if key:
                yield key


def split_key(prefix: str, key: str) -> Tuple[str, str]:
    if prefix and key.startswith(prefix):
        relative = key[len(prefix) :]
    else:
        relative = key
    relative = relative.lstrip("/")
    return key, relative


def download_s3_prefix(
    *,
    bucket: str,
    prefix: str,
    dest: str,
    region: str | None,
    profile: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    session_token: str | None,
    endpoint_url: str | None,
    no_sign_request: bool,
    skip_existing: bool,
) -> None:
    os.makedirs(dest, exist_ok=True)

    LOGGER.info(
        "Starting download: bucket=%s, prefix=%s, dest=%s",
        bucket,
        prefix or "(root)",
        dest,
    )
    if endpoint_url:
        LOGGER.info("Using endpoint URL: %s", endpoint_url)
    if profile:
        LOGGER.info("Using AWS profile: %s", profile)
    if access_key_id and secret_access_key:
        LOGGER.info("Using explicit access keys from script")
    if no_sign_request:
        LOGGER.info("Using unsigned requests")

    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    if access_key_id and secret_access_key:
        session_kwargs["aws_access_key_id"] = access_key_id
        session_kwargs["aws_secret_access_key"] = secret_access_key
    if session_token:
        session_kwargs["aws_session_token"] = session_token
    session = boto3.Session(**session_kwargs)
    config = Config(signature_version=UNSIGNED) if no_sign_request else None
    client = session.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        config=config,
    )

    LOGGER.info("Listing objects from S3...")

    total = 0
    downloaded = 0
    for key in iter_s3_objects(client, bucket, prefix):
        _, relative = split_key(prefix, key)
        if not relative or relative.endswith("/"):
            continue
        total += 1

        local_path = os.path.join(dest, relative)
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        if skip_existing and os.path.exists(local_path):
            LOGGER.info("Skip existing: %s", local_path)
            continue

        LOGGER.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
        client.download_file(bucket, key, local_path)
        downloaded += 1

    LOGGER.info(
        "Done. Listed %s objects under s3://%s/%s. Downloaded %s file(s) to %s.",
        total,
        bucket,
        prefix,
        downloaded,
        dest,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    download_s3_prefix(
        bucket="qppcc-upload",
        prefix="pipeline_inputs/",
        dest="data/dbt_pipeline/pipeline_inputs",
        region=None,
        profile=None,
        access_key_id=os.getenv("ACCESS_KEY_ID"),
        secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        session_token=None,
        endpoint_url="https://s3.fr-par.scw.cloud",
        no_sign_request=False,
        skip_existing=True,
    )


if __name__ == "__main__":
    main()
