from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import boto3
from botocore.handlers import disable_signing


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", help="output BASE directory")

    smk = parser.parse_args()

    s3 = boto3.resource("s3")
    s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)

    bucket = s3.Bucket("broad-alkesgroup-ukbb-ld")

    for item in bucket.objects.all():
        if "baseline" in str(item.key) or "readme" in str(item.key):
            continue

        output_path = Path(smk.output_dir) / item.key
        if output_path.is_file():
            continue
        with Path(output_path).open("wb") as f:
            bucket.download_fileobj(item.key, f)


if __name__ == "__main__":
    main()
