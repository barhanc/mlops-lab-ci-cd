import os
import boto3

from sentiment_app.settings import Settings


def download_s3_folder(settings: Settings):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(settings.s3_bucket)  # type: ignore

    for obj in bucket.objects.filter(Prefix=settings.s3_model_dir):
        target = obj.key

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue

        bucket.download_file(obj.key, target)
