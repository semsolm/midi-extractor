# backend/app/utils/storage.py
import os
from google.cloud import storage
from flask import current_app

def get_bucket():
    client = storage.Client()
    bucket_name = current_app.config['GCS_BUCKET_NAME']
    return client.bucket(bucket_name)

def upload_stream_to_gcs(file_obj, destination_blob_name, content_type):
    """업로드된 파일 스트림을 바로 GCS로 전송"""
    bucket = get_bucket()
    blob = bucket.blob(destination_blob_name)
    file_obj.seek(0)
    blob.upload_from_file(file_obj, content_type=content_type)

def upload_file_to_gcs(source_file_path, destination_blob_name):
    """로컬 파일을 GCS로 업로드"""
    bucket = get_bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

def download_file_from_gcs(source_blob_name, destination_file_path):
    """GCS 파일을 로컬로 다운로드"""
    bucket = get_bucket()
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)

def download_blob_to_memory(source_blob_name):
    """GCS 파일을 메모리로 다운로드 (바로 전송용)"""
    bucket = get_bucket()
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()