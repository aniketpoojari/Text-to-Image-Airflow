import boto3
import zipfile
import os
from typing import Dict, Any
from botocore.exceptions import ClientError

class DataUploader:
    """Core functionality for data upload operations"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def upload_data(self, local_path: str, s3_path: str) -> Dict[str, Any]:
        """Upload data to S3 with zip compression if needed"""
        
        try:
            # Parse S3 path
            s3_parts = s3_path.replace('s3://', '').split('/', 1)
            bucket_name = s3_parts[0]
            s3_key = s3_parts[1] if len(s3_parts) > 1 else ''
            
            # Check if local path is a directory
            if os.path.isdir(local_path):
                # Create zip file from directory
                zip_filename = f"{os.path.basename(local_path)}.zip"
                zip_path = os.path.join(os.path.dirname(local_path), zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(local_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Create relative path for zip archive
                            arcname = os.path.relpath(file_path, local_path)
                            zipf.write(file_path, arcname)
                
                upload_file = zip_path
                s3_key = s3_key or zip_filename
            else:
                upload_file = local_path
                s3_key = s3_key or os.path.basename(local_path)
            
            # Upload to S3
            print(f"Uploading {upload_file} to s3://{bucket_name}/{s3_key}")
            
            self.s3_client.upload_file(
                upload_file,
                bucket_name,
                s3_key
            )
            
            # Clean up temporary zip file if created
            if os.path.isdir(local_path) and os.path.exists(zip_path):
                os.remove(zip_path)
            
            return {
                "upload_status": "completed",
                "s3_path": f"s3://{bucket_name}/{s3_key}",
                "local_path": local_path,
                "file_size": os.path.getsize(upload_file) if os.path.exists(upload_file) else 0
            }
            
        except ClientError as e:
            raise Exception(f"AWS S3 upload failed: {e}")
        except Exception as e:
            raise Exception(f"Upload failed: {e}")
