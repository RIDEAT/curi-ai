import boto3

import os
from chatbot import chat, upsert
from dotenv import load_dotenv


load_dotenv()

def read_s3_text_file():
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_access_key = os.environ.get('AWS_ACCESS_SECRET_KEY')
    bucket_name = "workplug-workspace"
    file_key = "workspace2.txt"
    
    # S3 클라이언트 생성
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key_id, 
                      aws_secret_access_key=aws_secret_access_key, 
                      )

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        text_data = response['Body'].read().decode('utf-8')
        upsert(text_data)
        return 'OK'
    except Exception as e:
        print(f"파일을 읽어올 수 없음: {str(e)}")
        return None