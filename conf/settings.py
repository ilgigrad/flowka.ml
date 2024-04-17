"""
settings for flowka ML project.
"""
from os.path import abspath, dirname, join
import os
import sys
import json

def root(*dirs):
    base_dir=join(dirname(__file__),'..')
    return abspath(join(base_dir,*dirs))

with open(join(root(),'secrets.json')) as f:
    secrets = json.loads(f.read())

def get_secret(setting):
    try:
        secret=secrets[setting]
        if secret.lower()=='false':
            return False
        elif secret.lower()=='true':
            return True
        else:
            return secret
    except KeyError:
        error_msg = 'set the {0} environment variable'.format(setting)
        raise ImproperlyConfigured(error_message)

#                 RABBITMQ

#staging
env=os.environ.get('DJANGO_SETTINGS_MODULE','flowkaml.dev_settings').split('.')[-1]

#-------------------- AWS -------------------
AWS_DEFAULT_ACL = None
AWS_ACCESS_KEY_ID = get_secret('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = get_secret('AWS_SECRET_ACCESS_KEY')
AWS_S3_USER=get_secret('AWS_S3_USER')
AWS_STORAGE_BUCKET_NAME = 'ilgigrad-staging-bucket'
AWS_S3_OBJECT_PARAMETERS = {'Expires': 'Thu, 31 Dec 2099 20:00:00 GMT','CacheControl': 'max-age=86400000',}
AWS_S3_REGION_NAME='eu-west-3'
AWS_S3_URL = 'https://{0}.s3.{1}.amazonaws.com/'.format(AWS_STORAGE_BUCKET_NAME,AWS_S3_REGION_NAME)
AWS_MEDIA_DIR = 'uploads/'

RABBITMQ={
    'USER':get_secret('RABBITMQ-USER'),
    'PASSWORD':get_secret('RABBITMQ-PASSWORD'),
    'VHOST':'fl01',
    'SERVER':'rabbit.75.ilgigrad.net',
    'PORT':'5672',
    'QUEUE':'DC'
    }

#-------------------- MINIO -------------------
if env=='dev_settings':
    MINIO_STORAGE_ENDPOINT = 'minio.75.ilgigrad.net:9001'
    MINIO_STORAGE_BUCKET_NAME = 'uploads'
    MINIO_STORAGE_ACCESS_KEY = get_secret("MINIO_STORAGE_ACCESS_KEY")
    MINIO_STORAGE_SECRET_KEY = get_secret("MINIO_STORAGE_SECRET_KEY")

    STORAGE='MINIO'
    HTTP_FILE_SERVER='http://minio.75.ilgigrad.net:80'
    HTTP_DJANGO_SERVER='http://django.75.ilgigrad.net:8000'
    RABBITMQ['VHOST']='fl01'

if env=='staging_settings':
    STORAGE='AWS'
    HTTP_FILE_SERVER=AWS_S3_URL
    HTTP_DJANGO_SERVER='https://92.ilgigrad.net'
    RABBITMQ['VHOST']='fl02'
