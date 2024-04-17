from boto3 import Session, client,resource
from botocore.exceptions import ClientError
from conf.settings import (
    AWS_MEDIA_DIR,
    AWS_STORAGE_BUCKET_NAME,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_REGION_NAME,
    )


# Let's use Amazon S3



# Upload a new file

class Client_Storage():
    """low level storage
    """
    def __init__(self):
        #self.resource=resource('s3')
        self.mediadir=AWS_MEDIA_DIR
        self.bucket=AWS_STORAGE_BUCKET_NAME
        self.client = client(
            's3',
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name = AWS_S3_REGION_NAME,
            )
        self.type='s3'

    def upload(self,local,remote):
        try:
            with open(local,'rb') as file_data:
                response=self.client.put_object(Bucket = self.bucket,
                Key=self.mediadir+remote,Body=file_data)
            return response['ResponseMetadata']['HTTPStatusCode']
        except:
            return 500

    def download(self,remote,local):
        try:
            data=self.client.get_object(Bucket= self.bucket,Key=self.mediadir+remote)
            with open(local,'wb') as file_data:
                for f in data['Body']:
                    file_data.write(f)
            return data['ResponseMetadata']['HTTPStatusCode']
        except:
            return 500

    def get(self,remote):
        try:
            import io
            data=self.client.get_object(Bucket= self.bucket,Key=self.mediadir+remote)
            return io.BytesIO(data['Body'].read())
        except:
            return None

    def delete(self,remote):
        try:
            if not self.exists(remote): #file does not exists
                return 404
            response=self.client.delete_object(Bucket= self.bucket,Key=self.mediadir+remote)
            #self.resource.Object(self.bucket,self.mediadir+remote).delete()
            if self.exists(remote): #file ahas not been deleted
                return 204
            else:
                return 200
            #return  response['ResponseMetadata']['HTTPStatusCode'] returns always 204
        except:
            return 500

    def exists(self, remote):
        """return the key's size if it exist, else None"""
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.mediadir+remote,
        )
        for obj in response.get('Contents', []):
            if obj['Key'] == self.mediadir+remote:
                #return obj['Size']
                return True
        return False

    def list(self,remote):
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.mediadir+remote,
        )
        if response:
            return [obj['Key'] for obj in response.get('Contents', [])]
        return []

    def deletes(self,remote):
        del_list=self.list(remote)
        for key in del_list:
            self.delete(key.split(self.mediadir)[-1])



class Ressource_Storage:
    """high level storage
    """
    def __init__(self):
        self.bucket=AWS_STORAGE_BUCKET_NAME
        session = Session(
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name = AWS_S3_REGION_NAME,
            )
        self.resource=session.resource('s3')

    def upload(self,local,remote):
        # DO NOT USE
        try:
            data = open(local, 'rb')
            self.resource.Bucket(self.bucket).put_object(Key=self.mediadir+remote,Body=data)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    def download (self,remote,local):
        # DO NOT USE
        try:
            self.resource.Bucket(self.bucket).download_file(Key=self.mediadir+remote, Filename=local)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
