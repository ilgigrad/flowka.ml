
from conf.settings import STORAGE





class Actual_Storage():
    def __init__(self,storage=None):
        if storage in ['aws','s3','AWS','boto3'] or (storage==None and STORAGE=='AWS'):
            from .boto3 import Client_Storage
        elif storage in ['minio','MINIO'] or (storage==None and STORAGE=='MINIO'):
            from .minio import Client_Storage
        else:#lets set minio as default storage
            from .minio import Client_Storage
        self.client=Client_Storage()
        self.type=self.client.type

    def upload(self,local,remote):
        return self.client.upload(local,remote)

    def download(self,remote,local):
        return self.client.download(remote,local)

    def get(self,remote):
        return self.client.get(remote)

    def delete(self,remote):
        return self.client.delete(remote)

    def exists(self, remote):
        return self.client.exists(remote)

    def list(self,remote):
        return self.client.list(remote)

    def deletes(self,remote):
        return self.client.deletes(remote)
