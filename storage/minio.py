from minio import Minio
from conf.settings import (
    MINIO_STORAGE_BUCKET_NAME,
    MINIO_STORAGE_ENDPOINT,
    MINIO_STORAGE_ACCESS_KEY,
    MINIO_STORAGE_SECRET_KEY,
    )
import os

class Client_Storage():

    def __init__(self):
        self.bucket=MINIO_STORAGE_BUCKET_NAME
        self.client = Minio(
        MINIO_STORAGE_ENDPOINT,
        access_key=MINIO_STORAGE_ACCESS_KEY,
        secret_key=MINIO_STORAGE_SECRET_KEY,
        secure=False
        )
        self.type='minio'

    def exists(self, remote):
        """return the key's size if it exist, else None"""
        response = self.client.list_objects_v2(
            self.bucket,
            prefix=remote,
            )
        for obj in response:
            if obj.object_name == remote:
                return True
        return False


    def upload(self,local,remote):
        try:
            with open(local,'rb') as file_data:
                file_stat = os.stat(local)
                response=self.client.put_object(self.bucket,remote,file_data,file_stat.st_size)
            return 200
        except:
            return 500

    def download(self,remote,local):
        try:
            data=self.client.get_object(self.bucket,remote)
            with open(local,'wb') as file_data:
                for f in data.stream(32*1024):
                    file_data.write(f)
            return data.status
        except:
            return 500

    def get(self,remote):
        try:
            import io
            response=self.client.get_object(self.bucket,remote)
            return io.BytesIO(response.data)
        except:
            return None

    def delete(self,remote):
        if not self.exists(remote): #file does not exists
            return 404
        try:
            response=self.client.remove_object(self.bucket,remote)
        except:
            return 500

        if self.exists(remote): #file ahas not been deleted
            return 204
        else:
            return 200

    def list(self,remote):
        response = self.client.list_objects_v2(
            self.bucket,
            prefix=remote,
            )
        if response:
            return [obj.object_name for obj in response]
        return []

    def deletes(self,remote):
        del_list=self.list(remote)
        for key in del_list:
            self.delete(key)
