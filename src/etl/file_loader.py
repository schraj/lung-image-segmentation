import os

class FileLoader:
    def __init__(self,bucket_name, s3resource):
        self.bucket_name = bucket_name
        self.s3resource = s3resource
        self.bucket = self.s3resource.Bucket(bucket_name)
    
    def upload_to_s3(self,channel,file):
        key = channel + '/' + file
        data = open(key,'rb')
        self.bucket.put_object(Key=key, Body=data)

    def upload_files(self, training_path):
        # TODO - remove path up to /input
        #        remove files or dirs starting with "."
        for root, dirs, files in os.walk(training_path):
            for name in files:
                self.upload_to_s3(root, name)

    def download_files(self, directory_name):
        targets = []
        for obj in self.bucket.objects.all():
            if ('.DS_Store' in obj.key):
                continue
            targets.append(obj.key)
        for target in targets:
            file_path = obj.key.split('/input/')[1]
            self.bucket.download_file(target, os.path.join(os.getcwd(), directory_name + '/' + file_path))