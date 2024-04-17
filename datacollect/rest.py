from conf.settings import HTTP_FILE_SERVER, HTTP_DJANGO_SERVER
from datacollect.dataset import DataSet as DS
import json
import requests


def restImport(headers,files):
    processed=0
    for i in range(len(files)):
        file=files[i]
        #datafile=headers[i]['file-id']
        uid=headers[i]['file-uid']
        datafile=headers[i]['file-id']

        ds=DS()
        #import ipdb; ipdb.set_trace()
        response=ds.read(file)
        if response:
            snippet=ds.snippet(100,10).to_json()
            metadata=ds.details.to_json()
            restHeaders = {'Accept' : 'application/json', 'contentType' : 'application/json'}
            url=HTTP_DJANGO_SERVER+'/datacollect/rest/dataset/{}'.format(uid)
            r = requests.post(url, data=json.dumps({'datafile':datafile,'snippet':snippet,'metadata':metadata}), headers=restHeaders)
            print("OK - {}\n url:{}\n response:{}\n".format(file, url,r))
            processed+=1
        else:

            url=HTTP_DJANGO_SERVER+'/filer/rest/file/error/{}'.format(uid)
            r = requests.put(url, data=json.dumps({'is_valid':False,}))
            print("Error on {}\n url:{}\n response:{}\n".format(file, url,r))

    return processed
