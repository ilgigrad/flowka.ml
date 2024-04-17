from rabbitMQ.receive import mqreceive
import sys
from datacollect.rest import restImport

def main(argv):
    if len(argv)>1:
        print("direct")
        files=argv[1:]
    else:
        loop=True
        while loop:
            print("waiting for queue...")
            headers, files = mqreceive(queue='DC', delivery_tag=1, stop_tag='stop')

            for header in headers:
                if header.get('STOP',None):
                    print("stop requested from RabbitMQ")
                    loop=False
                    break
            print('<br>'.join(files))
            processed=restImport(headers,files)
            print('{} file(s) processed'.format(processed) )
main(sys.argv)
