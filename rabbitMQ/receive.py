import pika
from conf.settings import RABBITMQ as RMQ


def mqreceive(queue=None, delivery_tag=None, stop_tag='stop'):
    """
       delivery_tag : max number of messages to receive before exiting the process
        stop_tag : tag for stopping consumming 'stop' -> stop_queue
    """
    queue = queue or RMQ['QUEUE']
    messages=[]
    headers=[]

    credentials = pika.PlainCredentials(RMQ['USER'], RMQ['PASSWORD'])
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RMQ['SERVER'],port=RMQ['PORT'], virtual_host=RMQ['VHOST'], credentials=credentials))
    channel = connection.channel()
    for method_frame, properties, body in channel.consume(queue,no_ack=False): #no_ack becomes auto_ack in next pikaversion (1.0.0)
        tag=body.decode('UTF-8')
        messages+=[tag]
        headers+=[properties.headers]

        channel.basic_ack(method_frame.delivery_tag)
        if (delivery_tag and method_frame.delivery_tag == delivery_tag) or (stop_tag and tag==stop_tag):
            break
    # Cancel the consumer and return any pending messages
    requeued_messages = channel.cancel()
    # Close the channel and the connection
    if channel.is_open:
        channel.close()
    connection.close()
    return headers, messages
