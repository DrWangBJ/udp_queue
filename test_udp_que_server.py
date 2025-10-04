from udp_queue import UDPQueue
import time

sender=UDPQueue(host='127.0.0.1',is_server=True)
try:
    while True:
        strdata= f"hello {time.time()}"
        sender.put(strdata.encode('utf-8'))

        time.sleep(5)
except KeyboardInterrupt:
    pass
finally:
    sender.clean()