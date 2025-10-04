from udp_queue import UDPQueue
import time

receiver=UDPQueue(host='127.0.0.1',is_server=False)
try:
    while True:
        # sender.put(b"world")
        print(receiver.get(timeout=2))
        time.sleep(5)
except KeyboardInterrupt:
    pass
finally:
    receiver.clean()