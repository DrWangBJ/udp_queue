# udp_queue

Use UDP socket send and receive data flow.

## Usage

```python
from udp_queue import UDPQueue
sender=UDPQueue(host='127.0.0.1',is_host=True)
```

examples can be found in `test_udp_que_client.py` and `test_udp_server.py`

# LV_notifier

A LabVIEW like notifier writen by python. You can use it create a notifier and listen for the notice. Multiple notice sender and receiver can be used.
