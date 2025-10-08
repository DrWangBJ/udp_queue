import threading
import time
from typing import Any, Optional, List, Tuple


class LVNotifier:
    """LabVIEW风格的通知器类，支持多生产者多消费者的广播通知
    """
    
    def __init__(self,name:str=None):
        self._name=name
        self._condition = threading.Condition()  # 条件变量用于广播通知
        self._current_msg: Optional[Any] = None   # 当前通知消息
        self._msg_available:dict[str,bool] = {}   # 消息可用标记
        self._subscribers: List[str] = []         # 订阅者ID列表
        self._read_count = 0                      # 消息已读取计数器
        self._lock = threading.Lock()             # 保护订阅者列表的锁
        self.stop=threading.Event()               # 停止标记
        self._get_last:List[bool]=[]              # 是否获取上一个，为真时获取注册前的一个消息
        self._timestamp=time.time()               # 发送通知的时间
        self._last_timestamp: dict[str,float]={}  # 收到上一次通知的时间，用这个时间可以区分是不是最新的消息
    def get_name(self)->str:
        return self._name  

    def subscribe(self, subscriber_id: str,get_last:bool=True) -> None:
        """订阅通知器（类似LabVIEW的Subscribe Notifier）
        subscriber_id用于标识不同的订阅者
        get_last用于标识是否获取上一个通知消息，如果是假，在这个订阅的时候，之前发送的通知是不知道的，为真的时候，可以获取到订阅之前的一条消息
        """
        with self._lock:
            if subscriber_id not in self._subscribers:
                self._subscribers.append(subscriber_id)
                self._get_last.append(get_last)
                self._msg_available[subscriber_id]=get_last
                self._last_timestamp[subscriber_id]=0.0

    def wait_for_notification(self, subscriber_id: str, timeout: Optional[float] = None) -> Tuple[bool, Any]:
        """
        等待通知（类似LabVIEW的Wait on Notifier）
        返回值：(是否成功接收, 数据) - 超时返回(False, None)，成功返回(True, 数据)
        """
        with self._condition:
            # 检查是否为有效订阅者
            with self._lock:
                # print(self._msg_available.get(subscriber_id,False))
                if subscriber_id not in self._subscribers:
                    return (False, "未订阅通知器")

            # 等待消息或超时
            received = self._condition.wait_for(
                lambda: self._msg_available.get(subscriber_id, False),
                timeout=timeout
            )

            if not received:
                # print("等待超时")
                return (False, None)  # 超时

            # 读取消息
            msg = self._current_msg
        
        # 更新读取计数器
        with self._lock:
            self._msg_available[subscriber_id] = False
            self._last_timestamp[subscriber_id]=self._timestamp

        return (True, msg)  # 成功接收
    def get_latest_timestamp(self,subscriber_id:str)->float:
        """获取最后一次接收消息的时间戳"""
        with self._lock:
            return self._last_timestamp.get(subscriber_id,0.0)    

    def unsubscribe(self, subscriber_id: str) -> None:
        """取消订阅（类似LabVIEW的Unsubscribe Notifier）"""
        with self._lock:
            if subscriber_id in self._subscribers:
                idx=self._subscribers.index(subscriber_id)
                self._get_last.pop(idx)
                self._subscribers.remove(subscriber_id)
                self._last_timestamp.pop(subscriber_id,None)
                self._msg_available.pop(subscriber_id,None)

    def send_notification(self, msg: Any) -> None:
        """发送通知（类似LabVIEW的Send Notifier）"""
        with self._lock:
            self._current_msg = msg
            self._timestamp=time.time()
        with self._condition:
            # 只有存在订阅者时才发送
            with self._lock:
                if not self._subscribers:
                    return

                # 设置消息并广播
                self._current_msg = msg
                self._timestamp=time.time()
                # print(f'广播消息是{self._current_msg}')
                for subscriber in self._subscribers:
                    self._msg_available[subscriber]=True
                self._condition.notify_all()
    def release(self) -> None:
        self.stop.set()
        with self._lock:
            self._last_timestamp.clear()
            self._get_last.clear()
            self._subscribers.clear()
            self._msg_available = False
            self._current_msg = None
        print("结束Notifier")
    def is_stop(self) -> bool:
        if self.stop.is_set():
            print("stop")
            return True
        else:
            return False
        
class LVNotifierMan():
    def __init__(self):
        self._Notifiers:dict[str,LVNotifier]={}
    def new_notifier(self,name:str=None)->LVNotifier:
        """创建一个新的通知器，若name已存在，则返回已存在的通知器"""
        if name is None:
            name=str(time.time())
        if name in self._Notifiers:
            print(f"通知器{name}已存在")
            return self._Notifiers[name]
        else:
            print(f"创建通知器{name}")
            notifier=LVNotifier(name)
            self._Notifiers[name]=notifier
            return notifier
    def close_notifier(self,name:str)->bool:
        """关闭一个通知器，返回是否成功关闭"""
        if name in self._Notifiers:
            print(f"关闭通知器{name}")
            notifier=self._Notifiers.pop(name)
            notifier.release()
            return True
        else:
            print(f"通知器{name}不存在")
            return False



# ------------------- 使用示例（LabVIEW风格调用） -------------------
def producer_task(notifier: LVNotifier, producer_id: str, delay: float):
    """生产者任务：周期性发送通知"""
    for i in range(2):
        msg = f"来自{producer_id}的消息{i}"
        print("*"*30)
        print(f"[{time.ctime()}] 生产者{producer_id}发送: {msg}")
        print("-"*30)
        notifier.send_notification(msg)
        time.sleep(delay)


def consumer_task(notifier: LVNotifier, consumer_id: str, timeout: float, get_last: bool=True):
    """消费者任务：订阅->等待通知->取消订阅"""
    # 1. 订阅通知器
    notifier.subscribe(consumer_id,get_last=get_last)
    print(f"[{time.ctime()}] 消费者{consumer_id}已订阅")

    try:
        # 2. 循环等待通知
        while True and not notifier.is_stop():
            success, data = notifier.wait_for_notification(consumer_id, timeout)
            
            if not success:
                # print(f"[{time.ctime()}] 消费者{consumer_id}等待超时")
                time.sleep(0.1)
                continue
                
            print(f"[{time.ctime()}] 消费者{consumer_id}收到: {data}, 时间戳:{notifier.get_latest_timestamp(consumer_id)}")

            time.sleep(0.5)  # 模拟处理时间
        print("消费者退出")
    except :
        print("发生错误")

    finally:
        # 3. 取消订阅
        notifier.unsubscribe(consumer_id)
        print(f"[{time.ctime()}] 消费者{consumer_id}已取消订阅")


if __name__ == "__main__":
    # 创建通知器对象
    # notifier = LVNotifier()
    Notice=LVNotifierMan()
    notifier=Notice.new_notifier("test1")

    # 启动生产者线程
    producers = [
        threading.Thread(target=producer_task, args=(notifier, "P1", 2), daemon=True),
        threading.Thread(target=producer_task, args=(notifier, "P2", 3), daemon=True)
    ]

    # 启动消费者线程
    consumers = [
        threading.Thread(target=consumer_task, args=(notifier, "C3", 5, False), daemon=True),
        threading.Thread(target=consumer_task, args=(notifier, "C1", 5, True), daemon=True),
        threading.Thread(target=consumer_task, args=(notifier, "C2", 5, False), daemon=True)
    ]

    # 启动所有线程
    for p in producers:
        p.start()
    time.sleep(0.5)  # 确保生产者先启动
    for c in consumers:
        c.start()
    time.sleep(10)  # 主线程等待一段时间后结束
    # 等待线程完成
    for p in producers:
        p.join()

    time.sleep(1)
    # notifier.release()
    Notice.close_notifier("test1")
    
    # 给消费者足够时间处理最后一条消息
    # time.sleep(5)
    print("程序结束")
