import socket
import threading
from queue import Queue, Empty
import time
import struct
import numpy as np
import pickle
from typing import Tuple, Optional, Dict, List, Any, Union
from typing_extensions import TypeAlias
import gc

# 数据类型定义（用于头部标识）
TYPE_BYTES = 0x01
TYPE_LIST = 0x02
TYPE_NUMPY = 0x03
TYPE_STRING = 0x04

DTYPE_INT=0x01
DTYPE_LONG=0x02
DTYPE_FLOAT=0x03
DTYPE_DOUBLE=0x04


# 类型到名称的映射（用于调试）
TYPE_NAMES = {
    TYPE_BYTES: "bytes",
    TYPE_LIST: "list",
    TYPE_NUMPY: "numpy.ndarray",
    TYPE_STRING: "string"
}

DTYPE_NAMES={
    DTYPE_INT:"int32",
    DTYPE_LONG:"int64",
    DTYPE_FLOAT:"float32",
    DTYPE_DOUBLE:"float64"
}

# 常量定义
MAX_FRAME_SIZE = 1024    # 最大单帧大小（字节）
HEADER_SIZE = 32         # 头部固定大小（32字节）
ENQ_HEARTBEAT = b'\x05'  # 客户端心跳标识
EOF_SIGNAL = b'\x04'     # 服务器退出标识

# 类型别名
Address: TypeAlias = Tuple[str, int]


class UDPFrame:
    """
    带完整头部信息的UDP数据帧类
    
    头部结构（32字节）：
    - 时间戳：8字节（float64，发送时的Unix时间戳）
    - 消息ID：4字节（uint32，唯一标识一个完整消息）
    - 总分片数：2字节（uint16，一个消息的总片数）
    - 当前分片索引：2字节（uint16，从1开始）
    - 数据类型：1字节（uint8，见TYPE_*常量）
    - 维度数量：1字节（uint8，数据维度，如列表/数组的维度）
    - 维度1：4字节（uint32，第一个维度大小）
    - 维度2：4字节（uint32，第二个维度大小，0表示未使用）
    - 维度3：4字节（uint32，第三个维度大小，0表示未使用）
    - 数据元素类型：1字节（uint16，仅numpy有效，见DTYPE_*常量）
    - 保留位：1字节（用于对齐和未来扩展）
    """
    
    def __init__(self):
        self.timestamp: float = 0.0          # 时间戳
        self.message_id: int = 0             # 消息ID
        self.total_fragments: int = 1        # 总分片数
        self.current_fragment: int = 1       # 当前分片索引
        self.data_type: int = TYPE_BYTES     # 数据类型
        self.dim_count: int = 0              # 维度数量
        self.dim1: int = 0                   # 维度1大小
        self.dim2: int = 0                   # 维度2大小
        self.dim3: int = 0                   # 维度3大小
        self.dtype: int = DTYPE_INT           # 数据元素类型（仅numpy有效）
        self.data: bytes = b''               # 数据内容
        
    def _pack_header(self) -> bytes:
        """将头部信息打包为字节流"""
        return struct.pack(
            '!dI HH BB III BB',
            self.timestamp,
            self.message_id,
            self.total_fragments,
            self.current_fragment,
            self.data_type,
            self.dim_count,
            self.dim1,
            self.dim2,
            self.dim3,
            self.dtype,
            0  # 保留位
        )
    
    def _unpack_header(self, header_bytes: bytes) -> None:
        """从字节流解析头部信息"""
        if len(header_bytes) != HEADER_SIZE:
            raise ValueError(f"头部大小错误，预期{HEADER_SIZE}字节，实际{len(header_bytes)}字节")
            
        unpacked = struct.unpack(
            '!dI HH BB III BB',
            header_bytes
        )
        
        self.timestamp = unpacked[0]
        self.message_id = unpacked[1]
        self.total_fragments = unpacked[2]
        self.current_fragment = unpacked[3]
        self.data_type = unpacked[4]
        self.dim_count = unpacked[5]
        self.dim1 = unpacked[6]
        self.dim2 = unpacked[7]
        self.dim3 = unpacked[8]
        self.dtype = unpacked[9]
        # unpacked[10] 是保留位，暂不使用
    
    def serialize(self, data: Any) -> List['UDPFrame']:
        """
        将数据序列化为一个或多个UDPFrame（自动分片）
        
        :param data: 要序列化的数据（支持bytes、str、list、np.ndarray）
        :return: UDPFrame对象列表
        """
        # 确定数据类型并设置元信息
        if isinstance(data, bytes):
            self.data_type = TYPE_BYTES
            serialized_data = data
            self.dim_count = 1
            self.dim1 = len(data)
            
        elif isinstance(data, str):
            self.data_type = TYPE_STRING
            serialized_data = data.encode('utf-8')
            self.dim_count = 1
            self.dim1 = len(data)
            
        elif isinstance(data, list):
            self.data_type = TYPE_LIST
            serialized_data = pickle.dumps(data)
            self.dim_count = 1
            self.dim1 = len(data)
            # 对于嵌套列表，可以扩展维度信息
            if data and isinstance(data[0], list):
                self.dim_count = 2
                self.dim2 = len(data[0])
                
        elif isinstance(data, np.ndarray):
            self.data_type = TYPE_NUMPY
            serialized_data = data.tobytes()
            self.dim_count = min(len(data.shape), 3)  # 最多记录3个维度
            data_type_str = str(data.dtype)
            if data_type_str == 'int32':
                self.dtype = DTYPE_INT
            elif data_type_str == 'int64':
                self.dtype = DTYPE_LONG
            elif data_type_str == 'float32':
                self.dtype = DTYPE_FLOAT
            elif data_type_str == 'float64':
                self.dtype = DTYPE_DOUBLE
            else:
                raise ValueError(f"不支持的numpy数据类型: {data.dtype}")
            if self.dim_count >= 1:
                self.dim1 = data.shape[0]
            if self.dim_count >= 2:
                self.dim2 = data.shape[1]
            if self.dim_count >= 3:
                self.dim3 = data.shape[2]
                
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        # 设置时间戳和计算分片
        self.timestamp = time.time()
        max_data_per_frame = MAX_FRAME_SIZE - HEADER_SIZE
        total_size = len(serialized_data)
        self.total_fragments = (total_size + max_data_per_frame - 1) // max_data_per_frame
        
        # 生成所有分片
        frames = []
        for i in range(self.total_fragments):
            frame = UDPFrame()
            # 复制头部信息
            frame.timestamp = self.timestamp
            frame.message_id = self.message_id
            frame.total_fragments = self.total_fragments
            frame.current_fragment = i + 1
            frame.data_type = self.data_type
            frame.dim_count = self.dim_count
            frame.dim1 = self.dim1
            frame.dim2 = self.dim2
            frame.dim3 = self.dim3
            frame.dtype = self.dtype
            # 设置分片数据
            start = i * max_data_per_frame
            end = start + max_data_per_frame
            frame.data = serialized_data[start:end]
            frames.append(frame)
            
        return frames
    
    def to_bytes(self) -> bytes:
        """将当前帧（头部+数据）转换为字节流"""
        return self._pack_header() + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'UDPFrame':
        """从字节流解析出UDPFrame对象"""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"数据长度小于头部大小，无法解析: {len(data)}字节")
            
        frame = cls()
        frame._unpack_header(data[:HEADER_SIZE])
        frame.data = data[HEADER_SIZE:]
        return frame
    
    def deserialize(self, all_fragments: List['UDPFrame']) -> Any:
        """
        将所有分片重组并反序列化为原始数据
        
        :param all_fragments: 同一消息的所有分片
        :return: 反序列化后的原始数据
        """
        # 验证所有分片属于同一消息
        message_ids = {f.message_id for f in all_fragments}
        if len(message_ids) != 1:
            raise ValueError("分片属于不同的消息，无法重组")
            
        # 按分片索引排序
        sorted_fragments = sorted(all_fragments, key=lambda x: x.current_fragment)
        
        # 拼接数据
        full_data = b''.join([f.data for f in sorted_fragments])
        timestamp = sorted_fragments[0].timestamp  # 可选：获取时间戳
        
        # 根据数据类型反序列化
        data_type = sorted_fragments[0].data_type
        
        if data_type == TYPE_BYTES:
            return full_data,timestamp
            
        elif data_type == TYPE_STRING:
            return full_data.decode('utf-8'),timestamp
            
        elif data_type == TYPE_LIST:
            return pickle.loads(full_data),timestamp
            
        elif data_type == TYPE_NUMPY:
            # 从头部获取形状信息
            dim_count = sorted_fragments[0].dim_count
            shape = []
            if dim_count >= 1:
                shape.append(sorted_fragments[0].dim1)
            if dim_count >= 2:
                shape.append(sorted_fragments[0].dim2)
            if dim_count >= 3:
                shape.append(sorted_fragments[0].dim3)
            # 尝试自动推断数据类型（这里简化处理，实际可根据需求扩展）
            try:
                if sorted_fragments[0].dtype in DTYPE_NAMES:
                    dtype_str = DTYPE_NAMES[sorted_fragments[0].dtype]
                    arr = np.frombuffer(full_data, dtype=dtype_str)
                    if arr.nbytes == len(full_data):
                        return arr.reshape(shape),timestamp
                # 如果失败，使用默认类型
                return np.frombuffer(full_data).reshape(shape),timestamp
            except Exception as e:
                raise ValueError(f"反序列化numpy数组失败: {str(e)}")
                
        else:
            raise ValueError(f"未知的数据类型: {data_type}")


class UDPQueue:
    """基于UDPFrame的队列实现，提供put和get方法"""
    
    def __init__(self, 
                 host: str = '0.0.0.0', 
                 port: int = 10050, 
                 is_server: bool = True,
                 buffer_size: int = 4096,
                 heartbeat_interval: int = 5,
                 heartbeat_timeout: int = 15):
        self.host = host
        self.port = port
        self.is_server = is_server
        self.buffer_size = buffer_size
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # 核心队列
        self._send_queue = Queue()      # 待发送数据队列（原始数据）
        self._recv_queue = Queue()      # 已接收数据队列（原始数据）
        # self._recv_queue()  # 限制接收队列大小，防止内存过大
        # 网络与状态
        self._socket: Optional[socket.socket] = None
        self._running = True
        self._received_eof = False
        
        # 分片重组缓冲区：{地址: {消息ID: [UDPFrame列表]}}
        self._reassembly_buffers: Dict[Address, Dict[int, List[UDPFrame]]] = {}
        
        # 消息ID生成器
        self._next_message_id = 0
        
        # 服务器模式：客户端管理
        self._clients: Dict[Address, float] = {}  # {地址: 最后活动时间}
        
        # 客户端模式：服务器地址
        self._server_addr = (host, port)
        
        # 工作线程
        self._recv_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self.alive = True
        self._exit_counter=0
        self._init_components()

    def _init_components(self) -> None:
        """初始化组件和线程"""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            if self.is_server:
                self._socket.bind((self.host, self.port))
                print(f"UDP队列(服务器)启动在 {self.host}:{self.port}")
                self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
                self._cleanup_thread.start()
            else:
                print(f"UDP队列(客户端)连接到 {self.host}:{self.port}")
                self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self._heartbeat_thread.start()
            
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            self.clean()
            raise

    def _get_next_message_id(self) -> int:
        """生成下一个消息ID"""
        msg_id = self._next_message_id
        self._next_message_id = (self._next_message_id + 1) % (2**32)  # 4字节范围
        return msg_id

    def _heartbeat_loop(self) -> None:
        """客户端心跳循环"""
        if self.is_server:
            return
            
        while self._running and not self._received_eof and self._socket:
            try:
                self._socket.sendto(ENQ_HEARTBEAT, self._server_addr)
            except Exception as e:
                if self._running:
                    print(f"心跳发送错误: {str(e)}")
            time.sleep(self.heartbeat_interval)
    def is_alive(self) -> bool:
        """检查队列是否仍然运行"""
        return self.alive
    def _recv_loop(self) -> None:
        """接收循环：处理帧重组和反序列化"""
        while self._running and not self._received_eof and self._socket:
            try:
                data, addr = self._socket.recvfrom(self.buffer_size)
                
                if not data:
                    continue
                
                # 处理EOF信号
                if data == EOF_SIGNAL:
                    self._received_eof = True
                    self._running = False
                    print("收到退出信号，准备关闭")
                    self.alive = False
                    break
                
                # 处理心跳包
                if data == ENQ_HEARTBEAT:
                    if self.is_server:
                        self._clients[addr] = time.time()
                        # print(self._clients)
                    continue
                
                # 解析UDPFrame
                try:
                    frame = UDPFrame.from_bytes(data)
                except Exception as e:
                    print(f"解析帧失败: {str(e)}")
                    continue
                
                # 服务器模式更新客户端活动时间
                if self.is_server:
                    self._clients[addr] = time.time()
                
                # 初始化缓冲区
                if addr not in self._reassembly_buffers:
                    self._reassembly_buffers[addr] = {}
                if frame.message_id not in self._reassembly_buffers[addr]:
                    self._reassembly_buffers[addr][frame.message_id] = []
                
                # 添加当前帧到缓冲区
                self._reassembly_buffers[addr][frame.message_id].append(frame)
                
                # 检查是否所有分片都已收到
                current_fragments = self._reassembly_buffers[addr][frame.message_id]
                if len(current_fragments) == frame.total_fragments:
                    # 反序列化完整数据
                    try:
                        original_data,timestamp = frame.deserialize(current_fragments)
                        # 放入接收队列
                        if self.is_server:
                            self._recv_queue.put((timestamp, original_data, addr))
                        else:
                            self._recv_queue.put((timestamp, original_data))
                    except Exception as e:
                        print(f"数据反序列化失败: {str(e)}")
                    
                    # 清理缓冲区
                    del self._reassembly_buffers[addr][frame.message_id]
                    if not self._reassembly_buffers[addr]:
                        del self._reassembly_buffers[addr]
            
            except Exception as e:
                if self._running:
                    print(f"接收错误: {str(e)}")
                time.sleep(0.1)
        self._exit_counter+=1
        if self._exit_counter>=3:
            raise RuntimeError("Queue线程退出")

    def _send_loop(self) -> None:
        """发送循环：处理序列化和分片发送"""
        # print('enter send loop')
        while self._running and not self._received_eof and self._socket:
            try:
                item = self._send_queue.get(timeout=1.0)
                if self.is_server:
                    # 服务器模式：item=(数据, 目标地址)
                    data, target_addr = item
                    # print(f'收到发送请求，目标地址: {target_addr}')
                else:
                    # 客户端模式：item=数据
                    data = item
                    target_addr = self._server_addr
                
                # 创建帧并序列化
                frame = UDPFrame()
                frame.message_id = self._get_next_message_id()
                fragments = frame.serialize(data)
                
                # 发送所有分片
                if self.is_server and target_addr is None:
                    for client in list(self._clients.keys()):
                        for frag in fragments:
                            self._socket.sendto(frag.to_bytes(), client)
                else:
                    for frag in fragments:
                        self._socket.sendto(frag.to_bytes(),target_addr)
                
                self._send_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                if self._running:
                    print(f"发送错误: {str(e)}")
                time.sleep(0.1)
        self._exit_counter+=1
        if self._exit_counter>=3:
            raise RuntimeError("Queue线程退出")

    def _cleanup_loop(self) -> None:
        """服务器清理超时客户端"""
        if not self.is_server:
            return
            
        while self._running and not self._received_eof:
            current_time = time.time()
            timeout_addrs = [
                addr for addr, last_time in self._clients.items()
                if current_time - last_time > self.heartbeat_timeout
            ]
            
            for addr in timeout_addrs:
                # 清理客户端相关的缓冲区
                if addr in self._reassembly_buffers:
                    del self._reassembly_buffers[addr]
                del self._clients[addr]
                print(f"客户端 {addr} 超时，已清理")
            
            time.sleep(self.heartbeat_interval)
        self._exit_counter+=1
        if self._exit_counter>=3:
            raise RuntimeError("Queue线程退出")

    def put(self, data: Any, target_addr: Optional[Address] = None) -> None:
        """
        向队列添加数据（公共方法）
        
        :param data: 要发送的数据（支持bytes、str、list、np.ndarray）
        :param target_addr: 服务器模式需指定目标客户端地址
        """
        if not self._running or self._received_eof:
            raise RuntimeError("UDP队列已关闭或收到退出信号")
        
        if self.is_server:
            if not target_addr:
                self._send_queue.put((data, None))
                # raise ValueError("服务器模式必须指定目标客户端地址")
            else:
                self._send_queue.put((data, target_addr))
        else:
            self._send_queue.put(data)

    def get(self, timeout: Optional[float] = None) -> Tuple[Optional[Any], Optional[Address]]:
        """
        从队列获取数据（公共方法）
        
        :param timeout: 超时时间(秒)
        :return: 服务器模式返回(data, client_addr)，客户端模式返回(data, None)
        """
        if self._received_eof:
            raise RuntimeError("UDP队列已收到退出信号")
            
            
        try:
            if self.is_server:
                timestamp, item, addr = self._recv_queue.get(timeout=timeout)
                self._recv_queue.task_done()
                return (timestamp, item, addr)
            else:
                timestamp, item = self._recv_queue.get(timeout=timeout)
                self._recv_queue.task_done()
                return (timestamp,item, None)
        except Empty:
            return (None, None, None)

    def clean(self) -> None:
        """清理资源并关闭队列"""
        # 服务器发送EOF信号
        if self.is_server and self._running and self._socket:
            try:
                for client_addr in list(self._clients.keys()):
                    self._socket.sendto(EOF_SIGNAL, client_addr)
                print("已向所有客户端发送退出信号")
            except Exception as e:
                print(f"发送退出信号错误: {str(e)}")
        
        # 停止所有线程
        self._running = False
        self._received_eof = True
        
        for thread in [self._recv_thread, self._send_thread, 
                      self._heartbeat_thread, self._cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        # 关闭socket
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                print(f"关闭socket错误: {str(e)}")
            self._socket = None
        
        # 清空队列和缓冲区
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
                self._send_queue.task_done()
            except Empty:
                break
        self._send_queue=None
        
        while not self._recv_queue.empty():
            try:
                self._recv_queue.get_nowait()
                self._recv_queue.task_done()
            except Empty:
                break
        self._recv_queue=None
        self._reassembly_buffers.clear()
        self._clients.clear()
        gc.collect()
        print("UDP队列已清理关闭")

    def get_active_clients(self) -> list[Address]:
        """服务器模式：获取当前活跃客户端列表"""
        if not self.is_server:
            raise RuntimeError("仅服务器模式支持此方法")
        return list(self._clients.keys())


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基于UDPFrame的队列测试')
    parser.add_argument('--server', action='store_true', help='运行服务器模式')
    parser.add_argument('--host', default='127.0.0.1', help='主机地址')
    parser.add_argument('--port', type=int, default=10050, help='端口号')
    args = parser.parse_args()
    
    if args.server:
        # 服务器模式测试
        q = UDPQueue(host=args.host, port=args.port, is_server=True)
        print("服务器启动，按Ctrl+C退出")
        
        try:
            while True:
                timestamp, data, addr = q.get(timeout=0.2)
                if data is not None and addr:
                    data_type = type(data).__name__
                    print(f"{timestamp} 收到客户端 {addr} 的{data_type}: {data}")
                    
                    # 根据数据类型回复
                    if isinstance(data, list):
                        response = [x * 2 if isinstance(x, (int, float)) else x for x in data]
                    elif isinstance(data, np.ndarray):
                        response = data
                    else:
                        response = f"已收到: {data}"
                        
                    q.put(response, target_addr=addr)
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n用户中断，服务器退出")
        finally:
            q.clean()
    
    else:
        # 客户端模式测试
        q = UDPQueue(host=args.host, port=args.port, is_server=False)
        print("客户端启动，按Ctrl+C退出")
        
        try:
            counter = 0
            while not q._received_eof:
                if counter % 1 == 0:  # 每4秒发送一次数据
                    if counter % 1 == 0 and counter > 0:
                        # 发送numpy数组
                        # data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
                        data = np.random.rand(3,1000).astype(np.float32)
                        print(f"发送numpy数组: \n{data.shape}")
                    elif counter % 2 == 0:
                        # 发送列表
                        data = [10, "hello", 3.14, True, [5, 6]]
                        print(f"发送列表: {data}")
                    elif counter % 3 == 0:
                        # 发送字符串
                        data = f"客户端消息 #{counter//4}"
                        print(f"发送字符串: {data}")
                    else:
                        # 发送字节流
                        data = b"raw bytes data"
                        print(f"发送字节流: {data}")
                        
                    q.put(data)
                
                # 接收服务器回复
                timestamp, data, _= q.get(timeout=0.2)
                if data is not None:
                    print(f"{timestamp} 收到服务器回复: {data}")
                
                counter += 1
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断，客户端退出")
        finally:
            if not q._received_eof:
                q.clean()
