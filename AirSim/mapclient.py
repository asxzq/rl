import time
import socketio

g_target = 'web'


class MapClient:
    def __init__(self, server_address, room_name) -> None:
        self.sio = socketio.Client()
        self.server_address = server_address.strip('/')
        self.room_name = room_name
        self.timeout = 0.5  # sec
        self.interval = 0.005

        self.connected = False

        self.event_listeners = {}

        @self.sio.on('send_to')
        def message_received(msg):
            if 'func' in msg and 'params' in msg and type(msg['params']) == dict:
                if msg['func'] in self.event_listeners:
                    for func in self.event_listeners[msg['func']]:
                        func(**msg['params'])

        @self.sio.event
        def connect():
            def on_room_joined():
                print("MapClient connected to " + self.server_address)
                print(f"浏览器打开 : {self.server_address}/index.html?room={self.room_name}")
                self.connected = True
            self.sio.emit('join', {'room': room_name, 'role': 'py'}, callback=on_room_joined)

        @self.sio.event
        def connect_error(data):
            print("MapClient connection failed " + self.server_address)

        @self.sio.event
        def disconnect():
            print("MapClient disconnected")
            self.connected = False
    
    def connect(self):
        self.sio.connect(self.server_address)
        for i in range(int(self.timeout // self.interval)):
            if self.connected:
                break
            time.sleep(self.interval)
        if not self.connected:
            # timeout
            raise TimeoutError()

    def on(self, event_name):
        def deco(fn):
            if not (event_name in self.event_listeners):
                self.event_listeners[event_name] = []
            self.event_listeners[event_name].append(fn)
            return fn
        return deco

    def call(self, funcName, params_dict):
        if not self.connected:
            print('MapClient is not connected!')
            return
        finished_flag = [False]

        def finished():
            finished_flag[0] = True
        self.sio.emit('send_to', {
            'target': g_target,
            'func': funcName,
            'params': params_dict,
        }, callback=finished)
        for i in range(int(self.timeout // self.interval)):
            if finished_flag[0]:
                break
            time.sleep(self.interval)

    # 网页右下角打印消息
    def addMessage(self, message, duration=5):
        # print('addMessage')
        self.call('addMessage', {
            'message': message,
            'duration': duration,
        })
    
    # 更新所有标记
    def updateMarkers(self, markers):
        # print('updateMarkers')
        self.call('updateMarkers', {
            'markers': markers,
        })
    
    # 只更新部分标记
    def updateMarkersIncremental(self, markers):
        # print('updateMarkersIncremental')
        self.call('updateMarkersIncremental', {
            'markers' : markers,
        })

    def disconnect(self):
        self.sio.disconnect()


if __name__ == '__main__':
    pass
