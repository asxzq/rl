# 示例程序
import math
import time
from mapclient import MapClient

# 连接，room_name自己填，不同room_name互不干扰
room_name = 'my_room_name'
client = MapClient('http://172.17.135.162:8888/', room_name)
# client = MapClient('http://127.0.0.1:8888/', room_name)
client.connect()

# 初始化要显示的图标
markers = {
    'Drone1': {
        'id': 'Drone1',
        'type': 0,
        'x': 10,
        'y': 20,
        'z': -20,
        'vx': 2,
        'vy': 4,
        'vz': -4,
        'hdg': 120,
        'color': '#fff',
        'waypoints': [
            {'x': 0, 'y': 0, 'z': -50},
            {'x': -30, 'y': -40, 'z': -30},
        ],
    },
    'UAV1': {
        'id': 'UAV1',
        'type': 1,
        'x': 40,
        'y': -10,
        'z': -50,
        'color': '#fff',
    },
    '??': {
        'id': '??',
        'type': 2,
        'x': 80,
        'y': 5,
        'z': -30,
        'color': '#fff',
    },
    'Facility1': {
        'id': 'Facility1',
        'type': 3,
        'x': 40,
        'y': 40,
        'z': 0,
        'color': '#FF0',
    },
    'Facility2': {
        'id': 'Facility2',
        'icon': 'flag',
        'type': 3,
        'x': 80,
        'y': 60,
        'z': 0,
        'color': '#FF0',
    },
}

# 首先初始化更新全部图标
client.updateMarkers(markers)


# 接受右键指令信号
@client.on('order')
def on_order(coordinates, ids):
    print('接收到指令', coordinates, ids)
    # 如果是对Drone1发出的指令，则将Drone1路径点设为指令位置
    if 'Drone1' in ids:
        client.updateMarkersIncremental({
            'Drone1': {
                'waypoints': [
                    {'x': coordinates['x'], 'y': coordinates['y'], 'z': -30},
                ]
            }
        })
        # addMessage可以在页面右下输出信息
        client.addMessage('move to ' + str(coordinates))


try:
    radius = 50
    freq = 0.3
    t = 0
    dt = 0.5
    while True:
        time.sleep(dt)
        t += dt
        # 之后可以只更新有变化的部分
        client.updateMarkersIncremental({
            'Drone1': {
                'x': math.cos(t * freq) * radius + 100,
                'y': math.sin(t * freq * 1.414) * radius + 50,
                'z': -math.cos(t * freq * 1.114514) - 20,
                'vx': -math.sin(t * freq) * freq * radius,
                'vy': math.cos(t * freq * 1.414) * freq * 1.414 * radius,
                'vz': math.sin(t * freq * 1.114514) * freq * 1.114514,
                'hdg': t * 30,
            }
        })
except KeyboardInterrupt as e:
    pass
finally:
    # 最后必须要断开连接
    client.disconnect()
