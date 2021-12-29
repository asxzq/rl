


# 示例程序
import math, time, base64
from mapclient import MapClient

# 连接，room_name自己填，不同room_name互不干扰
room_name = 'default'
#client = MapClient('http://172.17.135.162:8888/', room_name)
client = MapClient('http://127.0.0.1:8888/', room_name)
client.connect()

try:

    # 嵌入图片
    # with open('D:/LZR/Pictures/-9lddQ16q-2mvwZcT3cSu3-u0.jpg.medium.jpg', 'rb') as f:
    #     img_b64 = 'data:image;base64,' + base64.b64encode(f.read()).decode('utf-8')

    # 初始化要显示的图标
    markers = {
        'Drone1' : {
            'id' : 'Drone1',
            'type' : 0,
            'x' : 10,
            'y' : 20,
            'z' : -20,
            'vx' : 2,
            'vy' : 4,
            'vz' : -4,
            'hdg' : 120,
            'color' : '#fff',
            'waypoints' : [
                { 'x' : 0, 'y' : 0, 'z' : -50 },
                { 'x' : -30, 'y' : -40, 'z' : -30 },
            ],
            'desc' : 'adfasdfsadfsaffasdf',
            'html' : '<b>bbb</b>',
            # 'img' : img_b64,
        },
        'UAV1' : {
            'id' : 'UAV1',
            'type' : 1,
            'x' : 40,
            'y' : -10,
            'z' : -50,
            'color' : '#fff',
        },
        '??' : {
            'id' : '??',
            'type' : 2,
            'x' : 80,
            'y' : 5,
            'z' : -30,
            'color' : '#fff',
        },
        'Facility1' : {
            'id' : 'Facility1',
            'type' : 3,
            'x' : 40,
            'y' : 40,
            'z' : 0,
            'color' : '#FF0',
        },
        'Facility2' : {
            'id' : 'Facility2',
            'icon' : 'flag',
            'type' : 3,
            'x' : 80,
            'y' : 60,
            'z' : 0,
            'color' : '#FF0',
        },
    }

    # 首先初始化更新全部图标
    client.updateMarkers(markers)

    buttons = {
        'button1' : {
            # 使用文本
            'text' : 'button1',
        },
        'button2' : {
            # 使用svg图标
            'html' : '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><path d="M19 6.873a2 2 0 0 1 1 1.747v6.536a2 2 0 0 1-1.029 1.748l-6 3.833a2 2 0 0 1-1.942 0l-6-3.833A2 2 0 0 1 4 15.157V8.62a2 2 0 0 1 1.029-1.748l6-3.572a2.056 2.056 0 0 1 2 0l6 3.573z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        },
        'button3' : {
            # 使用svg图标
            'html' : '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h2"></path><path d="M17 12h2"></path><path d="M11 12h2"></path></g></svg>',
            # 可以绑定快捷键
            'key' : 'e',
        },
    }
    client.updateButtons(buttons)

    # 接受右键指令信号
    @client.on('order')
    def on_order(coordinates, ids):
        print('接收到指令', coordinates, ids)
        # 如果是对Drone1发出的指令，则将Drone1路径点设为指令位置
        if ('Drone1' in ids):
            client.updateMarkersIncremental({
                'Drone1' : {
                    'waypoints' : [
                        { 'x' : coordinates['x'], 'y' : coordinates['y'], 'z' : -30 },
                    ]
                }
            })
            # addMessage可以在页面右下输出信息
            client.addMessage('move to ' + str(coordinates))

    # 接受按钮事件
    @client.on('button')
    def on_button(button_id, ids):
        print('按钮被按下', button_id, ids)
        client.addMessage('按钮' + button_id)

    # 选择事件
    @client.on('select')
    def on_select(ids):
        print('选择更新', ids)

    radius = 50
    freq = 0.3
    t = 0
    dt = 0.5

    while (True):
        time.sleep(dt)
        t += dt
        # 之后可以只更新有变化的部分
        client.updateMarkersIncremental({
            'Drone1' : {
                'x' : math.cos(t * freq) * radius + 100,
                'y' : math.sin(t * freq * 1.414) * radius + 50,
                'z' : -math.cos(t * freq * 1.114514) - 20,
                'vx' : -math.sin(t * freq) * freq * radius,
                'vy' : math.cos(t * freq * 1.414) * freq * 1.414 * radius,
                'vz' : math.sin(t * freq * 1.114514) * freq * 1.114514,
                'hdg' : t * 30,
            }
        })

except KeyboardInterrupt as e:
    pass
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    #最后必须要断开连接
    client.disconnect()
