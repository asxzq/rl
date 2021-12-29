import sys
import airsim
import multiUAV1
from mapclient import MapClient
import numpy as np
import math
import time
from scipy import linalg
if __name__ == "__main__":

    # 初始化mapclient
    room_name = 'default'
    # client = MapClient('http://172.17.135.162:8888/', room_name)
    client = MapClient('http://127.0.0.1:8888/', room_name)

    client.connect()

    try:
        # 嵌入图片
        # with open('D:/LZR/Pictures/-9lddQ16q-2mvwZcT3cSu3-u0.jpg.medium.jpg', 'rb') as f:
        #     img_b64 = 'data:image;base64,' + base64.b64encode(f.read()).decode('utf-8')

        ############################
        # map_init
        # 初始化要显示的图标
        markers = {
            'UAVc': {
                'id': 'UAVc',
                'type': 1,
                'x': 50,
                'y': 0,
                'z': 0,
                'color': '#f0f',
                'hdg': 90,
            },
            'UAV1': {
                'id': 'UAV1',
                'type': 1,
                'x': 50,
                'y': -6,
                'z': 0,
                'color': '#f0f',
                'hdg': 90,
            },
            'UAV2': {
                'id': 'UAV2',
                'type': 1,
                'x': 53,
                'y': 0,
                'z': 0,
                'color': '#f0f',
                'hdg': 90,
            },
            'UAV3': {
                'id': 'UAV3',
                'type': 1,
                'x': 56,
                'y': 0,
                'z': 0,
                'color': '#f0f',
                'hdg': 90,
            },
            'UAV4': {
                'id': 'UAV4',
                'type': 1,
                'x': 50,
                'y': -3,
                'z': 0,
                'color': '#f0f',
                'hdg': 90,
            },
            'T1': {
                'id': 'T1',
                'type': 1,
                'x': 78.900,
                'y': -25.900,
                'z': -3.600,
                'color': '#0ff',
            },
            'T2': {
                'id': 'T2',
                'type': 1,
                'x': 111.800,
                'y': -40.400,
                'z': -2.500,
                'color': '#0ff',
            },
            'T3': {
                'id': 'T3',
                'type': 1,
                'x': 0.300,
                'y': -233.700,
                'z': -1.600,
                'color': '#0ff',
            },
            'T4': {
                'id': 'T4',
                'type': 1,
                'x': 111.800,
                'y': 18.500,
                'z': -3.363,
                'color': '#0ff',
            },
        }

        # 首先初始化更新全部图标
        client.updateMarkers(markers)

        buttons = {
            'button1': {
                # 使用文本
                'text': 'button1',
            },
            'button2': {
                # 使用svg图标
                'html': '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><path d="M19 6.873a2 2 0 0 1 1 1.747v6.536a2 2 0 0 1-1.029 1.748l-6 3.833a2 2 0 0 1-1.942 0l-6-3.833A2 2 0 0 1 4 15.157V8.62a2 2 0 0 1 1.029-1.748l6-3.572a2.056 2.056 0 0 1 2 0l6 3.573z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
            },
            'button3': {
                # 使用svg图标
                'html': '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h2"></path><path d="M17 12h2"></path><path d="M11 12h2"></path></g></svg>',
                # 可以绑定快捷键
                'key': 'e',
            },
        }
        client.updateButtons(buttons)

        #############################
        # xzq_init 初始化控制器变量
        num_uav = 4
        m = multiUAV1.MultiUAVController(num=num_uav, alpha=0, scale=4)
        airsim_on = True
        name_uavc = 'UAVc'
        # clientc = airsim.MultirotorClient(ip='172.17.167.208')
        clientc = airsim.MultirotorClient()
        clientweapon = airsim.MultirotorClient()
        clientinfo = airsim.MultirotorClient()
        uavc_select = False
        clients = []
        for i in range(num_uav):
            # clients.append(airsim.MultirotorClient(ip='172.17.167.208'))
            clients.append(airsim.MultirotorClient())

        # 实现界面事件的回调
        # 接受右键指令信号
        @client.on('order')
        def on_order(coordinates, ids):
            print('接收到指令', coordinates, ids)
            # 如果是对Drone1发出的指令，则将Drone1路径点设为指令位置
            if 'UAVc' in ids:
                # addMessage可以在页面右下输出信息
                client.addMessage('move to ' + str(coordinates))

                state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
                UAVc_q = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
                UAVc_q += origin_c
                UAVc_q += start
                target = np.array([coordinates['x'], coordinates['y'], UAVc_q[2]], dtype=float)

                v = 2 / np.linalg.norm(target - UAVc_q) * (target - UAVc_q)
                '''
                # 会报错IOLoop is already running
                client.updateMarkersIncremental({
                    'UAVc': { 'waypoints': [
                            {'x': target[0], 'y': target[1], 'z': target[2]},
                        ]
                    }
                })'''
                clientc.moveByVelocityAsync(v[0], v[1], v[2], 1000, vehicle_name=name_uavc)


        # 接受按钮事件
        @client.on('button')
        def on_button(button_id, ids):
            print('按钮被按下', button_id, ids)
            if button_id == 'button1':
                print(m.c)
                m.make_c(num_uav, alpha=np.pi / 2, theta=0, scale=4)
                print(m.c)
            elif button_id == 'button2':
                print(m.c)
                m.c = np.array([
                    [4, 0, 0],
                    [2, 0, 0],
                    [-2, 0, 0],
                    [-4, 0, 0],
                ], dtype=float)
            elif button_id == 'button3':
                print("weapon on")
                #for i in range(num_uav):
                    # clients[i].client.call("fireWeapon")
                clientweapon.client.call('fireWeapon', 'UAVc')

        start = np.array([90, 15, -2.756], dtype=float)
        dict_T = {'T1', 'T2', 'T3', 'T4'}
        # 选择事件
        @client.on('select')
        def on_select(ids):
            print('选择更新', ids)
            weapon = False
            target = np.zeros((3,), dtype=float)
            for target_name in dict_T:
                if target_name in ids:
                    weapon = True
                    target[0] = markers[target_name]['x']
                    target[1] = markers[target_name]['y']
                    target[2] = markers[target_name]['z']
                    print('hit:', target_name)
                    break
            if weapon:
                state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
                UAVc_q = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
                UAVc_q += origin_c
                UAVc_q += start
                v = 2 / np.linalg.norm(target - UAVc_q) * (target - UAVc_q)
                print(v)
                clientc.moveByVelocityAsync(v[0], v[1], v[2], 1000, vehicle_name=name_uavc)


        radius = 50
        freq = 0.3
        t = 0
        dt = 0.03

        # 初始化无人机——起飞

        uav_name = []
        for i in range(num_uav):
            uav_name.append("UAV" + str(i + 1))

        origin_UAVs = np.array([
            [50, -6, -30],
            [53, 0, -30],
            [56, 0, -30],
            [50, -3, -30]
        ], dtype=float)
        origin_c = np.array([50, 0, -30], dtype=float)
        rotation_integrator_controller_T = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])

        futures = []
        for i in range(num_uav):
            clients[i].enableApiControl(True, uav_name[i])  # 获取控制权
            clients[i].armDisarm(True, uav_name[i])  # 解锁（螺旋桨开始转动）
            future = clients[i].takeoffAsync(vehicle_name=uav_name[i])
            futures.append(future)

        clientc.enableApiControl(True, name_uavc)
        clientc.armDisarm(True, name_uavc)
        future = clientc.takeoffAsync(vehicle_name=name_uavc)
        futures.append(future)
        for i in range(num_uav):
            futures[i].join()
        futures = []
        for i in range(num_uav):
            future = clients[i].moveToZAsync(-20, 10, vehicle_name=uav_name[i])
            futures.append(future)
        future = clientc.moveToZAsync(-20, 10, vehicle_name=name_uavc)
        futures.append(future)
        for i in range(num_uav):
            futures[i].join()

        m.o_mat = np.array([
            [80, 5, 60, 3],
            [100, -3, 60, 4],
        ], dtype=float)

        UAVs_q = np.zeros((num_uav, 3), dtype=float)
        UAVs_p = np.zeros((num_uav, 3), dtype=float)
        UAVc_q = np.zeros((3, ), dtype=float)
        UAVc_p = np.zeros((3, ), dtype=float)
        UAVc_p_ = np.zeros((3, ), dtype=float)
        clientc.moveByVelocityAsync(-2, 0, 0, 1000, vehicle_name=name_uavc)

        step = 0
        while (True):

            state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
            UAVc_q = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
            UAVc_p = np.array([state.linear_velocity.x_val, -state.linear_velocity.y_val, -state.linear_velocity.z_val])
            UAVc_q += origin_c @ rotation_integrator_controller_T
            for i in range(num_uav):
                state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
                UAVs_q[i] = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
                UAVs_p[i] = np.array([state.linear_velocity.x_val, -state.linear_velocity.y_val, -state.linear_velocity.z_val])
            UAVs_q += origin_UAVs @ rotation_integrator_controller_T

            # 输出一些参数
            if step % 10 == 0:
                dict_command = {}
                dict_command['UAVc'] = {
                        'x': UAVc_q[0] + start[0],
                        'y': -UAVc_q[1] + start[1],
                        'z': -UAVc_q[2] + start[2],
                        'vx': UAVc_p[0],
                        'vy': -UAVc_p[1],
                        'vz': -UAVc_p[2],
                    }
                for i in range(num_uav):
                    dict_command[uav_name[i]] = {
                            'x': UAVs_q[i, 0] + start[0],
                            'y': -UAVs_q[i, 1] + start[1],
                            'z': -UAVs_q[i, 2] + start[2],
                            'vx': UAVs_p[i, 0],
                            'vy': -UAVs_p[i, 1],
                            'vz': -UAVs_p[i, 2],
                        }
                client.updateMarkersIncremental(dict_command)

            info = clientinfo.client.call("getHitInfo", "UAVc")
            if info['hasHit']:
                print(info)

            a = m.calculate_u(UAVs_q, UAVs_p, UAVc_q, UAVc_p, 0)
            UAVs_p += a
            for i in range(num_uav):
                clients[i].moveByVelocityAsync(UAVs_p[i, 0], -UAVs_p[i, 1], -UAVs_p[i, 2], 2.0, vehicle_name=uav_name[i])

            
            time.sleep(dt)
            t += dt
            step += 1

    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 最后必须要断开连接
        client.disconnect()

