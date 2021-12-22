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
    room_name = 'my_room_name'
    client = MapClient('http://172.17.135.162:8888/', room_name)
    client.connect()

    try:
        # 嵌入图片
        # with open('D:/LZR/Pictures/-9lddQ16q-2mvwZcT3cSu3-u0.jpg.medium.jpg', 'rb') as f:
        #     img_b64 = 'data:image;base64,' + base64.b64encode(f.read()).decode('utf-8')

        ############################
        # map_init
        # 初始化要显示的图标
        markers = {
            '''
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
                'desc': 'adfasdfsadfsaffasdf',
                'html': '<b>bbb</b>',
                # 'img' : img_b64,
            },
            '''
            'UAVc': {
                'id': 'UAVc',
                'type': 1,
                'x': 50,
                'y': 0,
                'z': 0,
                'color': '#f0f',
            },
            'UAV1': {
                'id': 'UAV1',
                'type': 1,
                'x': 50,
                'y': -6,
                'z': 0,
                'color': '#f0f',
            },
            'UAV2': {
                'id': 'UAV2',
                'type': 1,
                'x': 53,
                'y': 0,
                'z': 0,
                'color': '#f0f',
            },
            'UAV3': {
                'id': 'UAV3',
                'type': 1,
                'x': 56,
                'y': 0,
                'z': 0,
                'color': '#f0f',
            },
            'UAV4': {
                'id': 'UAV4',
                'type': 1,
                'x': 50,
                'y': -3,
                'z': 0,
                'color': '#f0f',
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

        num_uav = 4
        m = multiUAV1.MultiUAVController(num=num_uav, alpha=0, scale=4)

        # 接受右键指令信号
        @client.on('order')
        def on_order(coordinates, ids):
            print('接收到指令', coordinates, ids)
            # 如果是对Drone1发出的指令，则将Drone1路径点设为指令位置
            if ('Drone1' in ids):
                client.updateMarkersIncremental({
                    'Drone1': {
                        'waypoints': [
                            {'x': coordinates['x'], 'y': coordinates['y'], 'z': -30},
                        ]
                    }
                })
                # addMessage可以在页面右下输出信息
                client.addMessage('move to ' + str(coordinates))

        # 接受按钮事件
        @client.on('button')
        def on_button(button_id, ids):
            print('按钮被按下', button_id, ids)
            if button_id == 'button1':
                print(m.c)
                m.make_c(num_uav, alpha=np.pi / 2, theta=0, scale=4)
                print(m.c)
            if button_id == 'button2':
                print(m.c)
                m.c = np.array([
                    [8, 0, 0],
                    [4, 0, 0],
                    [-4, 0, 0],
                    [-8, 0, 0],
                ], dtype=float)

        # 选择事件
        @client.on('select')
        def on_select(ids):
            print('选择更新', ids)

        radius = 50
        freq = 0.3
        t = 0
        dt = 0.5

        #############################
        # xzq_init
        airsim_on = True
        name_uavc = 'UAVc'
        clientc = airsim.MultirotorClient(ip='172.17.167.208')
        clients = []
        for i in range(num_uav):
            clients.append(airsim.MultirotorClient(ip='172.17.167.208'))

        uav_name = []
        for i in range(num_uav):
            uav_name.append("UAV" + str(i + 1))

        origin_UAVs = np.array([
            [50, -6, 0],
            [53, 0, 0],
            [56, 0, 0],
            [50, -3, 0]
        ], dtype=float)
        origin_c = np.array([50, 0, 0], dtype=float)
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

        UAVs_q_in_game_UAVs_origin = np.zeros((num_uav, 3), dtype=float)
        UAVs_q_in_game_Airsim_origin = np.zeros((num_uav, 3), dtype=float)
        UAVs_q_in_controller = np.zeros((num_uav, 3), dtype=float)

        UAVs_p_in_game = np.zeros((num_uav, 3), dtype=float)
        UAVs_p_in_controller = np.zeros((num_uav, 3), dtype=float)

        # 领航者相对领航者出生点的位置
        UAVc_q_in_game_UAVc_origin = np.zeros((3,), dtype=float)
        # 领航者相对于游戏原点的位置
        UAVc_q_in_game_Airsim_origin = np.zeros((3,), dtype=float)
        # 领航者在控制器中的位置
        UAVc_q_in_controller = np.zeros((3,), dtype=float)

        UAVc_p_in_game = np.zeros((3,), dtype=float)
        UAVc_p_in_controller = np.zeros((3,), dtype=float)
        # UAVc_p_ = np.zeros((3,), dtype=float)

        clientc.moveByVelocityAsync(-2, 0, 0, 1000, vehicle_name=name_uavc)

        while (True):

            state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
            UAVc_q_in_game_UAVc_origin = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
            UAVc_q_in_game_Airsim_origin = UAVc_q_in_game_UAVc_origin + origin_c
            UAVc_q_in_controller = UAVc_q_in_game_Airsim_origin @ rotation_integrator_controller_T
            UAVc_p_in_game = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
            UAVc_p_in_controller = UAVc_p_in_game @ rotation_integrator_controller_T
            for i in range(num_uav):
                state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
                UAVs_q_in_game_UAVs_origin[i] = np.array(
                    [state.position.x_val, state.position.y_val, state.position.z_val])
                UAVs_q_in_game_Airsim_origin = UAVs_q_in_game_UAVs_origin + origin_UAVs
                UAVs_p_in_game[i] = np.array(
                    [state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
            UAVs_q_in_controller = UAVs_q_in_game_Airsim_origin @ rotation_integrator_controller_T
            UAVs_p_in_controller = UAVs_p_in_game @ rotation_integrator_controller_T

            client.updateMarkersIncremental({
                'UAVc': {
                    'x': UAVc_q_in_game_Airsim_origin[0],
                    'y': UAVc_q_in_game_Airsim_origin[1],
                    'z': UAVc_q_in_game_Airsim_origin[2],
                    'vx': UAVc_p_in_game[0],
                    'vy': UAVc_p_in_game[1],
                    'vz': UAVc_p_in_game[2],
                    'hdg': t * 30,
                }
            })
            for i in range(num_uav):
                client.updateMarkersIncremental({
                    name_uavc[i]: {
                        'x': UAVs_q_in_game_Airsim_origin[i, 0],
                        'y': UAVs_q_in_game_Airsim_origin[i, 1],
                        'z': UAVs_q_in_game_Airsim_origin[i, 2],
                        'vx': UAVs_p_in_game[i, 0],
                        'vy': UAVs_p_in_game[i, 1],
                        'vz': UAVs_p_in_game[i, 2],
                        'hdg': t * 30,
                    }
                })

            a = m.calculate_u(UAVs_q_in_controller, UAVs_p_in_controller, UAVc_q_in_controller, UAVc_p_in_controller, 0)
            UAVs_p_in_controller += a
            UAVs_p_in_game = UAVs_p_in_controller @ rotation_integrator_controller_T
            for i in range(num_uav):
                clients[i].moveByVelocityAsync(UAVs_p_in_game[i, 0], UAVs_p_in_game[i, 1], UAVs_p_in_game[i, 2], 2.0,
                                               vehicle_name=uav_name[i])
            time.sleep(dt)
            t += dt

    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 最后必须要断开连接
        client.disconnect()

