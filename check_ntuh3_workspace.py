# import numpy as np
# import mujoco
# import os

# def check_robot_workspace():
#     # 1. 加载模型
#     model_path = './model/ntuh3_with_sabdpart.SLDASM/ntuh3_with_sabdpart.xml'
#     if not os.path.exists(model_path):
#         print(f"错误: 找不到模型文件 {model_path}")
#         return

#     model = mujoco.MjModel.from_xml_path(model_path)
#     data = mujoco.MjData(model)

#     # 2. 设置配置
#     # 请确认你的末端执行器名称 (根据你之前的代码是 'lower_fts')
#     ee_name = 'lower_fts' 
#     try:
#         ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
#     except:
#         print(f"错误: 在XML中找不到名为 '{ee_name}' 的body。请检查XML中的名称。")
#         return

#     n_joints = 3  # 你的是3轴
#     n_samples = 50000  # 采样次数，越多越准
    
#     print(f"正在进行 {n_samples} 次随机采样...")

#     # 用于记录最大最小值
#     min_xyz = np.array([np.inf, np.inf, np.inf])
#     max_xyz = np.array([-np.inf, -np.inf, -np.inf])

#     # 3. 循环采样
#     for i in range(n_samples):
#         # 随机生成合法的关节角度 (在 range 范围内)
#         qpos = np.zeros(model.nq)
#         for j in range(n_joints):
#             # 获取 XML 中定义的关节范围
#             jnt_range = model.jnt_range[j]
#             # 均匀采样
#             qpos[j] = np.random.uniform(jnt_range[0], jnt_range[1])

#         # 直接设置关节位置 (不做物理模拟，只做运动学计算)
#         data.qpos[:] = qpos
        
#         # 计算正运动学 (Forward Kinematics)
#         mujoco.mj_kinematics(model, data)

#         # 获取末端位置
#         ee_pos = data.body(ee_id).xpos
        
#         # 更新范围
#         min_xyz = np.minimum(min_xyz, ee_pos)
#         max_xyz = np.maximum(max_xyz, ee_pos)

#     # 4. 输出结果
#     print("\n" + "="*30)
#     print("  機械臂工作空間測量結果  ")
#     print("="*30)
#     print(f"X 軸範圍: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}]  (寬度: {max_xyz[0]-min_xyz[0]:.3f}m)")
#     print(f"Y 軸範圍: [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}]  (深度: {max_xyz[1]-min_xyz[1]:.3f}m)")
#     print(f"Z 軸範圍: [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}]  (高度: {max_xyz[2]-min_xyz[2]:.3f}m)")
#     print("\n建议在 Python 代码中设置如下 (略微收缩边界以保证安全):")
#     print("-" * 40)
    
#     # 为了保险，我们在测量极限上稍微往回缩 2cm (0.02m)，确保目标点容易达到
#     margin = 0.02 
#     print("self.workspace = {")
#     print(f"    'x': [{min_xyz[0]+margin:.2f}, {max_xyz[0]-margin:.2f}],")
#     print(f"    'y': [{min_xyz[1]+margin:.2f}, {max_xyz[1]-margin:.2f}],")
#     print(f"    'z': [{min_xyz[2]+margin:.2f}, {max_xyz[2]-margin:.2f}]")
#     print("}")
#     print("-" * 40)

# if __name__ == "__main__":
#     check_robot_workspace()


import numpy as np
import mujoco
import mujoco.viewer
import os
import time

def check_robot_workspace_visual():
    # 1. 加载模型
    model_path = './model/ntuh3_with_sabdpart.SLDASM/ntuh3_with_sabdpart.xml'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 2. 设置配置
    ee_name = 'lower_fts' 
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
    except:
        print(f"错误: 在XML中找不到名为 '{ee_name}' 的body。")
        return

    n_joints = 3
    # 稍微减少采样次数，因为可视化会变慢，或者你可以保持大数值但接受等待
    n_samples = 5000  
    
    print(f"正在进行 {n_samples} 次随机采样并可视化...")
    print("观察画面中的：\n - 红色小球: 当前末端位置\n - 蓝色透明框: 计算出的工作空间范围 (Workspace)")

    min_xyz = np.array([np.inf, np.inf, np.inf])
    max_xyz = np.array([-np.inf, -np.inf, -np.inf])

    # 3. 启动被动查看器 (Passive Viewer)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 调整一下相机视角
        viewer.cam.distance = 2.0
        viewer.cam.lookat = np.array([0, 0, 0.3])
        viewer.cam.elevation = -20

        for i in range(n_samples):
            # --- A. 随机采样 ---
            qpos = np.zeros(model.nq)
            for j in range(n_joints):
                jnt_range = model.jnt_range[j]
                qpos[j] = np.random.uniform(jnt_range[0], jnt_range[1])

            # 设置关节并计算运动学
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            
            # 获取末端位置
            ee_pos = data.body(ee_id).xpos.copy()
            
            # 更新范围
            min_xyz = np.minimum(min_xyz, ee_pos)
            max_xyz = np.maximum(max_xyz, ee_pos)

            # --- B. 可视化绘制 ---
            # 只有当查看器开着的时候才绘制
            if viewer.is_running():
                # 1. 告诉 MuJoCo 我们要自己画点东西 (User Geoms)
                viewer.user_scn.ngeom = 0 
                
                # 2. 绘制当前末端点 (红色小球)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0], # 半径 2cm
                    pos=ee_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1]  # 红色
                )
                viewer.user_scn.ngeom += 1

                # 3. 绘制累计的工作空间包围盒 (蓝色半透明立方体)
                # 计算中心点和半长宽
                box_center = (min_xyz + max_xyz) / 2.0
                box_size = (max_xyz - min_xyz) / 2.0
                
                # 防止刚开始 min/max 是 inf 导致报错，做个简单判断
                if not np.isinf(box_center).any():
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=box_size,
                        pos=box_center,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 0.5, 1, 0.3] # 蓝色，透明度 0.3
                    )
                    viewer.user_scn.ngeom += 1

                # 4. 同步画面
                viewer.sync()
                
                # 稍微睡一点点时间，不然闪太快看不清动作 (可以注释掉以加速)
                time.sleep(0.002) 

    # 4. 输出结果
    print("\n" + "="*30)
    print("  機械臂工作空間測量結果  ")
    print("="*30)
    print(f"X 軸範圍: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}]")
    print(f"Y 軸範圍: [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}]")
    print(f"Z 軸範圍: [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}]")
    
    margin = 0.02
    print("\n请将以下代码复制到你的 Env __init__ 中:")
    print("-" * 40)
    print("self.workspace = {")
    print(f"    'x': [{min_xyz[0]+margin:.2f}, {max_xyz[0]-margin:.2f}],")
    print(f"    'y': [{min_xyz[1]+margin:.2f}, {max_xyz[1]-margin:.2f}],")
    print(f"    'z': [{min_xyz[2]+margin:.2f}, {max_xyz[2]-margin:.2f}]")
    print("}")
    print("-" * 40)
    
    # 采样结束后保持窗口不关闭，直到用户手动关闭，方便观察最终结果
    print("采样完成！请在查看器窗口查看最终的蓝色包围盒。按 Ctrl+C 退出。")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)

if __name__ == "__main__":
    check_robot_workspace_visual()