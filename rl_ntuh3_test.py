import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class NTUH3Env(gym.Env):
    def __init__(self, visualize: bool = False):
        super(NTUH3Env, self).__init__()

        self.n_joints = 3     # ntuh3的关节数 
        self.ee_name = 'lower_fts'  # 此處取lower_fts作末端执行器
        
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        #改为自己的model
        self.model = mujoco.MjModel.from_xml_path('./model/ntuh3_with_sabdpart.SLDASM/ntuh3_with_sabdpart.xml')
        self.data = mujoco.MjData(self.model)
        
        # 可视化设置
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 1.2])    # 调整观察点位置，設定觀察點的中心坐標
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.home_joint_pos = np.array([ 0.0, 0.0, 0.0 ], dtype=np.float32)
        
        self.goal_size = 0.03   # 目标点大小
        
        # 约束工作空间 
        # self.workspace = {
        #     'x': [-0.46, 0.50],
        #     'y': [0.03, 0.62],
        #     'z': [1.00, 1.99]
        # }

        #再縮小一下，放在左半球
        self.workspace = {
            'x': [0.1, 0.4],
            'y': [0.3, 0.5],
            'z': [1.1, 1.8]
        }

        
        # 动作空间与观测空间 輸出範圍[-1, 1] 是否轉換為實際角度由step中處理？
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)  

        # 7轴关节角度、目标位置
        # 3(Pos) + 3(Vel) + 3(Goal) + 3(EE Pos) + 3(Rel Dist) = 15
        self.obs_size = self.n_joints*2 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(self.n_joints, dtype=np.float32)
        self.goal_threshold = 0.03    # 误差门槛
        self.max_steps = 3000

    # 測試用
    # def _get_valid_goal(self) -> np.ndarray:
    #     # 直接生成一個隨機點，不檢查任何條件
    #     goal = self.np_random.uniform(
    #         low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
    #         high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
    #     )
    #     return goal.astype(np.float32)


    # 修改后 (训练版：加入简单的距离判断)
    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if 0.4 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """渲染目标点"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
    

        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()
        


        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()        
        
        obs = self._get_observation()
        self.start_t = time.time()

        self.current_step = 0  # <--- [新增] 重置步數
        
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:self.n_joints].copy().astype(np.float32)
        joint_vel = self.data.qvel[:self.n_joints].copy().astype(np.float32) # 新增
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)

        # [新增] 獲取末端執行器(手)的絕對位置
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)

        # [新增] 計算「向量差」：目標在哪裡？(相對於手)
        # 這對神經網絡來說是極其直接的指引
        diff_to_goal = self.goal - ee_pos

        # 5個部分拼接
        # return np.concatenate([joint_pos, self.goal, ee_pos, diff_to_goal])
        return np.concatenate([joint_pos, joint_vel, self.goal, ee_pos, diff_to_goal])
    


    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)

        # 1. 距離獎勵：改為負值 (Dense Reward)，距離越遠扣分越多
        # 這樣 Agent 為了少扣分，會想辦法靠近目標
        distance_reward = -1.0 * dist_to_goal

        # 2. 確保有時間懲罰 (Time Penalty)
        time_penalty = 0.1  # 每一步扣 1 分
    
        # 非线性距离奖励（保持不变）
        if dist_to_goal < self.goal_threshold:
            distance_reward += 300.0
        # elif dist_to_goal < 2*self.goal_threshold:
        #     distance_reward = 50.0
        # elif dist_to_goal < 3*self.goal_threshold:
        #     distance_reward = 10.0
        # else:
        #     distance_reward = 1.0 / (1.0 + dist_to_goal)

        # 计算起点到目标的向量及相关参数
        start_to_goal = self.goal - self.start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)
        linearity_reward = 0.0
        deviation_penalty = 0.0
        
        if start_to_goal_norm >= 1e-6:  # 起点和目标不重合时才计算直线相关奖励/惩罚
            # 计算当前位置到起点的向量
            start_to_current = ee_pos - self.start_ee_pos
            # 计算当前位置在“起点→目标”直线上的投影比例（限制在0~1，避免超出目标后惩罚）
            projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
            projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
            # 计算直线上的投影点，得到当前位置偏离直线的垂直距离
            projected_point = self.start_ee_pos + projection_ratio * start_to_goal
            linearity_error = np.linalg.norm(ee_pos - projected_point)  # 偏离直线的距离
            
            # 1. 直线接近奖励：离直线越近，奖励越高（非线性递增）
            linearity_reward = 0.5 / (1.0 + linearity_error)  # 系数8.0可根据重要性调整
            
            # 2. 远离趋势惩罚：检测“先靠近后远离”的行为
            # 初始化或更新历史最小偏离距离（跟踪最近点）
            if not hasattr(self, 'min_linearity_error'):
                self.min_linearity_error = np.inf  # 首次运行初始化
            if linearity_error < self.min_linearity_error:
                self.min_linearity_error = linearity_error  # 更近时更新最小值，无惩罚
            else:
                # 比最近点更远时，惩罚远离的程度（距离差越大，惩罚越重）
                deviation_penalty = 1.0 * (linearity_error - self.min_linearity_error)  # 系数3.0可调整

        # # 姿态约束：保持末端朝下（保持不变）
        # target_orient = np.array([0, 0, -1])
        # ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)   
        # dot_product = np.dot(ee_orient_norm, target_orient) 
        # angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        # orientation_penalty = 0.3 * angle_error
        
        # 动作相关惩罚（保持不变）
        action_diff = 1*(action - self.prev_action)   #
        smooth_penalty = 5 * np.linalg.norm(action_diff)
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)  

        # 碰撞惩罚（保持不变）
        contact_reward = 1.0 * self.data.ncon
        
        # 关节角度限制惩罚（保持不变）
        joint_penalty = 0.0
        for i in range(self.n_joints):
            min_angle, max_angle = self.model.jnt_range[:self.n_joints][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # 时间惩罚（保持不变）
        # time_penalty = 0.01
        
        # 总奖励：整合新的直线奖励和远离惩罚
        total_reward = (distance_reward 
                    + linearity_reward  # 新增：靠近直线的奖励
                    - time_penalty
                    - contact_reward 
                    - smooth_penalty 
                    # - orientation_penalty 
                    - joint_penalty 
                    - deviation_penalty)  # 新增：先近后远的惩罚
        
        # 更新上一步动作
        self.prev_action = action.copy()
        
        # return total_reward, dist_to_goal, angle_error
        return total_reward, dist_to_goal, 0.0


    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        truncated = False
        self.current_step += 1

        # 动作缩放
        joint_ranges = self.model.jnt_range[:self.n_joints]
        scaled_action = np.zeros(self.n_joints, dtype=np.float32)
        for i in range(self.n_joints):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 执行动作
        self.data.ctrl[:self.n_joints] = scaled_action
        mujoco.mj_step(self.model, self.data)

        # 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[:self.n_joints], action)
        terminated = False
        collision = False

        # 目标达成
        if dist_to_goal < self.goal_threshold:     
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        # collision = False
        
    
        if not terminated:
            # if time.time() - self.start_t > 20.0:

            if self.current_step >= self.max_steps:
                truncated = True  # 強制結束
                if not terminated: # 如果還沒成功
                    reward -= 10.0 # 可以給個超時懲罰
                # print(f"[超时] 时间过长，奖励减半")
                # terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, truncated, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 32,                                     # 并行环境数， 可以是24,32,64等
    total_timesteps: int = 40_000_000,                    # 本次训练的新增步数
    # model_save_path: str = "panda_ppo_reach_target",
    model_save_path="ntuh3_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    env = make_vec_env(
        env_id=lambda: NTUH3Env(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )

    
    if resume_from is not None:
        model = PPO.load(resume_from, env=env)  # 加载时需传入当前环境
    else:
        POLICY_KWARGS = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=1024,          
            batch_size=2048,       
            n_epochs=10,           
            gamma=0.99,
            learning_rate=2e-4,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device="auto",
            tensorboard_log="./tensorboard/ntuh3_reach_target/"
        )
    
    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)

    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    # model_path: str = "panda_ppo_reach_target",
    model_path: str = "ntuh3_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    env = NTUH3Env(visualize=True)
    model = PPO.load(model_path, env=env)
    
    record_gif = False
    frames = [] if record_gif else None
    render_scene = None  
    render_context = None 
    pixel_buffer = None 
    viewport = None
    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


# 測試環境物理表現的函數
def debug_env_physics():
    # 1. 初始化环境，开启可视化
    # 注意：请确保你的 PandaObstacleEnv 类中已经修改了 XML 路径和 n_joints
    env = NTUH3Env(visualize=True)
    
    print("=== 环境初始化完成 ===")
    print(f"动作空间维度: {env.action_space.shape}")
    print(f"观测空间维度: {env.observation_space.shape}")
    
    obs, _ = env.reset()
    
    # 2. 运行测试循环
    # 这里的 total_steps 可以设大一点，方便观察
    total_steps = 1000  
    
    print("=== 开始物理测试 (正弦波动作) ===")
    print("观察要点：\n1. 机械臂是否平滑移动？\n2. 是否有剧烈抖动？\n3. 终端打印的奖励值是否合理？")
    
    try:
        for i in range(total_steps):
            # --- 生成平滑动作 ---
            # 使用正弦波生成 -1 到 1 之间的平滑数值
            # 频率设慢一点 (i * 0.05)，方便观察物理表现
            sine_value = np.sin(i * 0.05)
            
            # 创建动作：所有关节都按照同一个正弦波运动，或者你可以给不同关节设不同相位
            # env.action_space.shape[0] 会自动获取你设置的关节数 (例如 6 或 7)
            n_joints = env.action_space.shape[0]
            action = np.ones(n_joints) * sine_value
            
            # --- 环境步进 ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- 打印调试信息 (每20步打一次，避免刷屏) ---
            if i % 20 == 0:
                # 打印前3个关节的角度值，看看是否与预期的一样在变化
                current_joints = obs[:3] 
                print(f"Step {i:4d} | Action: {sine_value:.2f} | Reward: {reward:.4f} | Joint[0-2]: {current_joints}")
            
            # 如果触发结束条件（比如碰到目标），重置
            if terminated or truncated:
                print("--- 回合结束，重置环境 ---")
                obs, _ = env.reset()
                
            # 控制渲染帧率，太快看不清
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n测试手动停止")
    except Exception as e:
        print(f"\n!!! 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("环境已关闭")


if __name__ == "__main__":
    delete_flag_file()
    # TRAIN_MODE = True  # 设为True开启训练模式
    TRAIN_MODE = False  # 设为False开启测试模式

    MODEL_PATH = "assets/model/rl_reach_target_checkpoint/ntuh3_ppo_reach_target"    
    # RESUME_MODEL_PATH = "assets/model/rl_reach_target_checkpoint/ntuh3_ppo_reach_target"
    RESUME_MODEL_PATH = None

    if TRAIN_MODE:
        train_ppo(
            n_envs=12,                
            total_timesteps=5_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            resume_from=RESUME_MODEL_PATH
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=10,
        )

# 运行调试环境物理表现的函数
# if __name__ == "__main__":
#     # 清理可能存在的 flag 文件
#     delete_flag_file()
    
#     # 运行调试模式
#     debug_env_physics()