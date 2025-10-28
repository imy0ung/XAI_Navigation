# GradEclip 모델용 평가기

import os
import json
import numpy as np
import pandas as pd
import pickle
import pathlib
from typing import Dict, List
from tabulate import tabulate
from dataclasses import dataclass
import enum

# Habitat 관련
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# 평가 관련
from eval import get_closest_dist
from eval.grad_eclip_actor import GradEclipActor
from eval.dataset_utils import gibson_dataset
from eval.dataset_utils import hm3d_dataset
from eval.dataset_utils import hm3d_multi_dataset
from mapping import rerun_logger
import rerun as rr

# 설정
from config import EvalConf


class Result(enum.Enum):
    """평가 결과 타입"""
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4  # Out of Time
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6


@dataclass
class EpisodeMetrics:
    """에피소드별 평가 메트릭"""
    episode_id: str
    target_object: str
    result: Result
    spl: float
    path_length: float
    L_star: float
    steps: int
    success: bool


class GradEclipEvaluator:
    """GradEclip 모델을 사용한 객체 탐색 평가기"""
    
    def __init__(self, config: EvalConf, heatmap_scale_factor: float = 1.0):
        self.config = config
        self.heatmap_scale_factor = heatmap_scale_factor
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.is_gibson = config.is_gibson
        
        # 데이터 저장
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        
        # 시뮬레이터 및 Actor
        self.sim = None
        self.actor = GradEclipActor(config, heatmap_scale_factor)
        
        # 제어 관련
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.time_step = 1.0 / self.control_frequency
        
        # 결과 저장
        self.episode_metrics = []
        self.results_path = "results_grad_eclip/"
        
        # 데이터셋 로드
        self._load_dataset()
        
        # Rerun 로깅
        if self.log_rerun:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, False, "")
    
    def _load_dataset(self):
        """데이터셋 로드"""
        if self.is_gibson:
            dataset_info_file = str(pathlib.Path(self.object_nav_path).parent.absolute()) + "/val_info.pbz2"
            with open(dataset_info_file, 'rb') as f:
                self.dataset_info = pickle.load(f)
            self.episodes, self.scene_data = gibson_dataset.load_gibson_episodes(
                self.episodes, self.scene_data, self.dataset_info, self.object_nav_path)
        else:
            self.dataset_info = None
            if self.multi_object:
                self.episodes, self.scene_data = hm3d_multi_dataset.load_hm3d_multi_episodes(
                    self.episodes, self.scene_data, self.object_nav_path)
            else:
                self.episodes, self.scene_data = hm3d_dataset.load_hm3d_episodes(
                    self.episodes, self.scene_data, self.object_nav_path)
    
    def load_scene(self, scene_id: str):
        """씬 로드"""
        if self.sim is not None:
            self.sim.close()
            
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id
        
        if not self.is_gibson:
            backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        
        # 센서 설정
        hfov = 90
        res = 640
        
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [res, res]
        
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [res, res]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
        ))
        agent_cfg.sensor_specifications = [rgb, depth]
        
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        
        # Actor에 시뮬레이터 설정
        self.actor.set_simulator(self.sim)
        
        # 객체 정보 로드
        if self.scene_data[scene_id].objects_loaded:
            return
            
        if not self.is_gibson:
            self.scene_data = hm3d_dataset.load_hm3d_objects(
                self.scene_data, self.sim.semantic_scene.objects, scene_id)
        else:
            self.scene_data = gibson_dataset.load_gibson_objects(
                self.scene_data, self.dataset_info, scene_id)
    
    def execute_action(self, action: Dict):
        """액션 실행"""
        if 'discrete' in action.keys():
            # 이산 액션
            self.sim.step(action['discrete'])
        elif 'continuous' in action.keys():
            # 연속 액션
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )
            
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )
            
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )
            
            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)
    
    def evaluate(self) -> List[EpisodeMetrics]:
        """전체 평가 실행"""
        print(f"[INFO] GradEclip 모델 평가 시작 - 총 {len(self.episodes)}개 에피소드")
        
        success_count = 0
        success_per_obj = {}
        obj_count = {}
        results = []
        
        for n_ep, episode in enumerate(self.episodes):
            if n_ep in self.exclude_ids:
                continue
                
            print(f"\n[INFO] 에피소드 {n_ep+1}/{len(self.episodes)} 시작 - ID: {episode.episode_id}")
            
            # 씬 로드
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            
            # 에이전트 초기화
            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            
            # Actor 리셋
            self.actor.reset(episode)
            
            # 목표 객체 설정
            current_obj = episode.obj_sequence[0] if hasattr(episode, 'obj_sequence') else episode.goals[0].object_category
            
            if current_obj not in success_per_obj:
                success_per_obj[current_obj] = 0
                obj_count[current_obj] = 1
            else:
                obj_count[current_obj] += 1
            
            self.actor.set_query(current_obj)
            
            # Rerun 로깅 - 목표 객체 위치
            if self.log_rerun:
                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))
            
            # 에피소드 실행
            episode_result = self._run_episode(episode, current_obj)
            results.append(episode_result)
            
            # 성공 체크
            if episode_result.success:
                success_count += 1
                success_per_obj[current_obj] += 1
            
            # 중간 결과 출력
            if (n_ep + 1) % 10 == 0:
                current_sr = success_count / (n_ep + 1)
                avg_spl = np.mean([r.spl for r in results if r.spl > 0])
                print(f"[중간결과] SR: {current_sr:.3f}, 평균 SPL: {avg_spl:.3f}")
        
        # 최종 결과 계산 및 저장
        self.episode_metrics = results
        self._save_results()
        self._print_final_results(results, success_per_obj, obj_count)
        
        return results
    
    def _run_episode(self, episode, target_object) -> EpisodeMetrics:
        """단일 에피소드 실행"""
        steps = 0
        poses = []
        result_type = Result.FAILURE_OOT
        
        while steps < self.max_steps:
            # 관측 획득
            observations = self.sim.get_sensor_observations()
            observations['state'] = self.sim.get_agent(0).get_state()
            
            # 포즈 기록
            pose = np.zeros(4)
            pose[0] = -observations['state'].position[2]
            pose[1] = -observations['state'].position[0]
            pose[2] = observations['state'].position[1]
            
            # Yaw 계산
            orientation = observations['state'].rotation
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            yaw, _, _ = r.as_euler("yxz")
            pose[3] = yaw
            poses.append(pose)
            
            # Rerun 로깅
            if self.log_rerun:
                cam_x = -self.sim.get_agent(0).get_state().position[2]
                cam_y = -self.sim.get_agent(0).get_state().position[0]
                rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                self.logger.log_pos(cam_x, cam_y)
            
            # 액션 수행
            action, called_found = self.actor.act(observations)
            self.execute_action(action)
            
            if self.log_rerun:
                self.logger.log_map()
            
            # 객체 발견 체크
            if called_found:
                dist = get_closest_dist(
                    self.sim.get_agent(0).get_state().position[[0, 2]],
                    self.scene_data[episode.scene_id].object_locations[target_object],
                    self.is_gibson
                )
                
                if dist < self.max_dist:
                    result_type = Result.SUCCESS
                    print(f"[SUCCESS] 객체 발견! 거리: {dist:.3f}m")
                    break
                else:
                    # 탐지 위치 확인
                    pos = self.actor.mapper.chosen_detection
                    pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                    dist_detect = get_closest_dist(
                        [-pos_metric[1], -pos_metric[0]],
                        self.scene_data[episode.scene_id].object_locations[target_object],
                        self.is_gibson
                    )
                    
                    if dist_detect < self.max_dist:
                        result_type = Result.FAILURE_NOT_REACHED
                    else:
                        result_type = Result.FAILURE_MISDETECT
                    print(f"[FAILURE] 실제 거리: {dist:.3f}m, 탐지 거리: {dist_detect:.3f}m")
                    break
            
            # 진행 상황 출력
            if steps % 100 == 0:
                dist = get_closest_dist(
                    self.sim.get_agent(0).get_state().position[[0, 2]],
                    self.scene_data[episode.scene_id].object_locations[target_object],
                    self.is_gibson
                )
                print(f"[STEP {steps}] 목표까지 거리: {dist:.3f}m")
            
            steps += 1
        
        # 막힘 체크
        poses = np.array(poses)
        if result_type == Result.FAILURE_OOT and len(poses) >= 10:
            if np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                result_type = Result.FAILURE_STUCK
        
        # 모든 영역 탐색 완료 체크
        num_frontiers = len(self.actor.mapper.nav_goals)
        if result_type in [Result.FAILURE_STUCK, Result.FAILURE_OOT] and num_frontiers == 0:
            result_type = Result.FAILURE_ALL_EXPLORED
        
        # 메트릭 계산 (실제 성공 여부 전달)
        actual_success = (result_type == Result.SUCCESS)
        metrics = self.actor.get_metrics(success_override=actual_success)
        
        # 디렉토리 생성
        os.makedirs(f"{self.results_path}/trajectories", exist_ok=True)
        os.makedirs(f"{self.results_path}/state", exist_ok=True)
        
        # 궤적 저장
        np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv", poses, delimiter=",")
        
        # 상태 저장
        with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
            f.write(str(result_type.value))
        
        return EpisodeMetrics(
            episode_id=episode.episode_id,
            target_object=target_object,
            result=result_type,
            spl=metrics["spl"],
            path_length=metrics["path_length"],
            L_star=metrics["L_star"],
            steps=steps,
            success=result_type == Result.SUCCESS
        )
    
    def _save_results(self):
        """결과 저장"""
        # 디렉토리 생성
        os.makedirs(f"{self.results_path}/metrics", exist_ok=True)
        
        # JSON 형태로 저장
        results_data = []
        for metric in self.episode_metrics:
            results_data.append({
                "episode_id": metric.episode_id,
                "target_object": metric.target_object,
                "result": metric.result.name,
                "spl": metric.spl,
                "path_length": metric.path_length,
                "L_star": metric.L_star,
                "steps": metric.steps,
                "success": metric.success
            })
        
        with open(f"{self.results_path}/metrics/detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # CSV 형태로 저장
        df = pd.DataFrame(results_data)
        df.to_csv(f"{self.results_path}/metrics/results.csv", index=False)
        
        print(f"[INFO] 결과가 {self.results_path}에 저장되었습니다.")
    
    def _print_final_results(self, results: List[EpisodeMetrics], success_per_obj: Dict, obj_count: Dict):
        """최종 결과 출력"""
        total_episodes = len(results)
        successful_episodes = sum(1 for r in results if r.success)
        
        # 전체 성공률
        overall_sr = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        # SPL 계산 (기존 CLIP과 동일한 방식 - 모든 에피소드 포함)
        # inf, None, 음수 값 필터링
        all_spls = [r.spl for r in results if r.spl is not None and np.isfinite(r.spl) and r.spl >= 0]
        avg_spl = np.mean(all_spls) if all_spls else 0.0
        
        # 성공한 에피소드만의 SPL도 별도 계산
        successful_spls = [r.spl for r in results if r.success and r.spl is not None and np.isfinite(r.spl) and r.spl > 0]
        avg_successful_spl = np.mean(successful_spls) if successful_spls else 0.0
        
        # 결과 분포
        result_counts = {}
        for result_type in Result:
            result_counts[result_type.name] = sum(1 for r in results if r.result == result_type)
        
        print("\n" + "="*60)
        print("🎯 GradEclip 모델 평가 결과")
        print("="*60)
        print(f"총 에피소드: {total_episodes}")
        print(f"성공률 (SR): {overall_sr:.4f} ({successful_episodes}/{total_episodes})")
        print(f"평균 SPL (모든 에피소드): {avg_spl:.4f} (기존 CLIP과 동일)")
        print(f"평균 SPL (성공한 에피소드만): {avg_successful_spl:.4f}")
        
        if successful_spls:
            print(f"최고 SPL: {max(successful_spls):.4f}")
            print(f"최저 SPL: {min(successful_spls):.4f}")
        
        print(f"\n📊 결과 분포:")
        for result_type, count in result_counts.items():
            percentage = (count / total_episodes) * 100 if total_episodes > 0 else 0
            print(f"  {result_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 객체별 성공률:")
        obj_results = []
        for obj in success_per_obj.keys():
            obj_sr = success_per_obj[obj] / obj_count[obj] if obj_count[obj] > 0 else 0.0
            obj_results.append([obj, f"{obj_sr:.3f}", f"{success_per_obj[obj]}/{obj_count[obj]}"])
            
        if obj_results:
            print(tabulate(obj_results, headers=["객체", "성공률", "성공/전체"], tablefmt="grid"))
        
        print("="*60)