# GradEclip 모델을 사용하는 Actor 클래스

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr

# MON 모듈들
from vision_models.grad_eclip_model import GradEclipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from mapping import Navigator
from planning import Planning, Controllers
from eval.dataset_utils.object_nav_utils import object_nav_gen


class GradEclipActor:
    """GradEclipModel을 사용하는 객체 탐색 Actor"""
    
    def __init__(self, config, heatmap_scale_factor=1.0):
        # GradEclipModel 초기화 (CUDA 사용) - 단순화된 버전 (gamma 보정 제거)
        self.model = GradEclipModel(model_name="ViT-L/14", device="cuda", heatmap_scale_factor=heatmap_scale_factor)
        self.detector = YOLOWorldDetector(0.6)
        self.mapper = Navigator(self.model, self.detector, config)
        
        # 평가용 변수들
        self.path_length = 0.0
        self.start_pos = None
        self.prev_pos = None
        self.L_star = None
        self.target_object = None
        self.sim = None
        
        # 초기 회전 설정
        self.init = 36 * 2  # 360도 회전을 위한 스텝 수
        
        # 카메라 매트릭스 설정
        self.square = config.square_im if hasattr(config, 'square_im') else True
        hfov = 90 if self.square else 97
        res_x = 640
        res_y = 640 if self.square else 480
        hfov_rad = np.deg2rad(hfov)
        focal_length = (res_x / 2) / np.tan(hfov_rad / 2)
        principal_point_x = res_x / 2
        principal_point_y = res_y / 2
        K = np.array([
            [focal_length, 0, principal_point_x],
            [0, focal_length, principal_point_y],
            [0, 0, 1]
        ])
        self.mapper.set_camera_matrix(K)
        
        # 컨트롤러 설정
        self.controller = Controllers.HabitatController(None, config.controller)
        
        # 결과 저장 변수
        self.episode_results = []
        
    def reset(self, episode=None):
        """에피소드 시작 시 초기화"""
        self.mapper.reset()
        self.init = 36 * 2  # 360도 스캔을 위한 초기화
        
        if episode is not None:
            # 에피소드 정보 설정
            self.target_object = episode.goals[0].object_category if hasattr(episode, 'goals') else episode.obj_sequence[0]
            print(f"[INFO] 새로운 에피소드 시작 - 목표 객체: {self.target_object}")
            
            # 매퍼에 쿼리 설정
            self.mapper.set_query([self.target_object])
            self.mapper.query_text = [self.target_object]
            self.mapper.previous_sims = None
            
            # 시작 위치 설정 (시뮬레이터가 설정된 후)
            if self.sim is not None:
                self.start_pos = np.array(self.sim.get_agent(0).get_state().position, dtype=np.float32)
                self.prev_pos = self.start_pos.copy()
                self.path_length = 0.0
                
                # L* (최단 거리) 계산
                self._compute_L_star()
    
    def set_simulator(self, sim):
        """시뮬레이터 설정"""
        self.sim = sim
        
    def _compute_L_star(self):
        """목표 객체까지의 최단 거리 L* 계산"""
        try:
            if self.sim is None:
                print("[WARNING] 시뮬레이터가 설정되지 않았습니다.")
                self.L_star = None
                return
                
            objs = [o for o in self.sim.semantic_scene.objects 
                   if self.target_object in o.category.name().lower()]
            
            if not objs:
                print(f"[WARNING] {self.target_object} 객체를 씬에서 찾을 수 없습니다.")
                self.L_star = None
                return
                
            # 객체 중심 주변 링 샘플링으로 엔드포인트 후보 생성
            endpoints = []
            for obj in objs:
                center = np.array(obj.aabb.center, dtype=np.float32)
                for radius in [0.6, 0.8, 1.0, 1.2]:
                    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                        candidate = center + np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)], dtype=np.float32)
                        snapped = self.sim.pathfinder.snap_point(candidate)
                        if (not np.isnan(snapped).any()) and self.sim.pathfinder.is_navigable(snapped) and self.sim.pathfinder.island_radius(snapped) >= 1.0:
                            endpoints.append(np.array(snapped, dtype=np.float32))
            
            if endpoints:
                self.L_star, _ = object_nav_gen.geodesic_distance(self.sim, self.start_pos, endpoints)
                # inf 값 체크 및 처리
                if not np.isfinite(self.L_star) or self.L_star <= 0:
                    print(f"[WARNING] L* 값이 유효하지 않습니다: {self.L_star}")
                    self.L_star = None
                else:
                    print(f"[INFO] L* (최단 거리): {self.L_star:.3f} m")
            else:
                print(f"[WARNING] {self.target_object} 후보 목표점을 찾지 못했습니다.")
                self.L_star = None
                
        except Exception as e:
            print(f"[ERROR] L* 계산 중 오류 발생: {e}")
            self.L_star = None
    
    def act(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        """에이전트 액션 수행"""
        return_act = {}
        state = observations["state"]

        # 현재 위치 및 방향 계산
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))

        orientation = state.rotation
        q0, q1, q2, q3 = orientation.x, orientation.y, orientation.z, orientation.w
        r = R.from_quat([q0, q1, q2, q3])
        yaw, _, _1 = r.as_euler("yxz")

        # 변환 행렬 생성
        r = R.from_euler("xyz", [0, 0, yaw])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        
        # 맵 업데이트 및 객체 탐지
        obj_found = self.mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1),
            observations["depth"].astype(np.float32),
            transformation_matrix
        )
        
        # 경로 길이 업데이트
        if self.prev_pos is not None:
            cur_world_pos = np.array(state.position, dtype=np.float32)
            self.path_length += np.linalg.norm(cur_world_pos - self.prev_pos)
            self.prev_pos = cur_world_pos
        
        # 초기 360도 회전
        if self.init > 0:
            return_act['discrete'] = 'turn_left'
            self.init -= 1
            if self.init == 0:
                # 360도 스캔 완료 후 checked_map 리셋
                self.mapper.one_map.reset_checked_map()
                print("[INFO] 초기 360도 환경 스캔 완료")
            return return_act, False
        
        # 경로 계획 및 실행
        path = self.mapper.get_path()
        if isinstance(path, str):
            if path == "L":
                return_act['discrete'] = 'turn_left'
                return return_act, False
            elif path == "R":
                return_act['discrete'] = 'turn_right'
                return return_act, False
        
        if path and len(path) > 0:
            # 경로 단순화 및 미터 단위 변환
            path = Planning.simplify_path(np.array(path))
            path = np.array(path).astype(np.float32)
            
            # Rerun 로깅
            rr.log("map/path_simplified", 
                   rr.LineStrips2D(path, colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                                                         path.shape[0], axis=0)))
            
            for i in range(path.shape[0]):
                path[i, :] = self.mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
            
            # 연속 제어
            ang, lin = self.controller.control(pos, yaw, path, False)
            return_act['continuous'] = {}
            return_act['continuous']['linear'] = lin
            return_act['continuous']['angular'] = ang
            return return_act, obj_found
        else:
            # 경로가 없으면 전진
            return_act['discrete'] = 'move_forward'
            return return_act, False
    
    def set_query(self, query: str):
        """쿼리 설정"""
        self.mapper.set_query([query])
        self.target_object = query
    
    def get_metrics(self, success_override=None):
        """현재 에피소드의 평가 메트릭 반환 (기존 CLIP과 동일한 방식)"""
        metrics = {}
        
        # SPL 계산 (기존 CLIP과 정확히 동일한 방식)
        if self.L_star is not None and np.isfinite(self.L_star) and self.L_star > 0:
            # 기존 CLIP 공식: best_dist / max(best_dist, distance_traveled)
            spl = self.L_star / max(self.L_star, self.path_length)
        else:
            spl = 0.0
            
        # 성공 여부는 외부에서 전달받거나 기본값 사용
        if success_override is not None:
            is_success = success_override
        else:
            # 기본값: L*가 있고 path_length가 L*보다 작으면 성공으로 간주
            is_success = (self.L_star is not None and self.L_star > 0 and self.path_length <= self.L_star)
        
        # 실패한 경우 SPL = 0으로 설정 (기존 CLIP과 동일)
        if not is_success:
            metrics["spl"] = 0.0
        else:
            metrics["spl"] = spl
            
        metrics["success"] = is_success
        metrics["L_star"] = self.L_star if (self.L_star is not None and np.isfinite(self.L_star)) else 0.0
        metrics["path_length"] = self.path_length
        metrics["target_object"] = self.target_object
        
        return metrics
    
    def save_episode_result(self, result_type, episode_id):
        """에피소드 결과 저장"""
        metrics = self.get_metrics()
        metrics["result_type"] = result_type
        metrics["episode_id"] = episode_id
        self.episode_results.append(metrics)
        
        return metrics