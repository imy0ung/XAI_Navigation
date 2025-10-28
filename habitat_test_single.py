import time
import habitat_sim
from habitat_sim.utils import common as utils
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim import ActionSpec, ActuationSpec
from scipy.spatial.transform import Rotation as R
from mapping import Navigator
from vision_models.grad_eclip_model import GradEclipModel
#from vision_models.clip_dense import ClipModel
#from vision_models.grad_eclip_softmax_model import GradEclipSoftmaxModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from planning import Planning, Controllers
from config import *
from mapping import rerun_logger

from eval.dataset_utils.object_nav_utils import object_nav_gen


if __name__ == "__main__":
    config = load_config().Conf
    if type(config.controller) == HabitatControllerConf:
        pass
    else:
        raise NotImplementedError("Spot controller not suited for habitat sim")

    # GradEclipModel 사용 (feature_dim=1) - 단순화된 버전
    model = GradEclipModel(model_name="ViT-L/14", device="cuda", heatmap_scale_factor=1.0)
    print(f"🔧 GradEclipModel 초기화 완료 - 단순화된 버전 (gamma 보정 제거)")
    detector = YOLOWorldDetector(0.6)
    mapper = Navigator(model, detector, config)
    logger = rerun_logger.RerunLogger(mapper, False, "", debug=False) if config.log_rerun else None
    
    # 단일 목표 객체
    query_texts_for_clip = ["toilet"]  # 유사도 지도에 사용할 텍스트들
    detect_class = "toilet" 
    mapper.set_query([detect_class])
    # Grad-ECLIP의 경우 별도의 텍스트 임베딩 설정이 필요 없음
    # get_image_features()에서 직접 쿼리 텍스트를 받아서 heatmap 생성
    mapper.query_text = query_texts_for_clip
    mapper.previous_sims = None
    query = ", ".join(query_texts_for_clip)
    hm3d_path = "datasets/scene_datasets/hm3d"

    backend_cfg = habitat_sim.SimulatorConfiguration()   
    backend_cfg.scene_id = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    #"/val/00878-XB4GS9ShBRE/XB4GS9ShBRE.basis.glb"
    #"/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
    #"/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb"
    #"/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb"
    #"/val/00835-q3zU7Yy5E5s/q3zU7Yy5E5s.basis.glb"
    #"/val/00832-qyAac8rV8Zk/qyAac8rV8Zk.basis.glb"
    #"/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb"
    #"/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb" -> TV
    backend_cfg.scene_dataset_config_file = hm3d_path + "/hm3d_annotated_basis.scene_dataset_config.json"

    hfov = 90
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"

    rgb.hfov = hfov
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    res_x = 640
    res_y = 640
    rgb.resolution = [res_y, res_x]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.hfov = hfov
    depth.position = np.array([0, 0.88, 0])
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [res_y, res_x]

    hfov = np.deg2rad(hfov)
    focal_length = (res_x / 2) / np.tan(hfov / 2)
    principal_point_x = res_x / 2
    principal_point_y = res_y / 2
    K = np.array([
        [focal_length, 0, principal_point_x],
        [0, focal_length, principal_point_y],
        [0, 0, 1]
    ])

    agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
        move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
        turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
    ))
    agent_cfg.sensor_specifications = [rgb, depth]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

################ 최단 거리 구하기 ################
    # 시작 위치
    start_state = sim.get_agent(0).get_state()
    start_pos = np.array(start_state.position, dtype=np.float32)

    objs = [o for o in sim.semantic_scene.objects if detect_class in o.category.name().lower()]

    # 객체 중심 주변 링 샘플링으로 엔드포인트 후보를 생성 (네비게이션 가능한 지점만 채택)
    endpoints = []
    for obj in objs:
        center = np.array(obj.aabb.center, dtype=np.float32)
        for radius in [0.6, 0.8, 1.0, 1.2]:
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                candidate = center + np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)], dtype=np.float32)
                # 내비게이블 포인트로 스냅 및 필터링
                snapped = sim.pathfinder.snap_point(candidate)
                if (not np.isnan(snapped).any()) and sim.pathfinder.is_navigable(snapped) and sim.pathfinder.island_radius(snapped) >= 1.0:
                    endpoints.append(np.array(snapped, dtype=np.float32))

    # L* 계산
    if not endpoints:
        print(f"{detect_class} 후보 목표점을 찾지 못했습니다. L* 계산 불가")
        L_star = None
    else:
        L_star, _ = object_nav_gen.geodesic_distance(sim, start_pos, endpoints) # 최단거리 뽑기! 지오데식 방법으로
        print(f"최단거리 L*: {L_star:.3f} m")

################ 최단 거리 구하기 끝 ################

    # 경로 길이 누적 변수
    prev_pos = start_pos.copy()
    path_length = 0.0

    # 초기 360도 회전 시퀀스
    initial_sequence = ["turn_left"] * 28 * 2
    running = True
    autonomous = True
    controller = Controllers.HabitatController(sim, config.controller)

    while running:
        action = None
        if len(initial_sequence):
            action = initial_sequence[0]
            initial_sequence.pop(0)
            if not len(initial_sequence):
                mapper.one_map.reset_checked_map()
        elif autonomous:
            action = None
            state = sim.get_agent(0).get_state()
            orientation = state.rotation
            q0 = orientation.x
            q1 = orientation.y
            q2 = orientation.z
            q3 = orientation.w
            r = R.from_quat([q0, q1, q2, q3])
            pitch, yaw, roll = r.as_euler("yxz")
            yaw = pitch
            current_pos = np.array([[-state.position[2]], [-state.position[0]], [state.position[1]]])
            path = mapper.get_path()
            if path and len(path) > 1:
                path = Planning.simplify_path(np.array(path))
                path = path.astype(np.float32)
                for i in range(path.shape[0]):
                    path[i, :] = mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
                print(f"[DEBUG] 경로 추종 중... 경로 길이: {len(path)}, 목표: {path[-1]}")
                controller.control(current_pos, yaw, path)
                # HabitatController는 직접 시뮬레이터를 제어하므로 별도 액션 불필요
                observations = sim.get_sensor_observations()
                action = None  # sim.step() 건너뛰기
            else:
                print(f"[DEBUG] 경로 없음 또는 경로가 너무 짧음: {path}")
                # 경로가 없으면 제자리에서 회전하며 탐색
                action = "turn_left"
        if action and action != "enter_query":
            observations = sim.step(action)
        elif action is None and autonomous:
            # 컨트롤러가 직접 제어한 경우, 이미 observations를 받았으므로 계속 진행
            pass
        elif not autonomous:
            continue

        state = sim.get_agent(0).get_state()
        # 실제 이동거리 누적 (월드 좌표계)
        cur_world_pos = np.array(state.position, dtype=np.float32)
        path_length += np.linalg.norm(cur_world_pos - prev_pos)
        prev_pos = cur_world_pos
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
        mapper.set_camera_matrix(K)
        orientation = state.rotation
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        r = R.from_quat([q0, q1, q2, q3])
        pitch, yaw, roll = r.as_euler("yxz")
        r = R.from_euler("xyz", [0, 0, pitch])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        t = time.time()
        obj_found = mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1), observations["depth"].astype(np.float32),
            transformation_matrix)
        print("Time taken to add data: ", time.time() - t)
        
        # 히트맵 스케일 디버깅
        if hasattr(mapper.one_map, 'feature_map') and mapper.one_map.feature_map is not None:
            feature_map = mapper.one_map.feature_map
            if feature_map.shape[0] > 0:  # feature_dim > 0
                max_val = feature_map.max().item()
                min_val = feature_map.min().item()
                mean_val = feature_map.mean().item()
                print(f"📊 현재 히트맵 - Max: {max_val:.6f}, Min: {min_val:.6f}, Mean: {mean_val:.6f}")
                print(f"   스케일 팩터: {model.heatmap_scale_factor}")

        cam_x = pos[0, 0]
        cam_y = pos[1, 0]
        
        if logger:
            rr.log("camera/rgb", rr.Image(observations["rgb"]))
            rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                    observations["depth"].max() - observations["depth"].min())))
            rr.log("goal/text", rr.TextLog(query))
            logger.log_map()
            logger.log_pos(cam_y, cam_x)
        if obj_found:
            if L_star is not None and L_star > 0:
                spl = min(1.0, L_star / max(path_length, L_star))
                print(f"목표 객체({query})를 찾았습니다! SPL: {spl:.4f} (L*={L_star:.3f}, L={path_length:.3f})")
            else:
                print(f"목표 객체({query})를 찾았습니다! (L* 미계산으로 SPL 출력 불가)")
            break