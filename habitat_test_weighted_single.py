import time
import habitat_sim
from habitat_sim.utils import common as utils
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim import ActionSpec, ActuationSpec
from scipy.spatial.transform import Rotation as R
from mapping import Navigator
import torch
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from planning import Planning, Controllers
from config import *
from mapping import rerun_logger

from eval.dataset_utils.object_nav_utils import object_nav_gen
import argparse
import sys


if __name__ == "__main__":
    # Pre-parse custom CLI args and strip them before Spock parses its own
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--prompts', type=str, default='toilet,bathroom,sink,bathtub')
    pre_parser.add_argument('--weights', type=str, default='0.8,0.2,0.0,0.0')
    pre_parser.add_argument('--target', type=str, default='toilet')
    pre_parser.add_argument('--neg_prompts', type=str, default='')
    pre_parser.add_argument('--neg_weights', type=str, default='')
    custom_args, remaining_argv = pre_parser.parse_known_args()
    # 기본 YAML 자동 주입: -c/--config 미지정 시 시뮬레이터 기본 세트 사용
    has_config_flag = any(a in ('-c', '--config') for a in remaining_argv)
    # base_conf_sim.yaml 내부에서 다른 yaml들을 include 하므로, 여기서는 base만 전달해야 중복이 없음
    default_cfgs = [
        '/onemap/config/mon/base_conf_sim.yaml',
    ]
    if has_config_flag:
        sys.argv = [sys.argv[0]] + remaining_argv
    else:
        sys.argv = [sys.argv[0], '-c'] + default_cfgs + remaining_argv

    config = load_config().Conf
    if type(config.controller) == HabitatControllerConf:
        pass
    else:
        raise NotImplementedError("Spot controller not suited for habitat sim")

    model = ClipModel("weights/clip.pth")
    detector = YOLOWorldDetector(0.3)
    mapper = Navigator(model, detector, config)
    logger = rerun_logger.RerunLogger(mapper, False, "", debug=False) if config.log_rerun else None
    
    def parse_csv(s: str) -> list:
        return [t.strip() for t in s.split(',') if t.strip() != '']

    prompts = parse_csv(custom_args.prompts)
    weights = [float(x) for x in parse_csv(custom_args.weights)]
    # 파이프라인을 habitat_test_single.py와 동일하게: 단일 임베딩 경로 사용
    # OD는 target만, CLIP은 가중 평균 임베딩 1개를 사용
    detect_class = custom_args.target if len(custom_args.target) > 0 else (prompts[0] if len(prompts) > 0 else "toilet")
    mapper.set_query([detect_class])
    # 가중치 정규화 및 임베딩 배치 인코딩
    if len(prompts) == 0:
        prompts = [detect_class]
    if len(weights) != len(prompts):
        weights = [1.0] * len(prompts)
    w = np.maximum(np.array(weights, dtype=np.float32), 0.0)
    if w.sum() > 0:
        w = w / w.sum()
    text_prompts = [f"a {t}" for t in prompts]
    text_feats_all = mapper.model.get_text_features(text_prompts).to(mapper.one_map.map_device)  # (N, C)
    fused = (text_feats_all * torch.as_tensor(w, device=text_feats_all.device, dtype=text_feats_all.dtype)[:, None]).sum(dim=0, keepdim=True)
    fused = torch.nn.functional.normalize(fused, dim=1)
    mapper.query_text = [", ".join(prompts)]
    mapper.query_text_features = fused
    mapper.previous_sims = None
    query = ", ".join(prompts)
    hm3d_path = "datasets/scene_datasets/hm3d"

    backend_cfg = habitat_sim.SimulatorConfiguration()   
    backend_cfg.scene_id = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
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

    # '해당 scene에 존재하는 모든 toilet 오브젝트 수집
    toilet_objs = [o for o in sim.semantic_scene.objects if "toilet" in o.category.name().lower()]

    # 객체 중심 주변 링 샘플링으로 엔드포인트 후보를 생성 (네비게이션 가능한 지점만 채택)
    endpoints = []
    for obj in toilet_objs:
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
        print("toilet 후보 목표점을 찾지 못했습니다. L* 계산 불가")
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
                controller.control(current_pos, yaw, path)
                observations = sim.get_sensor_observations()
        if action and action != "enter_query":
            observations = sim.step(action)
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