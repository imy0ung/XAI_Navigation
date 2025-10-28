"""
This module contains the Navigator class, which is responsible for the main functionality. It updates Onemap and uses it
for navigation and exploration.
"""
import time

from mapping import (OneMap, detect_frontiers, get_frontier_midpoint,
                     cluster_high_similarity_regions, find_local_maxima,
                     watershed_clustering, gradient_based_clustering, cluster_thermal_image,
                     Cluster, NavGoal, Frontier)

from planning import Planning
from vision_models.base_model import BaseModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from onemap_utils import monochannel_to_inferno_rgb, log_map_rerun
from config import Conf, load_config
from config import SpotControllerConf
from mobile_sam import sam_model_registry, SamPredictor

# numpy
import numpy as np

# typing
from typing import List, Optional, Set, Any, Union

# torch
import torch

# warnings
import warnings

# rerun
import rerun as rr

# cv2
import cv2


def closest_point_within_threshold(nav_goals: List[NavGoal], target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        nav_goals (List[NavGoal]): An array of potential nav points, where each point is retrieved by nav_goal.get_descr_point
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    points_array = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


class HistoricDetectData:
    def __init__(self, position: np.ndarray, action: str, other: Any = None):
        self.position = position
        self.other = other
        self.action = action

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.action}_{self.other}"
        return hash(string_repr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistoricDetectData):
            return NotImplemented
        return (np.array_equal(self.position, other.position) and
                self.action == other.action and
                self.other == other.other)


class CyclicDetectChecker:
    history: Set[HistoricDetectData] = set()

    def check_cyclic(self, position: np.ndarray, action: str, other: Any = None) -> bool:
        state_action = HistoricDetectData(position, action, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: str, other: Any = None) -> None:
        state_action = HistoricDetectData(position, action, other)
        self.history.add(state_action)


class HistoricData:
    def __init__(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None):
        self.position = position
        self.frontier_pt = frontier_pt
        self.other = other

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.frontier_pt}_{self.other}"
        return hash(string_repr)


class CyclicChecker:
    history: Set[HistoricData] = set()

    def check_cyclic(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> bool:
        state_action = HistoricData(position, frontier_pt, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> None:
        state_action = HistoricData(position, frontier_pt, other)
        self.history.add(state_action)


class Navigator:
    ##### 추가된 변수 : 확인 필요 #####
    query_text: List[str]  # Display/metadata of prompts
    # If using single-prompt mode, query_text_features is (1, C)
    # If using weighted multi-prompt mode, multi_text_features is (N, C) and multi_weights is (N,)
    query_text_features: torch.Tensor
    multi_text_features: Optional[torch.Tensor]
    multi_weights: Optional[torch.Tensor]
    negative_text_features: Optional[torch.Tensor]
    negative_weights: Optional[torch.Tensor]
    use_weighted_multi: bool
    od_target_class: Optional[str]
    points_of_interest: List[Cluster]
    blacklisted_nav_goals: List[np.ndarray]
    nav_goals: List[NavGoal]
    last_nav_goal: Union[NavGoal, None]
    ##### 추가된 변수 : 확인 필요 #####
    
    def __init__(self,
                 model: BaseModel,
                 detector: YOLOWorldDetector,
                 config: Conf
                 ) -> None:

        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.config = config

        # Models
        self.model = model
        self.detector = detector
        sam_model_t = "vit_t"
        sam_checkpoint = "weights/mobile_sam.pt"
        self.sam = sam_model_registry[sam_model_t](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")
        self.sam.eval()
        self.sam_predictor = SamPredictor(self.sam)

        self.one_map = OneMap(self.model.feature_dim, config.mapping, map_device="cpu")

        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.multi_text_features = None
        self.multi_weights = None
        self.negative_text_features = None
        self.negative_weights = None
        self.use_weighted_multi = False
        self.od_target_class = None
        self.previous_sims = None

        # Frontier and POIs
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []

        self.last_nav_goal = None
        self.last_pose = None
        self.saw_left = False
        self.saw_right = False

        self.first_obs = True
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.is_goal_path = False
        self.navigation_scores = np.zeros_like(self.one_map.navigable_map, dtype=np.float32)
        self.path = None
        self.is_spot = type(config.controller) == SpotControllerConf
        self.initializing = True
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0

        self.percentile_exploitation = config.planner.percentile_exploitation
        self.frontier_depth = int(config.planner.frontier_depth / self.one_map.cell_size)
        self.no_nav_radius = int(config.planner.no_nav_radius / self.one_map.cell_size)
        self.max_detect_distance = int(config.planner.max_detect_distance / self.one_map.cell_size)
        self.obstcl_kernel_size = int(config.planner.obstcl_kernel_size / self.one_map.cell_size)
        self.min_goal_dist = int(config.planner.min_goal_dist / self.one_map.cell_size)

        self.path_id = 0
        self.filter_detections_depth = config.planner.filter_detections_depth
        self.consensus_filtering = config.planner.consensus_filtering

        self.log = config.log_rerun
        self.allow_replan = config.planner.allow_replan
        self.use_frontiers = config.planner.use_frontiers
        self.allow_far_plan = config.planner.allow_far_plan

        # For the closed-vocabulary object detector, not needed for OneMap
        self.class_map = {}
        self.class_map["chair"] = "chair"
        self.class_map["tv_monitor"] = "tv"
        self.class_map["tv"] = "tv"
        self.class_map["plant"] = "potted plant"
        self.class_map["potted plant"] = "potted plant"
        self.class_map["sofa"] = "couch"
        self.class_map["couch"] = "couch"
        self.class_map["bed"] = "bed"
        self.class_map["toilet"] = "toilet"

    def reset(self):
        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.previous_sims = None
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.last_pose = None
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self.is_goal_path = False
        self.navigation_scores = np.zeros_like(self.one_map.navigable_map, dtype=np.float32)
        self.path = None
        self.path_id = 0
        self.initializing = True
        self.one_map.reset()
        self.first_obs = True
        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.points_of_interest = []
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        self.one_map.set_camera_matrix(camera_matrix)

    def set_query(self,
                  txt: List[str]
                  ) -> None:
        """
        Sets the query text
        :param txt: List of strings
        :return:
        """
        for t in txt:
            if t in self.class_map:
                txt[txt.index(t)] = self.class_map[t]
        if txt != self.query_text:
            print(f"Setting query to {txt}")
            self.query_text = txt
            self.query_text_features = self.model.get_text_features(["a " + self.query_text[0]]).to(
                self.one_map.map_device)
            # reset weighted mode
            self.use_weighted_multi = False
            self.multi_text_features = None
            self.multi_weights = None
            self.negative_text_features = None
            self.negative_weights = None
            self.od_target_class = self.query_text[0]
            self.previous_sims = None
            self.one_map.reset_checked_map()
            # Ensure OD only uses the target class (first element)
            self.detector.set_classes([self.query_text[0]])
            self.object_detected = False
            self.get_map(False)

######## 추가된 함수 #########
    def set_weighted_prompts(self,
                             prompts: List[str],
                             weights: List[float],
                             target_class: str,
                             negative_prompts: Optional[List[str]] = None,
                             negative_weights: Optional[List[float]] = None,
                             ) -> None:
        """Configure weighted multi-prompt mapping while keeping OD on a single target class.

        - prompts: list of concept strings (e.g., ["toilet", "bathroom", ...])
        - weights: list of floats same length as prompts; will be normalized to sum=1 over non-zero entries
        - target_class: OD target class (e.g., "toilet")
        - negative_prompts/weights: optional negative/gating prompts (off by default if None)
        """
        assert len(prompts) == len(weights), "prompts and weights must have same length"
        # Normalize weights over positive entries
        w = np.array(weights, dtype=np.float32)
        w = np.maximum(w, 0.0)
        if w.sum() > 0:
            w = w / w.sum()
        # Encode prompts in one batch
        text_inputs = [f"a {p}" for p in prompts]
        feats = self.model.get_text_features(text_inputs).to(self.one_map.map_device)  # (N, C)
        self.multi_text_features = feats
        self.multi_weights = torch.as_tensor(w, device=feats.device, dtype=feats.dtype)
        # Negative prompts (optional)
        if negative_prompts and len(negative_prompts) > 0:
            assert negative_weights is not None and len(negative_prompts) == len(negative_weights), (
                "negative_prompts and negative_weights must match lengths")
            nw = np.array(negative_weights, dtype=np.float32)
            nw = np.maximum(nw, 0.0)
            self.negative_text_features = self.model.get_text_features([f"a {p}" for p in negative_prompts]) \
                .to(self.one_map.map_device)
            self.negative_weights = torch.as_tensor(nw, device=feats.device, dtype=feats.dtype)
        else:
            self.negative_text_features = None
            self.negative_weights = None

        # Book-keeping / reset caches
        self.use_weighted_multi = True
        self.query_text = prompts
        self.query_text_features = None
        self.previous_sims = None
        self.one_map.reset_checked_map()
        # Set OD target only
        if target_class in self.class_map:
            target_class = self.class_map[target_class]
        self.od_target_class = target_class
        self.detector.set_classes([target_class])
        self.object_detected = False
        self.get_map(False)
################ 추가된 함수 끝 ############


    def get_path(self
                 ) -> Union[np.ndarray, str]:
        if not self.path:
            return None
        if not self.object_detected:
            if self.saw_left:
                return "L"
            if self.saw_right:
                return "R"
        return self.path[min(self.path_id, len(self.path)):]

    def compute_best_path(self,
                          start: np.ndarray) -> None:
        """
        Computes the best path from the start point to a point on a frontier, or a point of interest
        :param start: start point as [X, Y]
        :return:
        """
        self.path_id = 0
        if not self.object_detected:
            # We are exploring
            if len(self.nav_goals) == 0:
                if not self.initializing:
                    self.one_map.reset_checked_map()  # We need new points of interest
                    self.compute_frontiers_and_POIs(start[0], start[1])
            # If we still have no nav goals, we can't plan anything
            if len(self.nav_goals) == 0:
                return
            self.initializing = False
            self.is_goal_path = False
            self.path = None
            while self.path is None and len(self.nav_goals) > 0:
                best_idx = None
                if len(self.nav_goals) == 1:
                    top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[0].get_score()))
                else:
                    top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[1].get_score()))

                # We have a frontier and we need to consider following up on that
                curr_index = None
                if self.last_nav_goal is not None:
                    last_pt = self.last_nav_goal.get_descr_point()
                    for nav_id in range(len(self.nav_goals)):
                        if np.array_equal(last_pt, self.nav_goals[nav_id].get_descr_point()):
                            # frontier still exists!
                            curr_index = nav_id
                            break
                    if curr_index is None:
                        closest_index = closest_point_within_threshold(self.nav_goals, last_pt,
                                                                       0.5 / self.one_map.cell_size)
                        if closest_index != -1:
                            curr_index = closest_index
                            # there is a close point to the previous frontier that we could consider instead
                    if curr_index is not None:
                        curr_value = self.nav_goals[curr_index].get_score()
                        if curr_value + 0.01 > self.last_nav_goal.get_score():
                            best_idx = curr_index
                if best_idx is None:
                    # Select the current best nav_goal, and check for cyclic
                    for nav_id in range(len(self.nav_goals)):
                        nav_goal = self.nav_goals[nav_id]
                        cyclic = self.cyclic_checker.check_cyclic(start, nav_goal.get_descr_point(), top_two_vals)
                        if cyclic:
                            continue
                        best_idx = nav_id
                        # rr.log("path_updates", rr.TextLog(f"Selected frontier or POI based on score {self.frontiers[best_idx, 2]}. Max score is {self.frontiers[0, 2]}"))

                        break
                # TODO We should check if the chosen waypoint is reachable via simple path planning!
                best_nav_goal = self.nav_goals[best_idx]
                self.cyclic_checker.add_state_action(start, best_nav_goal.get_descr_point(), top_two_vals)
                if isinstance(best_nav_goal, Frontier):
                    self.path = Planning.compute_to_goal(start, self.one_map.navigable_map & (
                            self.one_map.confidence_map > 0).cpu().numpy(),
                                                         (self.one_map.confidence_map > 0).cpu().numpy(),
                                                         best_nav_goal.get_descr_point(),
                                                         self.obstcl_kernel_size, 2)
                elif isinstance(best_nav_goal, Cluster):
                    self.path = Planning.compute_to_goal(start, self.one_map.navigable_map & (
                            self.one_map.confidence_map > 0).cpu().numpy(),
                                                         (self.one_map.confidence_map > 0).cpu().numpy(),
                                                         best_nav_goal.get_descr_point(),
                                                         # TODO we might want to consider all the points of the cluster!
                                                         self.obstcl_kernel_size, 4)
                if self.path is None:
                    # remove the nav goal from the list, we don't know how to reach it
                    self.nav_goals.pop(best_idx)
            if self.path is None:
                if not self.initializing:
                    if self.log:
                        rr.log("path_updates", rr.TextLog(f"Resetting checked map as no path found."))
                    self.one_map.reset_checked_map()
            if self.last_nav_goal is not None and not np.array_equal(self.last_nav_goal.get_descr_point(),
                                                                     best_nav_goal.get_descr_point()):
                self.stuck_at_nav_goal_counter = 0
            else:
                if self.last_pose is not None:
                    if self.path is not None:
                        if len(self.path) < 5 and self.last_pose[0] == start[0] and self.last_pose[1] == start[1]:
                            self.stuck_at_nav_goal_counter += 1
            if self.stuck_at_nav_goal_counter > 10:
                # We probably are trying to reach an unreachable goal, for instance a frontier to the void in habitat
                self.blacklisted_nav_goals.append(best_nav_goal.get_descr_point())
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Frontier at position {best_nav_goal.get_descr_point()[0]}"
                                                      f",{best_nav_goal.get_descr_point()[1]} invalid."))
            self.last_nav_goal = best_nav_goal

            if self.path:
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Computed path of length {len(self.path)}"))
                    # Convert path coordinates from (x,y) to (y,x) for visualization
                    path_viz = np.array(self.path)[:, [1, 0]]  # Swap x and y coordinates
                    rr.log("map/path", rr.LineStrips2D(path_viz, colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                                                                                   len(self.path), axis=0)))
        else:
            # We go to an object
            print(f"[DEBUG] 목표 객체로 경로 계산 중... 시작: {start}, 목표: {self.chosen_detection}")
            distance_to_goal = np.linalg.norm(start - self.chosen_detection)
            print(f"[DEBUG] 목표까지 거리: {distance_to_goal:.2f}, 최대 탐지 거리: {self.max_detect_distance}")
            
            if distance_to_goal < self.max_detect_distance:
                self.path = [start] * 5
                print(f"[DEBUG] 목표에 충분히 가까움! 정지")
                # We are close to the object, we don't need to move
                return
                
            self.path = Planning.compute_to_goal(start, self.one_map.navigable_map,
                                                 (self.one_map.confidence_map > 0).cpu().numpy(),
                                                 self.chosen_detection,
                                                 self.obstcl_kernel_size, self.min_goal_dist)
            self.is_goal_path = True
            
            if self.path and len(self.path) > 0:
                print(f"[DEBUG] 목표 객체로의 경로 계산 성공! 길이: {len(self.path)}")
                if self.log:
                    # Convert path coordinates from (x,y) to (y,x) for visualization
                    path_viz = np.array(self.path)[:, [1, 0]]  # Swap x and y coordinates
                    rr.log("map/path", rr.LineStrips2D(path_viz, colors=np.repeat(np.array([0, 255, 0])[np.newaxis, :],
                                                                                   len(self.path), axis=0)))
                    rr.log("path_updates",
                           rr.TextLog(f"Path to object {self.query_text[0]} of length {len(self.path)} computed."))
            else:
                print(f"[DEBUG] 목표 객체로의 경로 계산 실패!")
                self.object_detected = False
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"No path to object {self.query_text[0]} found."))

    def compute_frontiers_and_POIs(self, px, py):
        """
        Computes the frontiers (at the border from fully explored to confidence > 0),
        and points of interest (high similarity regions within the fully explored, but not checked map)
        :return:
        """
        self.nav_goals = []
        if self.previous_sims is not None:
            # Compute the frontiers
            frontiers, unexplored_map, largest_contour = detect_frontiers(
                self.one_map.navigable_map.astype(np.uint8),
                self.one_map.fully_explored_map.astype(np.uint8),
                self.one_map.confidence_map > 0,
                int(1.0 * ((
                                   self.one_map.n_cells /
                                   self.one_map.size) ** 2)))

            # moreover we compute points of interest. These are high similarity regions within the fully explored,
            # but not checked map
            # For that we make use of the cluster_high_similarity_regions function, and project the points to the
            # navigable map
            # previous_sims를 2D 스코어 맵으로 정규화
            adjusted_score = self.previous_sims.cpu().numpy()
            if adjusted_score.ndim >= 3:
                adjusted_score = np.squeeze(adjusted_score)
            if adjusted_score.ndim == 1:
                adjusted_score = adjusted_score.reshape(self.one_map.confidence_map.shape)
            adjusted_score = adjusted_score + 1.0  # only positive scores
            map_def = self.previous_sims.squeeze().numpy()
            normalized_map = (map_def - map_def.min()) / (map_def.max() - map_def.min())
            # TODO This will give us wrong cluster scores, we will need to adjust this to match the frontier scores!
            clusters = cluster_high_similarity_regions(normalized_map,
                                                       (self.one_map.confidence_map > 0.0).cpu().numpy()) # 유사도가 높은 지점을 찾고 clustering.
            # clusters = cluster_high_similarity_regions(normalized_map, map_def > 0.0)

            # 클러스터 후보를 nav_goals에 추가
            for cluster in clusters:
                cluster.compute_score(adjusted_score)
                if len(self.blacklisted_nav_goals) == 0 or not np.any(
                        np.all(cluster.get_descr_point() == self.blacklisted_nav_goals, axis=1)):
                    if ((largest_contour is None or cv2.pointPolygonTest(largest_contour, cluster.center.astype(float),
                                                                         measureDist=True) > -15.0) or
                        self.one_map.fully_explored_map[cluster.center[0], cluster.center[1]]) and \
                            (not self.one_map.checked_map[cluster.center[0], cluster.center[1]]):
                        self.nav_goals.append(cluster) # 클러스터 후보를 nav_goals에 추가
            if self.log:
                # 2D 스코어 맵과 동일한 크기의 배열 생성
                cluster_max_similarity = np.zeros_like(adjusted_score)

                # Fill each cluster with its maximum similarity value
                if len(clusters) > 0:  # clusters가 비어있지 않을 때만 처리
                    min_c = np.min([cluster.get_score() for cluster in clusters])
                    max_c = np.max([cluster.get_score() for cluster in clusters])
                    for cluster in clusters:
                        cluster_pts = cluster.points
                        score = (cluster.get_score() - min_c) / (max_c - min_c) # 0~1 사이의 값으로 정규화
                        # 맵 경계로 인덱스 클램프 (안전)
                        h, w = self.one_map.confidence_map.shape
                        xs = np.clip(cluster_pts[:, 0], 0, h - 1)
                        ys = np.clip(cluster_pts[:, 1], 0, w - 1)
                        cluster_max_similarity[xs, ys] = score
                    log_map_rerun(cluster_max_similarity, path="map/similarity_th2")

            if self.log:
                log_map_rerun(unexplored_map, path="map/unexplored")

            frontiers = [f[:, :, ::-1] for f in frontiers]  # need to flip coords for some reason
            adjusted_score_frontier = adjusted_score.copy()

            # set the score of the fully explored map to 0 for the frontiers
            valid_frontiers_mask = np.zeros((len(frontiers),), dtype=bool)

            for i_frontier, frontier in enumerate(frontiers):
                frontier_mp = get_frontier_midpoint(frontier).astype(np.uint32)
                score, n_els, best_reachable, reachable_area = Planning.compute_reachable_area_score(
                    frontier_mp,
                    (self.one_map.confidence_map > 0).cpu().numpy(),
                    adjusted_score_frontier,
                    self.frontier_depth)
                frontier_mp = np.round(frontier_mp)
                if len(self.blacklisted_nav_goals) == 0 or not np.any(
                        np.all(frontier_mp == self.blacklisted_nav_goals, axis=1)):
                    valid_frontiers_mask[i_frontier] = True
                    self.nav_goals.append(
                        Frontier(frontier_midpoint=frontier_mp, points=frontier, frontier_score=score))

            if self.log:
                if len(self.nav_goals) > 0:
                    pts = np.array([nav_goal.get_descr_point() for nav_goal in self.nav_goals])
                    scores = np.array([nav_goal.get_score() for nav_goal in self.nav_goals])
                    rr.log("map/frontiers",
                           rr.Points2D(pts, colors=np.flip(monochannel_to_inferno_rgb(scores), axis=-1),
                                       radii=[1] * pts.shape[0]))

            self.nav_goals = sorted(self.nav_goals, key=lambda x: x.get_score(), reverse=True) # nav_goals를 점수 기준으로 정렬

    def add_data(self,
                 image: np.ndarray,
                 depth: np.ndarray,
                 odometry: np.ndarray,
                 ) -> bool:
        """
        Adds data to the navigator
        :param image: RGB image of dimension [C, H, W]
        :param depth: depth image of dimension [H, W]
        :param odometry: 4x4 transformation matrix from camera to world
        :return: boolean indicating if the episode is over
        """
        odometry = odometry.astype(np.float32)
        x = odometry[0, 3]
        y = odometry[1, 3]
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])

        px, py = self.one_map.metric_to_px(x, y)
        if self.last_pose:
            if np.linalg.norm(np.array([px, py, yaw]) - np.array(self.last_pose)) < 0.01:
                if self.path is not None:
                    self.stuck_at_cell_counter += 1
            else:
                self.stuck_at_cell_counter = 0
        if self.stuck_at_cell_counter > 5:
            # we are stuck we need to add an obstacle right in front of us!
            dx = np.cos(yaw)
            dy = np.sin(yaw)

            # Round to nearest integer to get facing direction
            facing_dx = round(dx)
            facing_dy = round(dy)

            # Calculate coordinates of facing cell
            facing_px = px + facing_dx
            facing_py = py + facing_dy
            self.artificial_obstacles.append((facing_px, facing_py))

        if not self.one_map.camera_initialized:
            warnings.warn("Camera matrix not set, please set camera matrix first")
            return

        # detections = self.detector.detect(np.flip(image, axis=0))
        # Check if RGB or BGR correct?
        # TODO I think yolo wants rgb
        # detections = self.detector.detect(np.flip(image.transpose(1, 2, 0), axis=-1))
        detections = self.detector.detect(image.transpose(1, 2, 0))
        a = time.time()
        
        # GradEclipModel의 경우 쿼리 텍스트를 전달
        if hasattr(self.model, 'feature_dim') and self.model.feature_dim == 1:
            # Grad-ECLIP 모델: 쿼리 텍스트와 함께 heatmap 생성
            if self.query_text and len(self.query_text) > 0:
                query_text = self.query_text[0]  # 첫 번째 쿼리 텍스트 사용
                image_features = self.model.get_image_features(image, query_text)
            else:
                # 기본값 사용
                image_features = self.model.get_image_features(image, "toilet")
        else:
            # 기존 CLIP 모델: 기존 방식 사용
            image_features = self.model.get_image_features(image[np.newaxis, ...]).squeeze(0)
        
        b = time.time()
        self.one_map.update(image_features, depth, odometry, self.artificial_obstacles)
        c = time.time()
        self.get_map(False)
        d = time.time()
        detected_just_now = False
        start = np.array([px, py])
        old_path = self.path.copy() if self.path else self.path
        old_id = self.path_id
        if self.first_obs:
            self.one_map.confidence_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.one_map.checked_conf_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.first_obs = False
        # if detections.class_id.shape[0] > 0:
        last_saw_left = self.saw_left
        last_saw_right = self.saw_right
        self.saw_left = False
        self.saw_right = False
        if len(detections["boxes"]) > 0:
            # wants rgb
            self.sam_predictor.set_image(image.transpose(1, 2, 0))
            for area, confidence in zip(detections["boxes"], detections['scores']):
                if self.log:
                    rr.log("camera/detection", rr.Boxes2D(array_format=rr.Box2DFormat.XYXY, array=area))
                    rr.log("object_detections", rr.TextLog(f"Object {self.query_text[0]} detected"))

                # TODO Find free point in front of object
                chosen_detection = (
                    int((area[3] + area[1]) / 2), int((area[2] + area[0]) / 2))
                masks, _, _ = self.sam_predictor.predict(point_coords=None,
                                                         point_labels=None,
                                                         box=np.array(area)[None, :],
                                                         multimask_output=False, )
                # Project the points where the mask is one
                mask_ids = np.argwhere(masks[0] & (depth != 0))
                depth_detection = depth[chosen_detection[0], chosen_detection[1]]

                depths = depth[mask_ids[:, 0], mask_ids[:, 1]]

                if not self.filter_detections_depth or depth_detection < 2.5:
                    y_world = -(mask_ids[:, 1] - self.one_map.cx) * depths / self.one_map.fx
                    x_world = depths
                    r = np.array([[np.cos(yaw), -np.sin(yaw)],
                                  [np.sin(yaw), np.cos(yaw)]])
                    x_rot, y_rot = np.dot(r, np.stack((x_world, y_world)))
                    x_rot += odometry[0, 3]
                    y_rot += odometry[1, 3]

                    x_id = ((x_rot / self.one_map.cell_size)).astype(np.uint32) + \
                           self.one_map.map_center_cells[0].item()
                    y_id = ((y_rot / self.one_map.cell_size)).astype(np.uint32) + \
                           self.one_map.map_center_cells[1].item()

                    object_valid = True
                    # previous_sims를 2D 스코어 맵으로 정규화
                    adjusted_score = self.previous_sims.cpu().numpy()
                    if adjusted_score.ndim >= 3:
                        adjusted_score = np.squeeze(adjusted_score)
                    if adjusted_score.ndim == 1:
                        adjusted_score = adjusted_score.reshape(self.one_map.confidence_map.shape)
                    adjusted_score = adjusted_score + 1.0  # only positive scores
                    if self.log:
                        rr.log("map/proj_detect",
                               rr.Points2D(np.stack((x_id, y_id)).T, colors=[[0, 0, 255]], radii=[1]))
                        # log the segmentation mask as rgba
                        rr.log("camera", rr.SegmentationImage(masks[0].astype(np.uint8))
                               )
                    if self.consensus_filtering:
                        # adjusted_score의 차원을 확인하고 올바르게 인덱싱
                        if adjusted_score.ndim == 1:
                            # 1차원 배열인 경우 confidence_map과 차원을 맞춤
                            confidence_mask = self.one_map.confidence_map > 0
                            if confidence_mask.ndim == 2:
                                # confidence_map이 2차원인 경우 1차원으로 평탄화
                                confidence_mask_flat = confidence_mask.flatten()
                                adjusted_score_flat = adjusted_score
                            else:
                                confidence_mask_flat = confidence_mask
                                adjusted_score_flat = adjusted_score
                        else:
                            # 2차원 배열인 경우
                            confidence_mask = self.one_map.confidence_map > 0
                            confidence_mask_flat = confidence_mask.flatten()
                            adjusted_score_flat = adjusted_score.flatten()

                        top_10 = np.percentile(adjusted_score_flat[confidence_mask_flat],
                                               self.percentile_exploitation)
                        top_map = (adjusted_score > top_10).astype(np.uint8)

                        print(top_10)
                        # top_map과 confidence_map의 차원을 맞춤
                        if top_map.ndim == 1:
                            # 1차원인 경우 2차원으로 변환
                            top_map = top_map.reshape(self.one_map.confidence_map.shape)
                        
                        # confidence_map과 차원이 맞는지 확인
                        if top_map.shape == self.one_map.confidence_map.shape:
                            top_map[self.one_map.confidence_map == 0] = 0
                        else:
                            # 차원이 맞지 않는 경우 confidence_map을 평탄화하여 인덱싱
                            confidence_mask = self.one_map.confidence_map.flatten()
                            top_map_flat = top_map.flatten()
                            top_map_flat[confidence_mask == 0] = 0
                            top_map = top_map_flat.reshape(self.one_map.confidence_map.shape)
                        k = np.ones((7, 7), np.uint8)
                        top_map = cv2.dilate(top_map, k, iterations=1)
                        # log_map_rerun((adjusted_score > 1.0).astype(np.float32), path="map/similarity_th")
                        top_map_projections = top_map[x_id, y_id]
                        if not np.any(top_map_projections):
                            object_valid = False

                        if object_valid:
                            mask = top_map_projections
                            x_masked = x_id[mask == 1]
                            y_masked = y_id[mask == 1]
                            depths_masked = depths[mask == 1]
                            best = np.argmin(depths_masked)

                            if self.object_detected and object_valid:
                                # 이미 목표가 있는 경우, 새로운 목표가 더 좋을 때만 업데이트
                                try:
                                    current_score = adjusted_score[x_masked[best], y_masked[best]]
                                    existing_score = adjusted_score[self.chosen_detection[0], self.chosen_detection[1]]
                                    if current_score < existing_score * 1.1:
                                        object_valid = False
                                except (IndexError, TypeError):
                                    # 인덱스 오류 시 새로운 목표 사용
                                    pass
                            if object_valid:
                                # self.object_detected = True
                                self.chosen_detection = (x_masked[best], y_masked[best])
                                print(f"[DEBUG] 목표 설정: {self.chosen_detection}, 깊이: {depths_masked[best]:.2f}m")
                    else:
                        best = np.argmin(depths)
                        if self.object_detected:
                            # adjusted_score의 차원을 확인하고 올바르게 인덱싱
                            if adjusted_score.ndim == 1:
                                # 1차원 배열인 경우 평탄화된 인덱스 사용
                                idx_best = x_id[best] * adjusted_score.shape[0] + y_id[best]
                                idx_chosen = self.chosen_detection[0] * adjusted_score.shape[0] + self.chosen_detection[1]
                                if adjusted_score[idx_best] < adjusted_score[idx_chosen] * 1.1:
                                    object_valid = False
                            else:
                                # 2차원 배열인 경우 기존 방식 사용
                                if adjusted_score[x_id[best], y_id[best]] < \
                                        adjusted_score[self.chosen_detection[0], self.chosen_detection[1]] * 1.1:
                                    object_valid = False
                        if object_valid:
                            self.chosen_detection = (x_id[best], y_id[best])
                    if object_valid:
                        self.object_detected = True
                        print(f"[DEBUG] 객체 탐지 완료! 목표: {self.chosen_detection}")
                        self.compute_best_path(start)
                        if not self.path:
                            print(f"[DEBUG] 경로 계산 실패! 목표로의 경로를 찾을 수 없음")
                            self.object_detected = False
                            self.path = old_path
                            self.path_id = old_id
                        else:
                            print(f"[DEBUG] 목표로의 경로 계산 성공! 경로 길이: {len(self.path)}")
                            if self.log:
                                rr.log("path_updates",
                                       rr.TextLog(f"The object {self.query_text[0]} has been detected just now."))
                                rr.log("map/goal_pos",
                                       rr.Points2D([self.chosen_detection[1], self.chosen_detection[0]], colors=[[0, 255, 0]], radii=[1]))
        elif not self.object_detected:
            self.chosen_detection = None
            self.object_detected = False
        if not self.object_detected:
            if self.saw_left:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "L")
            elif self.saw_right:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "R")
        self.compute_frontiers_and_POIs(*self.one_map.metric_to_px(odometry[0, 3], odometry[1, 3]))
        e = time.time()
        if self.log:
            # previous_sims를 2D 스코어 맵으로 정규화
            adjusted_score = self.previous_sims.cpu().numpy()
            if adjusted_score.ndim >= 3:
                adjusted_score = np.squeeze(adjusted_score)
            if adjusted_score.ndim == 1:
                adjusted_score = adjusted_score.reshape(self.one_map.confidence_map.shape)
            adjusted_score = adjusted_score + 1.0  # only positive scores
            
            # adjusted_score의 차원을 확인하고 올바르게 인덱싱
            if adjusted_score.ndim == 1:
                # 1차원 배열인 경우 confidence_map과 차원을 맞춤
                confidence_mask = self.one_map.confidence_map > 0
                if confidence_mask.ndim == 2:
                    # confidence_map이 2차원인 경우 1차원으로 평탄화
                    confidence_mask_flat = confidence_mask.flatten()
                    adjusted_score_flat = adjusted_score
                else:
                    confidence_mask_flat = confidence_mask
                    adjusted_score_flat = adjusted_score
            else:
                # 2차원 배열인 경우
                confidence_mask = self.one_map.confidence_map > 0
                confidence_mask_flat = confidence_mask.flatten()
                adjusted_score_flat = adjusted_score.flatten()

            top_10 = np.percentile(adjusted_score_flat[confidence_mask_flat],
                                   self.percentile_exploitation)
            top_map = (adjusted_score > top_10).astype(np.uint8)
            
            # top_map이 2차원인지 확인하고 cv2.dilate 적용
            if top_map.ndim == 2:
                k = np.ones((3, 3), np.uint8)
                top_map = cv2.dilate(top_map, k, iterations=1)
            else:
                # 1차원인 경우 2차원으로 변환
                top_map = top_map.reshape(self.one_map.confidence_map.shape)
                k = np.ones((3, 3), np.uint8)
                top_map = cv2.dilate(top_map, k, iterations=1)
                
            # chosen_detection이 None이 아닌지 확인
            if self.chosen_detection is not None:
                if not top_map[self.chosen_detection[0], self.chosen_detection[1]]:
                    self.object_detected = False
                    rr.log("path_updates", rr.TextLog("Current path lost similarity."))
            else:
                # chosen_detection이 None인 경우 object_detected를 False로 설정
                self.object_detected = False
        # Compute the new path
        # TODO Make the thresholds and distances to object a parameter
        if self.object_detected:
            if np.linalg.norm(start - self.chosen_detection) <= self.max_detect_distance:
                self.object_detected = False
                return True
            if self.consensus_filtering and self.object_detected:
                # previous_sims를 2D 스코어 맵으로 정규화
                adjusted_score = self.previous_sims.cpu().numpy()
                if adjusted_score.ndim >= 3:
                    adjusted_score = np.squeeze(adjusted_score)
                if adjusted_score.ndim == 1:
                    adjusted_score = adjusted_score.reshape(self.one_map.confidence_map.shape)
                adjusted_score = adjusted_score + 1.0  # only positive scores

                # adjusted_score의 차원을 확인하고 올바르게 인덱싱
                if adjusted_score.ndim == 1:
                    # 1차원 배열인 경우 confidence_map과 차원을 맞춤
                    confidence_mask = self.one_map.confidence_map > 0
                    if confidence_mask.ndim == 2:
                        # confidence_map이 2차원인 경우 1차원으로 평탄화
                        confidence_mask_flat = confidence_mask.flatten()
                        adjusted_score_flat = adjusted_score
                    else:
                        confidence_mask_flat = confidence_mask
                        adjusted_score_flat = adjusted_score
                else:
                    # 2차원 배열인 경우
                    confidence_mask = self.one_map.confidence_map > 0
                    confidence_mask_flat = confidence_mask.flatten()
                    adjusted_score_flat = adjusted_score.flatten()

                top_10 = np.percentile(adjusted_score_flat[confidence_mask_flat],
                                       self.percentile_exploitation)
                top_map = (adjusted_score > top_10).astype(np.uint8)
                
                # top_map이 2차원인지 확인하고 cv2.dilate 적용
                if top_map.ndim == 2:
                    k = np.ones((7, 7), np.uint8)
                    top_map = cv2.dilate(top_map, k, iterations=1)
                else:
                    # 1차원인 경우 2차원으로 변환
                    top_map = top_map.reshape(self.one_map.confidence_map.shape)
                    k = np.ones((7, 7), np.uint8)
                    top_map = cv2.dilate(top_map, k, iterations=1)
                
                # chosen_detection이 None이 아닌지 확인
                if self.chosen_detection is not None:
                    if not top_map[self.chosen_detection[0], self.chosen_detection[1]]:
                        self.object_detected = False
                        rr.log("path_updates", rr.TextLog("Current path lost similarity."))
                else:
                    # chosen_detection이 None인 경우 object_detected를 False로 설정
                    self.object_detected = False
        if self.allow_replan:
            self.compute_best_path(start)
        if self.object_detected and len(self.path) < 3:
            self.object_detected = False
            return True
        self.last_pose = (px, py, yaw)

    def get_map(self,
                return_map=True
                ) -> Optional[np.ndarray]:
        """
        Returns the similarity map given the query text
        :return: map as numpy array
        """
        # GradEclipModel의 경우 이미 heatmap이 생성되어 있음
        if hasattr(self.model, 'feature_dim') and self.model.feature_dim == 1:
            # Grad-ECLIP 모델: feature_map 자체가 이미 heatmap
            if self.previous_sims is None:
                # 첫 번째 호출: feature_map을 그대로 사용
                similarity = self.one_map.feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
            else:
                # 업데이트된 부분만 반영
                mask = self.one_map.updated_mask
                if mask.max() == 0:
                    if return_map:
                        return self.previous_sims.cpu().numpy()
                    else:
                        return
                
                # 업데이트된 부분만 새로운 값으로 교체
                similarity = self.previous_sims.clone()
                
                # 마스크와 텐서의 차원을 맞춤
                if mask.ndim == 2:  # [H, W] 형태
                    # similarity는 [1, 1, H, W] 형태이므로 마스크를 평탄화
                    mask_flat = mask.flatten()
                    # feature_map도 평탄화하여 인덱싱
                    feature_flat = self.one_map.feature_map.reshape(-1, self.one_map.feature_map.shape[-1])
                    # similarity를 평탄화하여 업데이트
                    similarity_flat = similarity.view(1, -1)
                    similarity_flat[:, mask_flat] = feature_flat[mask_flat, :].permute(1, 0)
                    # 다시 원래 형태로 복원
                    similarity = similarity_flat.view(similarity.shape)
                else:
                    # 마스크가 이미 평탄화된 경우
                    similarity[:, mask] = self.one_map.feature_map[mask, :].permute(1, 0)
        else:
            # 기존 CLIP 모델: similarity 계산 필요
            # Guard conditions: allow weighted-multi mode without single query_text_features
            if not self.use_weighted_multi and self.query_text_features is None:
                raise ValueError("No query text set")
            if self.use_weighted_multi and (self.multi_text_features is None or self.multi_weights is None):
                raise ValueError("No weighted prompts set")
            map_features = self.one_map.feature_map  # [X, Y, F]
            mask = self.one_map.updated_mask
            if mask.max() == 0:
                if return_map:
                    return self.previous_sims.cpu().numpy()
                else:
                    return
            if self.previous_sims is not None:
                map_features = map_features[mask, :].permute(1, 0).unsqueeze(0)
            else:
                map_features = map_features.permute(2, 0, 1).unsqueeze(0)

            if self.use_weighted_multi and self.multi_text_features is not None:
                # Compute per-prompt similarity in one go: (B=1, H*W) x (N, C) -> (N, H, W) via looped einsum over N
                # Efficiently: stack prompts into (N, C), use einsum bcw, nc -> nw (per mask chunk)
                # Here map_features shape depends on cache; handle both shapes consistently.
                # Normalize per-prompt sims to [0,1] over updated region, then weighted sum.
                sims_list = []
                # Compute similarity per prompt with shared map_features
                # model.compute_similarity already handles (BCHW, BC) or (BCX, BC). We call it N times but text encoded once.
                for i in range(self.multi_text_features.shape[0]):
                    sim_i = self.model.compute_similarity(map_features, self.multi_text_features[i:i+1, :])
                    sims_list.append(sim_i)
                sims = torch.stack(sims_list, dim=0)  # (N, 1, H, W) or (N, 1, M)
                # Squeeze batch dim
                sims = sims.squeeze(1)
                # Normalize each prompt's map to [0,1] on valid region
                if sims.dim() == 3:
                    # (N, H, W)
                    if self.previous_sims is None:
                        valid_mask = torch.ones_like(sims[0], dtype=torch.bool)
                    else:
                        # When incremental, mask is 1D; fallback to full normalization for simplicity
                        valid_mask = torch.ones_like(sims[0], dtype=torch.bool)
                else:
                    # (N, M) flattened; build valid mask from mask
                    if self.previous_sims is None:
                        valid_mask = torch.ones_like(sims[0], dtype=torch.bool)
                    else:
                        valid_mask = torch.ones_like(sims[0], dtype=torch.bool)

                sims_norm = []
                for i in range(sims.shape[0]):
                    s = sims[i]
                    v = s[valid_mask]
                    v_min = torch.min(v)
                    v_max = torch.max(v)
                    s_norm = (s - v_min) / (v_max - v_min + 1e-6)
                    sims_norm.append(s_norm)
                sims_norm = torch.stack(sims_norm, dim=0)  # (N, ...)

                # Apply optional negatives (gating)
                if self.negative_text_features is not None and self.negative_weights is not None:
                    neg_sims = []
                    for i in range(self.negative_text_features.shape[0]):
                        ns = self.model.compute_similarity(map_features, self.negative_text_features[i:i+1, :])
                        ns = ns.squeeze(0)
                        nv = ns  # same normalization scheme could be applied; keep as-is by default (off)
                        neg_sims.append(nv)
                    neg_sims = torch.stack(neg_sims, dim=0)
                    # simple gating: subtract weighted negatives after sigmoid
                    gate = torch.clamp(1.0 - torch.sigmoid((neg_sims * self.negative_weights[:, None, None]).sum(dim=0)), 0.0, 1.0)
                else:
                    gate = None

                # Weighted sum
                w = self.multi_weights
                if sims_norm.dim() == 3:
                    fused = (sims_norm * w[:, None, None]).sum(dim=0)
                    if gate is not None:
                        fused = fused * gate
                    similarity = fused.unsqueeze(0)  # add batch dim back
                else:
                    fused = (sims_norm * w[:, None]).sum(dim=0)
                    if gate is not None:
                        fused = fused * gate
                    similarity = fused.unsqueeze(0)
            else:
                similarity = self.model.compute_similarity(map_features, self.query_text_features)

        if self.previous_sims is None:
            self.previous_sims = similarity
        else:
            # then, similarity is only updated where the mask is true, otherwise it is the previous similarity
            # 마스크와 텐서의 차원을 맞춤
            if mask.ndim == 2:  # [H, W] 형태
                # previous_sims는 [1, 1, H, W] 형태이므로 마스크를 평탄화
                mask_flat = mask.flatten()
                # previous_sims를 평탄화하여 업데이트
                previous_sims_flat = self.previous_sims.view(1, -1)
                # similarity도 평탄화
                if similarity.ndim == 3:  # [1, H, W] 형태
                    similarity_flat = similarity.view(1, -1)
                else:  # [1, 1, H, W] 형태
                    similarity_flat = similarity.view(1, -1)
                previous_sims_flat[:, mask_flat] = similarity_flat[:, mask_flat]
                # 다시 원래 형태로 복원
                self.previous_sims = previous_sims_flat.view(self.previous_sims.shape)
            else:
                # 마스크가 이미 평탄화된 경우
                self.previous_sims[:, mask] = similarity
        self.one_map.reset_updated_mask()
        if return_map:
            return self.previous_sims.cpu().numpy()
        else:
            return

    def get_confidence_map(self,
                           ) -> np.ndarray:
        """
          Returns the confidence map
          :return: map as numpy array
          """
        return self.one_map.confidence_map.cpu().numpy()


if __name__ == "__main__":
    from vision_models.clip_dense import ClipModel
    # from vision_models.yolov7_model import YOLOv7Detector
    from vision_models.trt_yolo_world_detector import TRTYOLOWorldDetector
    import matplotlib.pyplot as plt
    import cv2

    # Yolo World
    # from ultralytics import YOLOWorld, YOLO

    # yw_detector = YOLO("yolov8s-worldv2.engine")

    print("Am I doing this?")

    camera_matrix = np.array([[384.41534423828125, 0.0, 328.7698059082031],
                              [0.0, 384.0389404296875, 245.87942504882812],
                              [0.0, 0.0, 1.0]])
    rgb = cv2.imread("/home/spot/Finn/MON/test_images/pairs/rgb_1.png")
    rgb = rgb[:, :, ::-1]
    rgb = cv2.resize(rgb, (640, 480)).transpose(2, 0, 1)
    depth = cv2.imread("/home/spot/Finn/MON/test_images/pairs/depth_1.png")
    depth = depth.astype(np.float32) / 255.0 * 3.0
    depth = cv2.resize(depth, (640, 480))[:, :, 0]
    odom = np.eye(4)
    # entire forward pass test
    base_conf = load_config()
    mapper = Navigator(ClipModel("", True), TRTYOLOWorldDetector(base_conf.Conf.planner.yolo_confidence), base_conf.Conf)
    mapper.set_camera_matrix(camera_matrix)
    mapper.set_query(["A fridge"])
    mapper.add_data(rgb, depth, odom)
    # test image, depth, odometry
    a = time.time()
    for i in range(10):
        mapper.add_data(rgb, depth, odom)
        # yw_detections = yw_detector(rgb.transpose(1, 2, 0))
        # yw_detections[0].show()

    print(f"Entire map update: {(time.time() - a) / 10}")
    sims = mapper.get_map()
    plt.imshow(sims[0])
    plt.savefig("firstmap.png")
    plt.show()
