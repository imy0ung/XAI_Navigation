from spock import spock, SpockBuilder

from config import HabitatControllerConf, MappingConf, PlanningConf


@spock
class EvalConf:
    multi_object: bool
    max_steps: int
    max_dist: float
    max_episodes: int = 0  # 0이면 모든 에피소드, 양수면 해당 수만큼만
    log_rerun: bool
    is_gibson: bool
    controller: HabitatControllerConf
    mapping: MappingConf
    planner: PlanningConf
    object_nav_path: str
    scene_path: str
    use_pointnav: bool
    square_im: bool
    gradeclip_scale_factor: float = 1.0  # GradEclip 히트맵 스케일링 팩터


def load_eval_config():
    return SpockBuilder(EvalConf, HabitatControllerConf, PlanningConf, MappingConf,
                        desc='Default MON config.').generate()
