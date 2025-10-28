"""
GradEclip 모델 평가 시스템 (기존 CLIP 설정과 호환)

기존 eval_habitat.py와 동일한 설정 파일을 사용하여 
GradEclipModel과 CLIP 모델의 성능을 공정하게 비교할 수 있습니다.
"""

import argparse
import time
from pathlib import Path

from eval.grad_eclip_evaluator import GradEclipEvaluator
from config import load_eval_config


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="GradEclip 모델 객체 탐색 평가 (CLIP 호환)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/grad_eclip_eval_conf.yaml",
        help="평가 설정 파일 경로 (기존 CLIP과 동일한 형식)"
    )

    
    args = parser.parse_args()
    
    print("🚀 GradEclip 모델 평가 시작 (CLIP 호환 모드)")
    print(f"📁 설정 파일: {args.config}")
    print(f"📊 결과 저장 위치: results_grad_eclip_compatible/")
    
    try:
        # 설정 로드 (기존 CLIP과 동일한 방식)
        print("⚙️  설정 로드 중...")
        from spock import SpockBuilder
        from config import EvalConf, HabitatControllerConf, MappingConf, PlanningConf
        
        eval_config = SpockBuilder(EvalConf, HabitatControllerConf, MappingConf, PlanningConf,
                                  desc='GradEclip eval config.').generate()
        
        # GradEclip 스케일링 팩터 설정 확인
        if hasattr(eval_config.EvalConf, 'gradeclip_scale_factor'):
            scale_factor = eval_config.EvalConf.gradeclip_scale_factor
            print(f"📊 GradEclip 스케일링 팩터: {scale_factor}")
        else:
            scale_factor = 1.0  # 단순화된 기본값
            print(f"📊 GradEclip 스케일링 팩터: {scale_factor} (단순화된 기본값)")
        
        # 평가기 생성
        print("🔧 평가기 초기화 중...")
        evaluator = GradEclipEvaluator(eval_config.EvalConf, heatmap_scale_factor=scale_factor)
        
        # 결과 디렉토리 설정
        evaluator.results_path = "results_grad_eclip_compatible/"
        Path(evaluator.results_path).mkdir(parents=True, exist_ok=True)
        
        # 평가할 에피소드 수 제한 (Spock 설정 활용)
        total_episodes = len(evaluator.episodes)
        print(f"📝 전체 에피소드 수: {total_episodes}")
        
        # 에피소드 수 제한 (설정 파일에서 max_episodes 설정 가능)
        if hasattr(eval_config.EvalConf, 'max_episodes') and eval_config.EvalConf.max_episodes > 0:
            max_episodes = min(eval_config.EvalConf.max_episodes, total_episodes)
            evaluator.episodes = evaluator.episodes[:max_episodes]
            print(f"📝 평가 에피소드: 0~{max_episodes-1} (총 {len(evaluator.episodes)}개)")
        else:
            print(f"📝 모든 에피소드 평가 (총 {len(evaluator.episodes)}개)")
        
        # 설정 비교 출력
        print(f"\n📋 평가 조건:")
        print(f"   🎯 성공 판정 거리: {eval_config.EvalConf.max_dist}m")
        print(f"   📦 데이터셋: {eval_config.EvalConf.object_nav_path}")
        print(f"   🏠 씬 경로: {eval_config.EvalConf.scene_path}")
        print(f"   🔢 최대 스텝: {eval_config.EvalConf.max_steps}")
        
        # 평가 시작
        print("🏁 평가 시작...")
        start_time = time.time()
        
        results = evaluator.evaluate()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 결과 요약
        successful_results = [r for r in results if r.success]
        total_episodes = len(results)
        
        print(f"\n🎉 평가 완료!")
        print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
        
        if total_episodes > 0:
            sr = len(successful_results) / total_episodes
            
            # 기존 CLIP과 동일한 방식: 모든 에피소드의 SPL 포함 (실패한 것도 SPL=0으로)
            all_spls = [r.spl for r in results]
            avg_spl_all = sum(all_spls) / len(all_spls) if all_spls else 0.0
            
            # 성공한 에피소드만의 SPL (참고용)
            avg_spl_success = sum(r.spl for r in successful_results) / len(successful_results) if successful_results else 0.0
            
            print(f"\n📊 최종 결과 (기존 CLIP과 동일한 조건):")
            print(f"   🎯 성공률 (SR): {sr:.4f} ({len(successful_results)}/{total_episodes})")
            print(f"   🏃 평균 SPL (모든 에피소드): {avg_spl_all:.4f} (기존 CLIP과 동일)")
            print(f"   🏃 평균 SPL (성공한 에피소드만): {avg_spl_success:.4f}")
            
            # 기존 CLIP 결과와 비교할 수 있도록 포맷 통일
            print(f"\n📋 비교용 결과:")
            print(f"Overall success: {sr:.4f}")
            print(f"Average SPL: {avg_spl_all:.4f}")
        
        print(f"\n💾 결과 저장 위치: {evaluator.results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()