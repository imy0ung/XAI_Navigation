"""
GradEclip 평가 결과 분석 스크립트

저장된 평가 결과를 로드하여 상세한 분석을 수행합니다.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_dir: str):
    """결과 파일 로드"""
    results_path = Path(results_dir)
    
    # JSON 파일 로드
    json_file = results_path / "metrics" / "detailed_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = []
    
    # CSV 파일 로드
    csv_file = results_path / "metrics" / "results.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(json_data)
    
    return df, json_data


def analyze_overall_performance(df):
    """전체 성능 분석"""
    total_episodes = len(df)
    successful_episodes = len(df[df['success'] == True])
    
    # 기본 통계
    sr = successful_episodes / total_episodes if total_episodes > 0 else 0.0
    
    # SPL 통계 (성공한 에피소드만)
    successful_df = df[df['success'] == True]
    if len(successful_df) > 0:
        avg_spl = successful_df['spl'].mean()
        std_spl = successful_df['spl'].std()
        max_spl = successful_df['spl'].max()
        min_spl = successful_df['spl'].min()
    else:
        avg_spl = std_spl = max_spl = min_spl = 0.0
    
    # 경로 길이 통계
    avg_path_length = df['path_length'].mean()
    std_path_length = df['path_length'].std()
    
    # 스텝 수 통계
    avg_steps = df['steps'].mean()
    std_steps = df['steps'].std()
    
    print("📊 전체 성능 분석")
    print("="*50)
    print(f"총 에피소드: {total_episodes}")
    print(f"성공 에피소드: {successful_episodes}")
    print(f"성공률 (SR): {sr:.4f} ({sr*100:.2f}%)")
    print(f"\nSPL 통계 (성공한 에피소드만):")
    print(f"  평균: {avg_spl:.4f} ± {std_spl:.4f}")
    print(f"  최고: {max_spl:.4f}")
    print(f"  최저: {min_spl:.4f}")
    print(f"\n경로 길이 통계:")
    print(f"  평균: {avg_path_length:.2f} ± {std_path_length:.2f} m")
    print(f"\n스텝 수 통계:")
    print(f"  평균: {avg_steps:.1f} ± {std_steps:.1f}")
    
    return {
        'sr': sr,
        'avg_spl': avg_spl,
        'std_spl': std_spl,
        'avg_path_length': avg_path_length,
        'avg_steps': avg_steps
    }


def analyze_by_object(df):
    """객체별 성능 분석"""
    print("\n🎯 객체별 성능 분석")
    print("="*50)
    
    object_stats = []
    for obj in df['target_object'].unique():
        obj_df = df[df['target_object'] == obj]
        total = len(obj_df)
        success = len(obj_df[obj_df['success'] == True])
        sr = success / total if total > 0 else 0.0
        
        # 성공한 에피소드의 SPL
        success_df = obj_df[obj_df['success'] == True]
        avg_spl = success_df['spl'].mean() if len(success_df) > 0 else 0.0
        
        # 평균 경로 길이와 스텝
        avg_path = obj_df['path_length'].mean()
        avg_steps = obj_df['steps'].mean()
        
        object_stats.append({
            'object': obj,
            'total': total,
            'success': success,
            'sr': sr,
            'avg_spl': avg_spl,
            'avg_path_length': avg_path,
            'avg_steps': avg_steps
        })
        
        print(f"{obj:15} | SR: {sr:.3f} ({success:2d}/{total:2d}) | "
              f"SPL: {avg_spl:.3f} | Path: {avg_path:5.1f}m | Steps: {avg_steps:5.1f}")
    
    return object_stats


def analyze_failure_types(df):
    """실패 유형 분석"""
    print("\n❌ 실패 유형 분석")
    print("="*50)
    
    result_counts = df['result'].value_counts()
    total = len(df)
    
    for result_type, count in result_counts.items():
        percentage = (count / total) * 100
        print(f"{result_type:20} | {count:3d} ({percentage:5.1f}%)")


def create_visualizations(df, output_dir):
    """시각화 생성"""
    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 성공률 by 객체
    plt.figure(figsize=(12, 6))
    object_sr = df.groupby('target_object')['success'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(object_sr)), object_sr.values)
    plt.xlabel('Target Object')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Target Object')
    plt.xticks(range(len(object_sr)), object_sr.index, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # 각 막대에 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / "success_rate_by_object.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SPL 분포 (성공한 에피소드만)
    successful_df = df[df['success'] == True]
    if len(successful_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(successful_df['spl'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('SPL')
        plt.ylabel('Frequency')
        plt.title('SPL Distribution (Successful Episodes)')
        plt.axvline(successful_df['spl'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {successful_df["spl"].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "spl_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 경로 길이 vs SPL
    if len(successful_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(successful_df['path_length'], successful_df['spl'], alpha=0.6)
        plt.xlabel('Path Length (m)')
        plt.ylabel('SPL')
        plt.title('Path Length vs SPL')
        
        # 추세선
        z = np.polyfit(successful_df['path_length'], successful_df['spl'], 1)
        p = np.poly1d(z)
        plt.plot(successful_df['path_length'], p(successful_df['path_length']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / "path_length_vs_spl.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 결과 분포 파이 차트
    plt.figure(figsize=(10, 8))
    result_counts = df['result'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(result_counts)))
    wedges, texts, autotexts = plt.pie(result_counts.values, labels=result_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Distribution of Episode Results')
    plt.tight_layout()
    plt.savefig(output_path / "result_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 시각화가 {output_path}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="GradEclip 평가 결과 분석")
    parser.add_argument("--results_dir", type=str, default="results_grad_eclip",
                       help="결과 디렉토리 경로")
    parser.add_argument("--visualize", action="store_true",
                       help="시각화 생성 여부")
    
    args = parser.parse_args()
    
    print("📈 GradEclip 평가 결과 분석 시작")
    print(f"📁 결과 디렉토리: {args.results_dir}")
    
    # 결과 로드
    df, json_data = load_results(args.results_dir)
    
    if len(df) == 0:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    # 분석 수행
    overall_stats = analyze_overall_performance(df)
    object_stats = analyze_by_object(df)
    analyze_failure_types(df)
    
    # 시각화 생성
    if args.visualize:
        try:
            create_visualizations(df, args.results_dir)
        except Exception as e:
            print(f"⚠️  시각화 생성 중 오류: {e}")
    
    # 요약 저장
    summary = {
        'overall_stats': overall_stats,
        'object_stats': object_stats,
        'total_episodes': len(df)
    }
    
    summary_path = Path(args.results_dir) / "metrics" / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 분석 요약이 {summary_path}에 저장되었습니다.")
    print("✅ 분석 완료!")


if __name__ == "__main__":
    main()