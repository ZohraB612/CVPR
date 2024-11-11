import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np

def parse_analysis_file(file_path):
    """Parse a single analysis.txt file."""
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Extract metrics using specific patterns
    precision_match = [line for line in content.split('\n') if 'Average Precision:' in line][0]
    precision = float(precision_match.split(':')[1].strip())
    
    # Extract precision at different ranks
    p10_match = [line for line in content.split('\n') if 'Precision@10:' in line][0]
    p20_match = [line for line in content.split('\n') if 'Precision@20:' in line][0]
    p10 = float(p10_match.split(':')[1].strip())
    p20 = float(p20_match.split(':')[1].strip())
    
    # Extract correct retrievals
    correct_match = [line for line in content.split('\n') if 'Correct retrievals' in line][0]
    correct = int(correct_match.split(':')[1].strip().split()[0])
    
    return {
        'avg_precision': precision,
        'p10': p10,
        'p20': p20,
        'correct_retrievals': correct
    }

def compare_pca_results():
    """Compare results from different PCA component settings."""
    result_dirs = glob('results/pca_*')
    if not result_dirs:
        print("No PCA results found. Please run PCA experiments first.")
        return
        
    plt.figure(figsize=(15, 10))
    colors = ['b', 'g', 'r', 'c']
    markers = ['o', 's', '^', 'D']
    
    print(f"Found {len(result_dirs)} PCA result directories")
    
    query_dirs = ['query_3_1_s', 'query_17_1_s', 'query_6_1_s', 'query_9_1_s']
    query_names = ['building', 'face', 'sheep', 'street']
    
    results_data = []
    
    for dir in sorted(result_dirs):
        try:
            n_components = int(dir.split('_')[1])
            print(f"\nProcessing results for {n_components} components")
            
            query_results = []
            
            for query_dir, query_name in zip(query_dirs, query_names):
                analysis_file = os.path.join(dir, query_dir, 'analysis.txt')
                if os.path.exists(analysis_file):
                    metrics = parse_analysis_file(analysis_file)
                    query_results.append(metrics)
                    print(f"Loaded {query_name}: AP={metrics['avg_precision']:.3f}, "
                          f"P@10={metrics['p10']:.3f}, P@20={metrics['p20']:.3f}")
            
            if query_results:
                avg_metrics = {
                    'components': n_components,
                    'map': np.mean([r['avg_precision'] for r in query_results]),
                    'p10': np.mean([r['p10'] for r in query_results]),
                    'p20': np.mean([r['p20'] for r in query_results]),
                    'correct': np.mean([r['correct_retrievals'] for r in query_results])
                }
                results_data.append(avg_metrics)
                
        except Exception as e:
            print(f"Error processing directory {dir}: {str(e)}")
            continue
    
    # Sort by number of components
    results_data.sort(key=lambda x: x['components'])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: MAP vs Components
    components = [r['components'] for r in results_data]
    map_scores = [r['map'] for r in results_data]
    ax1.plot(components, map_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Mean Average Precision vs PCA Components')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('MAP')
    ax1.grid(True)
    
    # Plot 2: P@K vs Components
    p10_scores = [r['p10'] for r in results_data]
    p20_scores = [r['p20'] for r in results_data]
    ax2.plot(components, p10_scores, 'ro-', label='P@10', linewidth=2, markersize=8)
    ax2.plot(components, p20_scores, 'go-', label='P@20', linewidth=2, markersize=8)
    ax2.set_title('Precision@K vs PCA Components')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Precision')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png')
    print("\nSaved comparison plot to 'pca_comparison.png'")
    plt.close()
    
    # Print summary table
    print("\nSummary of Results:")
    print("Components |   MAP   |  P@10   |  P@20   | Avg Correct")
    print("-" * 55)
    for r in results_data:
        print(f"{r['components']:>9} | {r['map']:.3f} | {r['p10']:.3f} | {r['p20']:.3f} | {r['correct']:.1f}")
    