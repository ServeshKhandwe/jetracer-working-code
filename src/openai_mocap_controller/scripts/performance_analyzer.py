#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os

class PerformanceAnalyzer:
    def __init__(self, log_file=None):
        self.log_file = log_file or "/tmp/openai_mocap_performance.json"
        self.data = []
        
    def load_data(self):
        """Load performance data from log file"""
        if not os.path.exists(self.log_file):
            print(f"Log file not found: {self.log_file}")
            return False
            
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.data.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            print(f"Loaded {len(self.data)} data points")
            return len(self.data) > 0
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def analyze_by_model(self):
        """Analyze performance by model"""
        if not self.data:
            print("No data to analyze")
            return
            
        # Group data by model
        model_data = {}
        for entry in self.data:
            model = entry.get('model', 'unknown')
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(entry)
            
        # Calculate metrics for each model
        results = {}
        for model, entries in model_data.items():
            distances = [e['distance_to_goal'] for e in entries]
            distances_traveled = [e.get('distance_traveled', 0) for e in entries]
            
            results[model] = {
                'count': len(entries),
                'avg_distance_to_goal': np.mean(distances),
                'min_distance_to_goal': np.min(distances),
                'final_distance_to_goal': distances[-1] if distances else float('inf'),
                'total_distance_traveled': distances_traveled[-1] if distances_traveled else 0,
                'std_distance_to_goal': np.std(distances),
                'success_rate': sum(1 for d in distances if d < 0.1) / len(distances) * 100
            }
            
        return results
        
    def print_analysis(self):
        """Print analysis results"""
        results = self.analyze_by_model()
        if not results:
            return
            
        print("\n" + "="*60)
        print("OPENAI MODEL PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Sort models by average distance to goal (best first)
        sorted_models = sorted(results.items(), key=lambda x: x[1]['avg_distance_to_goal'])
        
        for i, (model, metrics) in enumerate(sorted_models):
            print(f"\n{i+1}. {model.upper()}")
            print("-" * 40)
            print(f"Data points: {metrics['count']}")
            print(f"Average distance to goal: {metrics['avg_distance_to_goal']:.3f} m")
            print(f"Minimum distance to goal: {metrics['min_distance_to_goal']:.3f} m")
            print(f"Final distance to goal: {metrics['final_distance_to_goal']:.3f} m")
            print(f"Total distance traveled: {metrics['total_distance_traveled']:.3f} m")
            print(f"Standard deviation: {metrics['std_distance_to_goal']:.3f} m")
            print(f"Success rate (< 0.1m): {metrics['success_rate']:.1f}%")
            
            # Calculate efficiency
            if metrics['avg_distance_to_goal'] > 0:
                efficiency = metrics['total_distance_traveled'] / metrics['avg_distance_to_goal']
                print(f"Efficiency ratio: {efficiency:.2f}")
                
        print("\n" + "="*60)
        
    def plot_performance(self, save_path=None):
        """Create performance plots"""
        if not self.data:
            print("No data to plot")
            return
            
        # Group data by model
        model_data = {}
        for entry in self.data:
            model = entry.get('model', 'unknown')
            if model not in model_data:
                model_data[model] = {'times': [], 'distances': [], 'positions': []}
            
            model_data[model]['times'].append(entry['timestamp'])
            model_data[model]['distances'].append(entry['distance_to_goal'])
            model_data[model]['positions'].append((entry['robot_x'], entry['robot_y']))
            
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Distance to goal over time
        for model, data in model_data.items():
            times = np.array(data['times'])
            times = (times - times[0]) / 60  # Convert to minutes from start
            ax1.plot(times, data['distances'], label=model, linewidth=2)
            
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Distance to Goal (m)')
        ax1.set_title('Distance to Goal Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Robot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
        for i, (model, data) in enumerate(model_data.items()):
            positions = np.array(data['positions'])
            ax2.plot(positions[:, 0], positions[:, 1], 
                    label=model, color=colors[i], linewidth=2, alpha=0.7)
            ax2.scatter(positions[0, 0], positions[0, 1], 
                       color=colors[i], marker='o', s=100, label=f'{model} start')
            ax2.scatter(positions[-1, 0], positions[-1, 1], 
                       color=colors[i], marker='s', s=100, label=f'{model} end')
                       
        ax2.scatter(0, 0, color='red', marker='*', s=200, label='Goal')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Robot Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Plot 3: Distance distribution
        distances_by_model = [data['distances'] for data in model_data.values()]
        labels = list(model_data.keys())
        ax3.boxplot(distances_by_model, labels=labels)
        ax3.set_ylabel('Distance to Goal (m)')
        ax3.set_title('Distance Distribution by Model')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Performance metrics comparison
        results = self.analyze_by_model()
        models = list(results.keys())
        avg_distances = [results[m]['avg_distance_to_goal'] for m in models]
        min_distances = [results[m]['min_distance_to_goal'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax4.bar(x - width/2, avg_distances, width, label='Average Distance', alpha=0.8)
        ax4.bar(x + width/2, min_distances, width, label='Minimum Distance', alpha=0.8)
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Distance to Goal (m)')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze OpenAI mocap controller performance')
    parser.add_argument('--log-file', '-f', 
                       help='Path to performance log file',
                       default='/tmp/openai_mocap_performance.json')
    parser.add_argument('--plot', '-p', 
                       help='Save plot to file (optional)',
                       default=None)
    parser.add_argument('--no-display', 
                       action='store_true',
                       help='Don\'t display plots (useful for headless systems)')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.log_file)
    
    if not analyzer.load_data():
        print("Failed to load performance data")
        return
        
    # Print analysis
    analyzer.print_analysis()
    
    # Create plots
    if not args.no_display or args.plot:
        analyzer.plot_performance(args.plot)

if __name__ == '__main__':
    main()