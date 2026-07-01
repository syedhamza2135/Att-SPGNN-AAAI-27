"""
Output Results Analyzer

This script analyzes the outputs from all three modules (Individual Generation, 
Household Generation, and Assignment) across all MSOA area codes.

It generates two types of reports:
1. Hyperparameter Analysis: Average loss/accuracy for each hyperparameter combination
2. Per-Area Performance: Best model metrics (RMSE, R², Time) for each area code

Usage:
    python analyzeOutputResults.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re


def parse_time_to_seconds(time_str):
    """
    Parse time string to seconds.
    Handles formats like '0:01:04', '1:23:45', '5m 33s', '47s'
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_str = str(time_str).strip()
    
    # Format: '0:01:04' or '1:23:45' (H:MM:SS)
    if ':' in time_str and 'm' not in time_str and 's' not in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
    
    # Format: '5m 33s' or '47s'
    total_seconds = 0
    
    # Extract minutes
    m_match = re.search(r'(\d+)m', time_str)
    if m_match:
        total_seconds += int(m_match.group(1)) * 60
    
    # Extract seconds
    s_match = re.search(r'(\d+)s', time_str)
    if s_match:
        total_seconds += int(s_match.group(1))
    
    return total_seconds if total_seconds > 0 else None


def format_time(seconds):
    """Format seconds to readable time string."""
    if seconds is None or pd.isna(seconds):
        return 'N/A'
    
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f'{m}m {s:02d}s'
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f'{h}h {m:02d}m {s:02d}s'


def find_area_codes(outputs_dir):
    """Find all unique area codes from output directories."""
    area_codes = set()
    
    for item in os.listdir(outputs_dir):
        # Match patterns like 'individuals_E02005940', 'households_E02005940', 'assignment_E02005940'
        match = re.match(r'(individuals|households|assignment)_(E\d+)', item)
        if match:
            area_codes.add(match.group(2))
    
    return sorted(list(area_codes))


def load_individuals_results(outputs_dir, area_code):
    """Load individual generation results for an area code."""
    csv_path = os.path.join(outputs_dir, f'individuals_{area_code}', 'generateIndividuals_results.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['area_code'] = area_code
            df['module'] = 'individuals'
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    return None


def load_households_results(outputs_dir, area_code):
    """Load household generation results for an area code."""
    csv_path = os.path.join(outputs_dir, f'households_{area_code}', 'generateHouseholds_results.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['area_code'] = area_code
            df['module'] = 'households'
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    return None


def load_assignment_results(outputs_dir, area_code):
    """Load assignment results for an area code."""
    csv_path = os.path.join(outputs_dir, f'assignment_{area_code}', 'hp_tuning_results.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['area_code'] = area_code
            df['module'] = 'assignment'
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    return None


def get_population_and_households(outputs_dir, area_code):
    """Get population and household counts for an area code."""
    # Try to get from generated CSV files
    ind_csv = os.path.join(outputs_dir, f'individuals_{area_code}', 'generated_individuals.csv')
    hh_csv = os.path.join(outputs_dir, f'households_{area_code}', 'generated_households.csv')
    
    pop_count = None
    hh_count = None
    
    if os.path.exists(ind_csv):
        try:
            ind_df = pd.read_csv(ind_csv)
            pop_count = len(ind_df)
        except:
            pass
    
    if os.path.exists(hh_csv):
        try:
            hh_df = pd.read_csv(hh_csv)
            hh_count = len(hh_df)
        except:
            pass
    
    return pop_count, hh_count


def analyze_hyperparameters(outputs_dir, area_codes):
    """
    Analyze hyperparameter performance across all areas.
    Returns aggregated statistics for each module and hyperparameter combination.
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER ANALYSIS")
    print("="*80)
    
    results = {
        'individuals': [],
        'households': [],
        'assignment': []
    }
    
    # Load all results
    for area_code in area_codes:
        ind_df = load_individuals_results(outputs_dir, area_code)
        if ind_df is not None:
            results['individuals'].append(ind_df)
        
        hh_df = load_households_results(outputs_dir, area_code)
        if hh_df is not None:
            results['households'].append(hh_df)
        
        assign_df = load_assignment_results(outputs_dir, area_code)
        if assign_df is not None:
            results['assignment'].append(assign_df)
    
    hp_analysis = {}
    
    # ========== INDIVIDUALS ANALYSIS ==========
    if results['individuals']:
        ind_combined = pd.concat(results['individuals'], ignore_index=True)
        print(f"\nIndividuals Module: {len(results['individuals'])} areas loaded")
        print(f"Columns available: {list(ind_combined.columns)}")
        
        # Manually calculate averages for each HP combination using final_loss and final_accuracy
        hp_combinations = ind_combined.groupby(['learning_rate', 'hidden_channels'])
        
        ind_hp_list = []
        for (lr, hc), group in hp_combinations:
            # Filter out areas where model didn't converge (0 loss or 0 accuracy)
            valid_mask = (group['final_loss'] > 0) & (group['final_accuracy'] > 0)
            invalid_areas = group[~valid_mask]['area_code'].tolist()
            
            if len(invalid_areas) > 0:
                print(f"  [WARN] Individuals LR={lr}, Hidden={hc}: Excluding {len(invalid_areas)} areas with 0 loss/accuracy: {invalid_areas}")
            
            valid_group = group[valid_mask]
            
            if len(valid_group) == 0:
                print(f"  [WARN] Individuals LR={lr}, Hidden={hc}: No valid areas remaining, skipping this combination")
                continue
            
            # Extract final_loss and final_accuracy for this HP combination across valid areas
            losses = valid_group['final_loss'].values
            accuracies = valid_group['final_accuracy'].values
            rmses = valid_group['rmse'].values
            
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)
            avg_rmse = np.mean(rmses)
            
            ind_hp_list.append({
                'learning_rate': lr,
                'hidden_channels': hc,
                'avg_loss': avg_loss,
                'std_loss': np.std(losses),
                'avg_accuracy': avg_acc,
                'std_accuracy': np.std(accuracies),
                'avg_rmse': avg_rmse,
                'std_rmse': np.std(rmses),
                'count': len(losses),
                'excluded': len(invalid_areas)
            })
        
        ind_hp = pd.DataFrame(ind_hp_list)
        
        # Find best configuration (lowest loss)
        best_idx = ind_hp['avg_loss'].idxmin()
        ind_hp['is_best'] = False
        ind_hp.loc[best_idx, 'is_best'] = True
        
        hp_analysis['individuals'] = ind_hp
        
        print("\nIndividuals Hyperparameter Results:")
        print("(Averaging final_loss and final_accuracy across all areas for each HP combination)")
        print("NOTE: Areas with 0 loss or 0 accuracy are excluded from calculations")
        print("-" * 110)
        print(f"{'LR':<10} {'Hidden':<10} {'Avg Loss':<12} {'Avg Acc':<12} {'Avg RMSE':<12} {'Valid Areas':<14} {'Excluded':<10}")
        print("-" * 110)
        total_areas = len(results['individuals'])
        for _, row in ind_hp.sort_values('avg_loss').iterrows():
            best_marker = " *BEST*" if row['is_best'] else ""
            excluded = int(row.get('excluded', 0))
            areas_note = f"{int(row['count'])}/{total_areas}"
            excluded_note = str(excluded) if excluded > 0 else "-"
            print(f"{row['learning_rate']:<10} {int(row['hidden_channels']):<10} "
                  f"{row['avg_loss']:<12.4f} {row['avg_accuracy']:<12.4f} "
                  f"{row['avg_rmse']:<12.4f} {areas_note:<14} {excluded_note:<10}{best_marker}")
    
    # ========== HOUSEHOLDS ANALYSIS ==========
    if results['households']:
        hh_combined = pd.concat(results['households'], ignore_index=True)
        print(f"\nHouseholds Module: {len(results['households'])} areas loaded")
        print(f"Columns available: {list(hh_combined.columns)}")
        
        # Manually calculate averages for each HP combination using final_loss and average_accuracy
        hp_combinations = hh_combined.groupby(['learning_rate', 'hidden_channels'])
        
        hh_hp_list = []
        for (lr, hc), group in hp_combinations:
            # Filter out areas where model didn't converge (0 loss or 0 accuracy)
            valid_mask = (group['final_loss'] > 0) & (group['average_accuracy'] > 0)
            invalid_areas = group[~valid_mask]['area_code'].tolist()
            
            if len(invalid_areas) > 0:
                print(f"  [WARN] Households LR={lr}, Hidden={hc}: Excluding {len(invalid_areas)} areas with 0 loss/accuracy: {invalid_areas}")
            
            valid_group = group[valid_mask]
            
            if len(valid_group) == 0:
                print(f"  [WARN] Households LR={lr}, Hidden={hc}: No valid areas remaining, skipping this combination")
                continue
            
            # Extract final_loss and average_accuracy for this HP combination across valid areas
            losses = valid_group['final_loss'].values
            accuracies = valid_group['average_accuracy'].values  # Note: households only has average_accuracy
            rmses = valid_group['rmse'].values
            
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)
            avg_rmse = np.mean(rmses)
            
            hh_hp_list.append({
                'learning_rate': lr,
                'hidden_channels': hc,
                'avg_loss': avg_loss,
                'std_loss': np.std(losses),
                'avg_accuracy': avg_acc,
                'std_accuracy': np.std(accuracies),
                'avg_rmse': avg_rmse,
                'std_rmse': np.std(rmses),
                'count': len(losses),
                'excluded': len(invalid_areas)
            })
        
        hh_hp = pd.DataFrame(hh_hp_list)
        
        best_idx = hh_hp['avg_loss'].idxmin()
        hh_hp['is_best'] = False
        hh_hp.loc[best_idx, 'is_best'] = True
        
        hp_analysis['households'] = hh_hp
        
        print("\nHouseholds Hyperparameter Results:")
        print("(Averaging final_loss and average_accuracy across all areas for each HP combination)")
        print("NOTE: Areas with 0 loss or 0 accuracy are excluded from calculations")
        print("-" * 110)
        print(f"{'LR':<10} {'Hidden':<10} {'Avg Loss':<12} {'Avg Acc':<12} {'Avg RMSE':<12} {'Valid Areas':<14} {'Excluded':<10}")
        print("-" * 110)
        total_areas = len(results['households'])
        for _, row in hh_hp.sort_values('avg_loss').iterrows():
            best_marker = " *BEST*" if row['is_best'] else ""
            excluded = int(row.get('excluded', 0))
            areas_note = f"{int(row['count'])}/{total_areas}"
            excluded_note = str(excluded) if excluded > 0 else "-"
            print(f"{row['learning_rate']:<10} {int(row['hidden_channels']):<10} "
                  f"{row['avg_loss']:<12.4f} {row['avg_accuracy']:<12.4f} "
                  f"{row['avg_rmse']:<12.4f} {areas_note:<14} {excluded_note:<10}{best_marker}")
    
    # ========== ASSIGNMENT ANALYSIS ==========
    if results['assignment']:
        assign_combined = pd.concat(results['assignment'], ignore_index=True)
        print(f"\nAssignment Module: {len(results['assignment'])} areas loaded")
        print(f"Columns available: {list(assign_combined.columns)}")
        
        # Manually calculate averages for each HP combination using final_loss and overall_accuracy
        hp_combinations = assign_combined.groupby(['learning_rate', 'hidden_channels'])
        
        assign_hp_list = []
        for (lr, hc), group in hp_combinations:
            # Filter out areas where model didn't converge (0 loss or 0 accuracy)
            valid_mask = (group['final_loss'] > 0) & (group['overall_accuracy'] > 0)
            invalid_areas = group[~valid_mask]['area_code'].tolist()
            
            if len(invalid_areas) > 0:
                print(f"  [WARN] Assignment LR={lr}, Hidden={hc}: Excluding {len(invalid_areas)} areas with 0 loss/accuracy: {invalid_areas}")
            
            valid_group = group[valid_mask]
            
            if len(valid_group) == 0:
                print(f"  [WARN] Assignment LR={lr}, Hidden={hc}: No valid areas remaining, skipping this combination")
                continue
            
            # Extract final_loss and overall_accuracy for this HP combination across valid areas
            losses = valid_group['final_loss'].values
            accuracies = valid_group['overall_accuracy'].values
            
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)
            
            assign_hp_list.append({
                'learning_rate': lr,
                'hidden_channels': hc,
                'avg_loss': avg_loss,
                'std_loss': np.std(losses),
                'avg_accuracy': avg_acc,
                'std_accuracy': np.std(accuracies),
                'count': len(losses),
                'excluded': len(invalid_areas)
            })
        
        assign_hp = pd.DataFrame(assign_hp_list)
        
        best_idx = assign_hp['avg_loss'].idxmin()
        assign_hp['is_best'] = False
        assign_hp.loc[best_idx, 'is_best'] = True
        
        hp_analysis['assignment'] = assign_hp
        
        print("\nAssignment Hyperparameter Results:")
        print("(Averaging final_loss and overall_accuracy across all areas for each HP combination)")
        print("NOTE: Areas with 0 loss or 0 accuracy are excluded from calculations")
        print("-" * 100)
        print(f"{'LR':<10} {'Hidden':<10} {'Avg Loss':<12} {'Avg Acc':<12} {'Valid Areas':<14} {'Excluded':<10}")
        print("-" * 100)
        total_areas = len(results['assignment'])
        for _, row in assign_hp.sort_values('avg_loss').iterrows():
            best_marker = " *BEST*" if row['is_best'] else ""
            excluded = int(row.get('excluded', 0))
            areas_note = f"{int(row['count'])}/{total_areas}"
            excluded_note = str(excluded) if excluded > 0 else "-"
            print(f"{row['learning_rate']:<10} {int(row['hidden_channels']):<10} "
                  f"{row['avg_loss']:<12.4f} {row['avg_accuracy']:<12.4f} "
                  f"{areas_note:<14} {excluded_note:<10}{best_marker}")
    
    return hp_analysis


def analyze_per_area_performance(outputs_dir, area_codes):
    """
    Analyze best model performance for each area across all modules.
    Returns a DataFrame suitable for the results table.
    """
    print("\n" + "="*80)
    print("PER-AREA PERFORMANCE ANALYSIS")
    print("="*80)
    
    results_list = []
    
    for area_code in area_codes:
        area_result = {
            'area_code': area_code,
            'population': None,
            'households': None,
            # Individual Model (best model only)
            'ind_loss': None,
            'ind_rmse': None,
            'ind_acc': None,  # final_accuracy from best model
            'ind_time': None,
            # Household Model (best model only)
            'hh_loss': None,
            'hh_rmse': None,
            'hh_acc': None,  # average_accuracy from best model (final_accuracy not available)
            'hh_time': None,
            # Assignment (best model only)
            'assign_loss': None,
            'assign_acc': None,
            'assign_time': None
        }
        
        # Get population and household counts
        pop, hh = get_population_and_households(outputs_dir, area_code)
        area_result['population'] = pop
        area_result['households'] = hh
        
        # ===== INDIVIDUALS =====
        ind_df = load_individuals_results(outputs_dir, area_code)
        if ind_df is not None and len(ind_df) > 0:
            # Find best model row (where best_model=True)
            if 'best_model' in ind_df.columns:
                best_row = ind_df[ind_df['best_model'] == True]
                if len(best_row) == 0:
                    # If no best_model flag, use lowest loss
                    best_row = ind_df.loc[[ind_df['final_loss'].idxmin()]]
            else:
                best_row = ind_df.loc[[ind_df['final_loss'].idxmin()]]
            
            if len(best_row) > 0:
                best_row = best_row.iloc[0]
                area_result['ind_loss'] = best_row.get('final_loss')
                area_result['ind_rmse'] = best_row.get('rmse')
                # Use final_accuracy for the best model (not average_accuracy)
                area_result['ind_acc'] = best_row.get('final_accuracy', best_row.get('average_accuracy'))
                area_result['ind_time'] = parse_time_to_seconds(best_row.get('training_time'))
        
        # ===== HOUSEHOLDS =====
        hh_df = load_households_results(outputs_dir, area_code)
        if hh_df is not None and len(hh_df) > 0:
            if 'best_model' in hh_df.columns:
                best_row = hh_df[hh_df['best_model'] == True]
                if len(best_row) == 0:
                    best_row = hh_df.loc[[hh_df['final_loss'].idxmin()]]
            else:
                best_row = hh_df.loc[[hh_df['final_loss'].idxmin()]]
            
            if len(best_row) > 0:
                best_row = best_row.iloc[0]
                area_result['hh_loss'] = best_row.get('final_loss')
                area_result['hh_rmse'] = best_row.get('rmse')
                # Use average_accuracy for households (final_accuracy not available in this CSV)
                area_result['hh_acc'] = best_row.get('average_accuracy')
                area_result['hh_time'] = parse_time_to_seconds(best_row.get('training_time'))
        
        # ===== ASSIGNMENT =====
        assign_df = load_assignment_results(outputs_dir, area_code)
        if assign_df is not None and len(assign_df) > 0:
            if 'is_best' in assign_df.columns:
                best_row = assign_df[assign_df['is_best'] == True]
                if len(best_row) == 0:
                    best_row = assign_df.loc[[assign_df['final_loss'].idxmin()]]
            else:
                best_row = assign_df.loc[[assign_df['final_loss'].idxmin()]]
            
            if len(best_row) > 0:
                best_row = best_row.iloc[0]
                area_result['assign_loss'] = best_row.get('final_loss')
                area_result['assign_acc'] = best_row.get('overall_accuracy')
                area_result['assign_time'] = parse_time_to_seconds(best_row.get('training_time'))
        
        results_list.append(area_result)
    
    results_df = pd.DataFrame(results_list)
    
    # Print results table
    print("\nPer-Area Performance Results (Best Model Only):")
    print("-" * 160)
    print(f"{'Area Code':<12} {'Pop.':<8} {'HHs':<8} | "
          f"{'Ind Loss':<10} {'Ind RMSE':<10} {'Ind Acc':<10} {'Ind Time':<10} | "
          f"{'HH Loss':<10} {'HH RMSE':<10} {'HH Acc':<10} {'HH Time':<10} | "
          f"{'Assign Loss':<12} {'Assign Acc':<12} {'Assign Time':<12}")
    print("-" * 160)
    
    for _, row in results_df.iterrows():
        pop_str = str(int(row['population'])) if row['population'] else 'N/A'
        hh_str = str(int(row['households'])) if row['households'] else 'N/A'
        
        ind_loss = f"{row['ind_loss']:.2f}" if row['ind_loss'] else 'N/A'
        ind_rmse = f"{row['ind_rmse']:.2f}" if row['ind_rmse'] else 'N/A'
        ind_acc = f"{row['ind_acc']:.2f}" if row['ind_acc'] else 'N/A'
        ind_time = format_time(row['ind_time'])
        
        hh_loss = f"{row['hh_loss']:.2f}" if row['hh_loss'] else 'N/A'
        hh_rmse = f"{row['hh_rmse']:.2f}" if row['hh_rmse'] else 'N/A'
        hh_acc = f"{row['hh_acc']:.2f}" if row['hh_acc'] else 'N/A'
        hh_time = format_time(row['hh_time'])
        
        assign_loss = f"{row['assign_loss']:.2f}" if row['assign_loss'] else 'N/A'
        assign_acc = f"{row['assign_acc']:.2f}" if row['assign_acc'] else 'N/A'
        assign_time = format_time(row['assign_time'])
        
        print(f"{row['area_code']:<12} {pop_str:<8} {hh_str:<8} | "
              f"{ind_loss:<10} {ind_rmse:<10} {ind_acc:<10} {ind_time:<10} | "
              f"{hh_loss:<10} {hh_rmse:<10} {hh_acc:<10} {hh_time:<10} | "
              f"{assign_loss:<12} {assign_acc:<12} {assign_time:<12}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal Areas Analyzed: {len(results_df)}")
    
    if results_df['ind_rmse'].notna().any():
        print(f"\nIndividual Model (Best Models Only):")
        print(f"  Avg Loss: {results_df['ind_loss'].mean():.2f} (+/- {results_df['ind_loss'].std():.2f})")
        print(f"  Avg RMSE: {results_df['ind_rmse'].mean():.2f} (+/- {results_df['ind_rmse'].std():.2f})")
        print(f"  Avg Accuracy: {results_df['ind_acc'].mean():.2f} (+/- {results_df['ind_acc'].std():.2f})")
        print(f"  Avg Time: {format_time(results_df['ind_time'].mean())}")
    
    if results_df['hh_rmse'].notna().any():
        print(f"\nHousehold Model (Best Models Only):")
        print(f"  Avg Loss: {results_df['hh_loss'].mean():.2f} (+/- {results_df['hh_loss'].std():.2f})")
        print(f"  Avg RMSE: {results_df['hh_rmse'].mean():.2f} (+/- {results_df['hh_rmse'].std():.2f})")
        print(f"  Avg Accuracy: {results_df['hh_acc'].mean():.2f} (+/- {results_df['hh_acc'].std():.2f})")
        print(f"  Avg Time: {format_time(results_df['hh_time'].mean())}")
    
    if results_df['assign_loss'].notna().any():
        print(f"\nAssignment Model:")
        print(f"  Avg Loss: {results_df['assign_loss'].mean():.2f} (+/- {results_df['assign_loss'].std():.2f})")
        print(f"  Avg Accuracy: {results_df['assign_acc'].mean():.2f} (+/- {results_df['assign_acc'].std():.2f})")
        print(f"  Avg Time: {format_time(results_df['assign_time'].mean())}")
    
    return results_df


def generate_latex_tables(hp_analysis, per_area_df, output_dir):
    """Generate LaTeX formatted tables for the paper."""
    
    # ========== HYPERPARAMETER TABLE ==========
    latex_hp = []
    latex_hp.append("% --- UNIFIED HYPERPARAMETER TABLE ---")
    latex_hp.append("\\begin{table}[h]")
    latex_hp.append("\\caption{Hyperparameter Search Results across all Architectures. Optimal configurations are highlighted in \\textbf{bold}.}")
    latex_hp.append("\\label{tab:combined_hyperparams}")
    latex_hp.append("\\centering")
    latex_hp.append("\\resizebox{1\\textwidth}{!}{%")
    latex_hp.append("\\begin{tabular}{|l|c|c|c|c|}")
    latex_hp.append("\\hline")
    latex_hp.append("\\textbf{Model Architecture} & \\textbf{Learning Rate} & \\textbf{Hidden Dim.} & \\textbf{Avg Final Loss} & \\textbf{Avg Final Accuracy} \\\\ \\hline")
    
    # Individual Generation
    if 'individuals' in hp_analysis:
        ind_hp = hp_analysis['individuals'].sort_values('avg_loss')
        first = True
        for _, row in ind_hp.iterrows():
            model_col = "\\multirow{" + str(len(ind_hp)) + "}{*}{1. Individual Gen.}" if first else ""
            
            if row['is_best']:
                lr = f"\\textbf{{{row['learning_rate']}}}"
                hd = f"\\textbf{{{int(row['hidden_channels'])}}}"
                loss = f"\\textbf{{{row['avg_loss']:.3f}}}"
                acc = f"\\textbf{{{row['avg_accuracy']:.3f}}}"
            else:
                lr = str(row['learning_rate'])
                hd = str(int(row['hidden_channels']))
                loss = f"{row['avg_loss']:.3f}"
                acc = f"{row['avg_accuracy']:.3f}"
            
            latex_hp.append(f"{model_col} & {lr} & {hd} & {loss} & {acc} \\\\ " + ("\\cline{2-5}" if not row['is_best'] or first else "\\hline"))
            first = False
    
    latex_hp.append("\\multicolumn{5}{|c|}{} \\\\ \\hline")
    
    # Household Generation
    if 'households' in hp_analysis:
        hh_hp = hp_analysis['households'].sort_values('avg_loss')
        first = True
        for _, row in hh_hp.iterrows():
            model_col = "\\multirow{" + str(len(hh_hp)) + "}{*}{2. Household Gen.}" if first else ""
            
            if row['is_best']:
                lr = f"\\textbf{{{row['learning_rate']}}}"
                hd = f"\\textbf{{{int(row['hidden_channels'])}}}"
                loss = f"\\textbf{{{row['avg_loss']:.3f}}}"
                acc = f"\\textbf{{{row['avg_accuracy']:.3f}}}"
            else:
                lr = str(row['learning_rate'])
                hd = str(int(row['hidden_channels']))
                loss = f"{row['avg_loss']:.3f}"
                acc = f"{row['avg_accuracy']:.3f}"
            
            latex_hp.append(f"{model_col} & {lr} & {hd} & {loss} & {acc} \\\\ " + ("\\cline{2-5}" if not row['is_best'] or first else "\\hline"))
            first = False
    
    latex_hp.append("\\multicolumn{5}{|c|}{} \\\\ \\hline")
    
    # Assignment
    if 'assignment' in hp_analysis:
        assign_hp = hp_analysis['assignment'].sort_values('avg_loss')
        first = True
        for _, row in assign_hp.iterrows():
            model_col = "\\multirow{" + str(len(assign_hp)) + "}{*}{3. Assignment}" if first else ""
            
            if row['is_best']:
                lr = f"\\textbf{{{row['learning_rate']}}}"
                hd = f"\\textbf{{{int(row['hidden_channels'])}}}"
                loss = f"\\textbf{{{row['avg_loss']:.3f}}}"
                acc = f"\\textbf{{{row['avg_accuracy']:.3f}}}"
            else:
                lr = str(row['learning_rate'])
                hd = str(int(row['hidden_channels']))
                loss = f"{row['avg_loss']:.3f}"
                acc = f"{row['avg_accuracy']:.3f}"
            
            latex_hp.append(f"{model_col} & {lr} & {hd} & {loss} & {acc} \\\\ " + ("\\cline{2-5}" if not row['is_best'] or first else "\\hline"))
            first = False
    
    latex_hp.append("\\end{tabular}%")
    latex_hp.append("}")
    latex_hp.append("\\end{table}")
    
    # ========== RESULTS TABLE ==========
    latex_results = []
    latex_results.append("% --- UNIFIED MASTER TABLE ---")
    latex_results.append("\\begin{table}[htbp]")
    latex_results.append("\\caption{Comprehensive Performance Analysis across MSOAs. Time represents inference/generation duration.}")
    latex_results.append("\\label{tab:combined_results}")
    latex_results.append("\\centering")
    latex_results.append("\\resizebox{\\textwidth}{!}{%")
    latex_results.append("\\begin{tabular}{l|cc|ccc|ccc|ccc}")
    latex_results.append("\\toprule")
    latex_results.append("\\textbf{} & \\multicolumn{2}{c|}{\\textbf{Census Data}} & \\multicolumn{3}{c|}{\\textbf{Individual Model}} & \\multicolumn{3}{c|}{\\textbf{Household Model}} & \\multicolumn{3}{c}{\\textbf{Assignment}} \\\\")
    latex_results.append("\\textbf{Area Code} & \\textbf{Pop.} & \\textbf{HHs} & \\textbf{RMSE} & \\textbf{Acc ($R^2$)} & \\textbf{Time} & \\textbf{RMSE} & \\textbf{Acc ($R^2$)} & \\textbf{Time} & \\textbf{Loss} & \\textbf{Acc} & \\textbf{Time} \\\\")
    latex_results.append("\\midrule")
    
    for _, row in per_area_df.iterrows():
        area = row['area_code']
        pop = str(int(row['population'])) if row['population'] else '--'
        hh = str(int(row['households'])) if row['households'] else '--'
        
        ind_rmse = f"{row['ind_rmse']:.2f}" if row['ind_rmse'] else '--'
        ind_acc = f"{row['ind_acc']:.2f}" if row['ind_acc'] else '--'
        ind_time = format_time(row['ind_time']) if row['ind_time'] else '--'
        
        hh_rmse = f"{row['hh_rmse']:.2f}" if row['hh_rmse'] else '--'
        hh_acc = f"{row['hh_acc']:.2f}" if row['hh_acc'] else '--'
        hh_time = format_time(row['hh_time']) if row['hh_time'] else '--'
        
        assign_loss = f"{row['assign_loss']:.2f}" if row['assign_loss'] else '--'
        assign_acc = f"{row['assign_acc']:.2f}" if row['assign_acc'] else '--'
        assign_time = format_time(row['assign_time']) if row['assign_time'] else '--'
        
        latex_results.append(f"{area} & {pop} & {hh} & {ind_rmse} & {ind_acc} & {ind_time} & {hh_rmse} & {hh_acc} & {hh_time} & {assign_loss} & {assign_acc} & {assign_time} \\\\")
    
    latex_results.append("\\bottomrule")
    latex_results.append("\\end{tabular}%")
    latex_results.append("}")
    latex_results.append("\\end{table}")
    
    # Save LaTeX files
    hp_latex_path = os.path.join(output_dir, 'hyperparameter_table.tex')
    with open(hp_latex_path, 'w') as f:
        f.write('\n'.join(latex_hp))
    print(f"\nHyperparameter LaTeX table saved to: {hp_latex_path}")
    
    results_latex_path = os.path.join(output_dir, 'results_table.tex')
    with open(results_latex_path, 'w') as f:
        f.write('\n'.join(latex_results))
    print(f"Results LaTeX table saved to: {results_latex_path}")
    
    return '\n'.join(latex_hp), '\n'.join(latex_results)


def main():
    """Main function to run the analysis."""
    print("="*80)
    print("OUTPUT RESULTS ANALYZER")
    print("="*80)
    
    # Get outputs directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(current_dir, '..', 'outputs')
    outputs_dir = os.path.abspath(outputs_dir)
    
    print(f"\nOutputs directory: {outputs_dir}")
    
    if not os.path.exists(outputs_dir):
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        return
    
    # Find all area codes
    area_codes = find_area_codes(outputs_dir)
    print(f"Found {len(area_codes)} area codes: {', '.join(area_codes)}")
    
    if not area_codes:
        print("ERROR: No area codes found in outputs directory")
        return
    
    # Run hyperparameter analysis
    hp_analysis = analyze_hyperparameters(outputs_dir, area_codes)
    
    # Run per-area performance analysis
    per_area_df = analyze_per_area_performance(outputs_dir, area_codes)
    
    # Generate LaTeX tables
    latex_hp, latex_results = generate_latex_tables(hp_analysis, per_area_df, outputs_dir)
    
    # Save combined CSV results
    combined_csv_path = os.path.join(outputs_dir, 'combined_results_analysis.csv')
    per_area_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined results CSV saved to: {combined_csv_path}")
    
    # Save hyperparameter analysis to CSV
    for module, hp_df in hp_analysis.items():
        hp_csv_path = os.path.join(outputs_dir, f'{module}_hyperparameter_analysis.csv')
        hp_df.to_csv(hp_csv_path, index=False)
        print(f"{module.capitalize()} hyperparameter analysis saved to: {hp_csv_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Print LaTeX tables for easy copy-paste
    print("\n\n" + "="*80)
    print("LATEX HYPERPARAMETER TABLE (copy below):")
    print("="*80)
    print(latex_hp)
    
    print("\n\n" + "="*80)
    print("LATEX RESULTS TABLE (copy below):")
    print("="*80)
    print(latex_results)


if __name__ == '__main__':
    main()

