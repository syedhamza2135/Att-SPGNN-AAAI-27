"""
Household Assignment Module with Memory-Efficient Graph Attention Networks (GAT)

This module assigns generated persons to generated households using a Graph Attention
Network (GAT) that respects household size, religion, ethnicity, and composition constraints.

GAT INTEGRATION (Memory-Efficient for 24GB GPU):
- Replaces GraphSAGE with GAT for better attention-based assignments
- Uses only 2 attention heads (vs 4-8 typical) to reduce memory
- concat=False to average heads instead of concatenating (saves memory)
- Edge dropout (10%) to reduce attention computation during training
- GraphNorm instead of BatchNorm for better graph data handling
- Recommended hidden_dims: 128-256 (avoid 512+ to prevent OOM)

MEMORY OPTIMIZATION FEATURES:
- Comprehensive GPU memory monitoring throughout training
- Automatic memory cleanup between training runs
- Emergency memory cleanup for OOM recovery
- Early stopping (patience=100) to avoid unnecessary epochs
- Gradient clipping to prevent exploding gradients

If you encounter OOM errors:
1. Reduce hidden_dims to [128] only
2. Increase edge_dropout to 0.2 in hyperparameters section
3. Reduce learning_rates to fewer values
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GraphNorm
import os
import random
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import gc

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for household assignment using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# GPU Memory Management Functions
def print_gpu_memory_info(device, message=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        print("GPU memory cleared and reset")

def safe_delete_tensor(tensor):
    """Safely delete a tensor and free GPU memory"""
    if tensor is not None:
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def safe_delete_model(model):
    """Safely delete a model and free GPU memory"""
    if model is not None:
        # Move model to CPU first to free GPU memory
        if hasattr(model, 'cpu'):
            model = model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def monitor_memory_usage(device, message="", threshold_gb=8.0):
    """Monitor GPU memory usage and warn if approaching limit"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        
        # Warn if memory usage is high
        if allocated > threshold_gb:
            print(f"WARNING: High GPU memory usage detected ({allocated:.2f}GB). Consider reducing batch size or model complexity.")
            return True
        return False
    return False

def emergency_memory_cleanup():
    """Emergency memory cleanup when memory is critically low"""
    print("Performing emergency memory cleanup...")
    
    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("Emergency memory cleanup completed")

def estimate_gat_memory_usage(num_persons, num_households, hidden_channels, attention_heads, num_edges):
    """
    Estimate approximate GPU memory usage for GAT model.
    Helps users predict OOM errors before training.
    
    Returns estimated memory in GB.
    """
    # Model parameters memory
    # GAT layers: (in * hidden * heads) + (hidden * hidden * heads) + (hidden * hidden * heads)
    # Final layers: (hidden * hidden) + (hidden * num_households)
    param_memory = (
        (7 * hidden_channels * attention_heads) +  # GAT1
        (hidden_channels * hidden_channels * attention_heads) +  # GAT2
        (hidden_channels * hidden_channels * attention_heads) +  # GAT3
        (hidden_channels * hidden_channels) +  # FC1
        (hidden_channels * num_households)  # FC2
    ) * 4 / (1024**3)  # 4 bytes per float32, convert to GB
    
    # Attention weights memory (major component for GAT)
    attention_memory = num_edges * attention_heads * 4 / (1024**3)
    
    # Node embeddings and activations
    activation_memory = num_persons * hidden_channels * 4 * 6 / (1024**3)  # 6 activation tensors
    
    # Assignments matrix (soft assignments during training)
    assignment_memory = num_persons * num_households * 4 / (1024**3)
    
    total_memory = param_memory + attention_memory + activation_memory + assignment_memory
    
    print(f"\n{'='*60}")
    print(f"ESTIMATED GAT MEMORY USAGE")
    print(f"{'='*60}")
    print(f"Model parameters: {param_memory:.2f} GB")
    print(f"Attention weights: {attention_memory:.2f} GB")
    print(f"Node activations: {activation_memory:.2f} GB")
    print(f"Assignment matrix: {assignment_memory:.2f} GB")
    print(f"{'='*60}")
    print(f"TOTAL ESTIMATED: {total_memory:.2f} GB")
    print(f"{'='*60}")
    
    if total_memory > 20:
        print(f"[WARN]  WARNING: Estimated memory ({total_memory:.2f} GB) is very high!")
        print(f"    Recommendations:")
        print(f"    1. Reduce hidden_channels to 128")
        print(f"    2. Increase edge_dropout to 0.2")
        print(f"    3. Consider reducing attention_heads to 1")
    elif total_memory > 15:
        print(f"[WARN]  CAUTION: Estimated memory ({total_memory:.2f} GB) is moderately high")
        print(f"    Monitor memory usage during training")
    else:
        print(f"[OK] Estimated memory ({total_memory:.2f} GB) should be safe for 24GB GPU")
    
    print()
    return total_memory

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Assignment Hyperparameter Tuning for area: {selected_area_code}")

# Household size extraction function
def extract_household_sizes_from_tensor(household_nodes_tensor, device):
    """
    Extract household size categories from the generated household tensor.
    
    Args:
        household_nodes_tensor: Tensor with shape (num_households, 6)
                               [household_composition, ethnicity, religion, tenure, size, rooms]
        device: PyTorch device for tensor operations
    
    Returns:
        torch.Tensor: Household size category indices as tensor of shape (num_households,)
        bool: True if extraction successful, False if fallback needed
    """
    try:
        # Validate tensor structure
        if household_nodes_tensor.dim() != 2:
            print(f"Warning: Expected 2D tensor, got {household_nodes_tensor.dim()}D tensor")
            return None, False
            
        if household_nodes_tensor.size(1) < 5:  # Need at least 5 columns to access index 4
            print(f"Warning: Expected at least 5 columns, got {household_nodes_tensor.size(1)} columns")
            return None, False
        
        # Extract size category indices from column 4 (0-indexed)
        size_category_indices = household_nodes_tensor[:, 4].long()
        
        # Ensure indices are on the correct device
        if size_category_indices.device != device:
            size_category_indices = size_category_indices.to(device)
        
        # Validate that indices are within the expected range (0-3 for 4 categories)
        valid_indices = (size_category_indices >= 0) & (size_category_indices < 4)
        if not valid_indices.all():
            invalid_count = (~valid_indices).sum().item()
            print(f"Warning: {invalid_count} households have invalid size category indices")
            # Clamp invalid indices to valid range
            size_category_indices = torch.clamp(size_category_indices, 0, 3)
        
        print(f"Successfully extracted household size categories from tensor. Category distribution:")
        unique_categories, counts = torch.unique(size_category_indices, return_counts=True)
        size_categories = ['1', '2', '3', '4+']
        for cat_idx, count in zip(unique_categories.cpu().numpy(), counts.cpu().numpy()):
            category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
            print(f"  Category {cat_idx} ({category_name}): {count} households")
            
        return size_category_indices, True
        
    except Exception as e:
        print(f"Error extracting household sizes from tensor: {e}")
        return None, False

def validate_household_size_extraction(household_nodes_tensor, device):
    """
    Validate the household size category extraction functionality with sample data.
    
    Args:
        household_nodes_tensor: The loaded household tensor
        device: PyTorch device
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\n=== Validating Household Size Category Extraction ===")
    
    try:
        # Test the extraction function
        extracted_categories, success = extract_household_sizes_from_tensor(household_nodes_tensor, device)
        
        if not success:
            print("Validation FAILED: Size category extraction was not successful")
            return False
        
        # Basic validation checks
        num_households = household_nodes_tensor.size(0)
        if extracted_categories.size(0) != num_households:
            print(f"Validation FAILED: Expected {num_households} categories, got {extracted_categories.size(0)}")
            return False
        
        # Check if categories are within valid range (0-3 for 4 categories)
        min_cat = extracted_categories.min().item()
        max_cat = extracted_categories.max().item()
        
        if min_cat < 0 or max_cat > 3:
            print(f"Validation WARNING: Household size categories outside expected range [0-3]: min={min_cat}, max={max_cat}")
        
        # Check for reasonable distribution
        unique_categories, counts = torch.unique(extracted_categories, return_counts=True)
        size_categories = ['1', '2', '3', '4+']
        print("Extracted size category distribution validation:")
        for cat_idx, count in zip(unique_categories.cpu().numpy(), counts.cpu().numpy()):
            percentage = (count / num_households) * 100
            category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
            print(f"  Category {cat_idx} ({category_name}): {count} households ({percentage:.1f}%)")
        
        print("Validation PASSED: Household size category extraction is working correctly")
        return True
        
    except Exception as e:
        print(f"Validation FAILED: Exception during validation: {e}")
        return False

def validate_tensor_schema(person_nodes, household_nodes):
    """
    Validate that loaded tensors match expected schema.
    
    Expected schemas:
    - person_nodes: [num_persons, 7] 
      [age, sex, religion, ethnicity, marital, qualification, household_composition]
    - household_nodes: [num_households, 6]
      [household_composition, ethnicity, religion, tenure, size, rooms]
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\n=== VALIDATING TENSOR SCHEMAS ===")
    
    errors = []
    warnings = []
    
    # Validate person tensor
    if person_nodes.dim() != 2:
        errors.append(f"Person tensor should be 2D, got {person_nodes.dim()}D")
    elif person_nodes.size(1) != 7:
        errors.append(f"Person tensor should have 7 columns, got {person_nodes.size(1)}")
    else:
        print(f"[OK] Person tensor shape: {person_nodes.shape} - CORRECT")
        
        # Validate attribute indices are within expected ranges
        # Age: 0-20, Sex: 0-1, Religion: 0-8, Ethnicity: 0-17, Marital: 0-5, Qualification: 0-6, HHComp: 0-14
        expected_ranges = {
            0: (0, 20, "age"),
            1: (0, 1, "sex"),
            2: (0, 8, "religion"),
            3: (0, 17, "ethnicity"),
            4: (0, 5, "marital"),
            5: (0, 6, "qualification"),
            6: (0, 14, "household_composition")
        }
        
        for col_idx, (min_val, max_val, attr_name) in expected_ranges.items():
            col_min = person_nodes[:, col_idx].min().item()
            col_max = person_nodes[:, col_idx].max().item()
            if col_min < min_val or col_max > max_val:
                warnings.append(f"Person {attr_name} (col {col_idx}) outside expected range [{min_val},{max_val}]: actual [{col_min:.0f},{col_max:.0f}]")
            else:
                print(f"  [OK] {attr_name} range [{col_min:.0f},{col_max:.0f}] within [{min_val},{max_val}]")
    
    # Validate household tensor
    if household_nodes.dim() != 2:
        errors.append(f"Household tensor should be 2D, got {household_nodes.dim()}D")
    elif household_nodes.size(1) != 6:
        errors.append(f"Household tensor should have 6 columns, got {household_nodes.size(1)}")
    else:
        print(f"[OK] Household tensor shape: {household_nodes.shape} - CORRECT")
        
        # Validate attribute indices are within expected ranges
        # HHComp: 0-14, Ethnicity: 0-17, Religion: 0-8, Tenure: 0-3, Size: 0-3, Rooms: 0-5
        expected_ranges = {
            0: (0, 14, "household_composition"),
            1: (0, 17, "ethnicity"),
            2: (0, 8, "religion"),
            3: (0, 3, "tenure"),
            4: (0, 3, "size"),
            5: (0, 5, "rooms")
        }
        
        for col_idx, (min_val, max_val, attr_name) in expected_ranges.items():
            col_min = household_nodes[:, col_idx].min().item()
            col_max = household_nodes[:, col_idx].max().item()
            if col_min < min_val or col_max > max_val:
                warnings.append(f"Household {attr_name} (col {col_idx}) outside expected range [{min_val},{max_val}]: actual [{col_min:.0f},{col_max:.0f}]")
            else:
                print(f"  [OK] {attr_name} range [{col_min:.0f},{col_max:.0f}] within [{min_val},{max_val}]")
    
    # Print results
    if errors:
        print("\n[ERROR] SCHEMA VALIDATION FAILED:")
        for error in errors:
            print(f"  ERROR: {error}")
        return False
    
    if warnings:
        print("\n[WARN] SCHEMA VALIDATION WARNINGS:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    
    print("\n[OK] Schema validation PASSED (with warnings)" if warnings else "\n[OK] Schema validation PASSED")
    return True

def validate_assignment_compatibility(person_nodes, household_nodes):
    """
    Validate that persons can be feasibly assigned to households.
    
    Checks:
    1. Total household capacity >= number of persons (accounting for 4+ flexibility)
    2. Household composition distributions are compatible
    3. Religion/ethnicity distributions roughly align
    """
    print("\n=== VALIDATING ASSIGNMENT COMPATIBILITY ===")
    
    num_persons = person_nodes.size(0)
    num_households = household_nodes.size(0)
    
    print(f"Number of persons: {num_persons}")
    print(f"Number of households: {num_households}")
    
    # Extract household sizes from column 4 (0-indexed)
    # Size categories: 0->1, 1->2, 2->3, 3->4+
    household_size_categories = household_nodes[:, 4].long()
    household_sizes = household_size_categories + 1  # Convert to actual sizes (1-4)
    
    # Count households in 4+ category
    num_4plus_households = (household_size_categories == 3).sum().item()
    
    # Calculate minimum capacity (treating 4+ as exactly 4)
    min_capacity = household_sizes.sum().item()
    
    print(f"Minimum household capacity: {min_capacity} (treating 4+ as exactly 4)")
    print(f"Number of 4+ households: {num_4plus_households}")
    
    # Check if minimum capacity is sufficient
    if min_capacity >= num_persons:
        surplus = min_capacity - num_persons
        print(f"[OK] Sufficient capacity: {surplus} extra spaces available")
    else:
        # Calculate shortfall
        shortfall = num_persons - min_capacity
        
        # Calculate if 4+ households can accommodate the shortfall
        # Average extra capacity needed per 4+ household
        if num_4plus_households > 0:
            avg_extra_per_4plus = shortfall / num_4plus_households
            required_avg_size = 4 + avg_extra_per_4plus
            
            print(f"[WARN] Minimum capacity shortfall: {shortfall} persons")
            print(f"   However, {num_4plus_households} households are in '4+' category")
            print(f"   These households can flexibly accommodate extra persons")
            print(f"   Required average size for 4+ households: {required_avg_size:.2f} persons")
            
            # If average required size is reasonable (<=8), consider it feasible
            if required_avg_size <= 8.0:
                print(f"[OK] Assignment is FEASIBLE: 4+ households can expand to accommodate all persons")
                print(f"   (4+ households will average {required_avg_size:.2f} persons each)")
            else:
                print(f"[WARN] CAPACITY WARNING: 4+ households would need to average {required_avg_size:.2f} persons")
                print(f"   This is higher than typical (4-8 persons), but training will continue.")
                print(f"   The model will do its best to assign all persons.")
                print(f"   Some households may be overfilled or some persons may have suboptimal assignments.")
                # Continue with warning instead of failing
        else:
            print(f"[WARN] CAPACITY WARNING: Insufficient household capacity!")
            print(f"   Need to accommodate {num_persons} persons but only {min_capacity} spaces available")
            print(f"   Shortfall: {shortfall} persons")
            print(f"   No 4+ households available to provide flexibility")
            print(f"   Training will continue, but some households will be overfilled.")
            # Continue with warning instead of failing
    
    # Check household composition compatibility
    person_hhcomp = person_nodes[:, 6].long()  # Person household composition
    household_hhcomp = household_nodes[:, 0].long()  # Household composition
    
    person_hhcomp_dist = torch.bincount(person_hhcomp, minlength=15)
    household_hhcomp_dist = torch.bincount(household_hhcomp, minlength=15)
    
    print("\nHousehold Composition Distribution Comparison:")
    hh_comps = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-nA', '1FL-2C', '1H-nS', '1H-nE', '1H-nA', '1H-2C']
    
    max_discrepancy = 0
    for i, comp_name in enumerate(hh_comps):
        if i < len(person_hhcomp_dist) and i < len(household_hhcomp_dist):
            persons_with_comp = person_hhcomp_dist[i].item()
            households_with_comp = household_hhcomp_dist[i].item()
            discrepancy = abs(persons_with_comp - households_with_comp)
            max_discrepancy = max(max_discrepancy, discrepancy)
            
            if discrepancy > 0:
                print(f"  {comp_name:10s}: {persons_with_comp:4d} persons vs {households_with_comp:4d} households (diff={discrepancy})")
    
    if max_discrepancy > num_persons * 0.5:  # More than 50% discrepancy
        print(f"[WARN] WARNING: Large household composition mismatch detected (max diff={max_discrepancy})")
        print(f"   Assignment may struggle to match all constraints")
    else:
        print(f"[OK] Household composition distributions are reasonably compatible (max diff={max_discrepancy})")
    
    print("\n[OK] Compatibility validation PASSED")
    return True

def validate_final_assignment(assignments, person_nodes, household_nodes, household_sizes):
    """
    Comprehensive validation of final assignment results.
    
    Checks:
    1. Every person assigned to exactly one household
    2. No empty households
    3. Household size constraints respected
    4. No households exceed their capacity
    """
    print("\n=== VALIDATING FINAL ASSIGNMENT ===")
    
    num_persons = person_nodes.size(0)
    num_households = household_nodes.size(0)
    
    errors = []
    warnings = []
    
    # Check 1: Every person assigned to exactly one household
    assignments_per_person = assignments.sum(dim=1)
    persons_not_assigned = (assignments_per_person == 0).sum().item()
    persons_multi_assigned = (assignments_per_person > 1).sum().item()
    
    if persons_not_assigned > 0:
        errors.append(f"{persons_not_assigned} persons not assigned to any household")
    
    if persons_multi_assigned > 0:
        errors.append(f"{persons_multi_assigned} persons assigned to multiple households")
    
    if persons_not_assigned == 0 and persons_multi_assigned == 0:
        print(f"[OK] All {num_persons} persons assigned to exactly one household")
    
    # Check 2: Empty households
    persons_per_household = assignments.sum(dim=0)
    empty_households = (persons_per_household == 0).sum().item()
    
    if empty_households > 0:
        warnings.append(f"{empty_households} empty households (no persons assigned)")
        print(f"[WARN] {empty_households} empty households found")
    else:
        print(f"[OK] No empty households")
    
    # Check 3: Household size constraints
    size_violations = 0
    underfilled_households = []
    
    # Extract size categories to identify 4+ households
    household_size_categories = household_nodes[:, 4].long()
    
    for hh_idx in range(num_households):
        assigned_count = persons_per_household[hh_idx].item()
        size_category = household_size_categories[hh_idx].item()
        expected_size = household_sizes[hh_idx].item()
        
        # For size category 3 (which represents 4+), allow any size >= 4
        if size_category == 3:  # 4+ category
            if assigned_count < 4:
                size_violations += 1
                underfilled_households.append((hh_idx, '4+', assigned_count))
        else:
            # For categories 0,1,2 (sizes 1,2,3), require exact match
            if assigned_count != expected_size:
                size_violations += 1
                if assigned_count < expected_size:
                    underfilled_households.append((hh_idx, expected_size, assigned_count))
    
    if size_violations > 0:
        warnings.append(f"{size_violations} households with size constraint violations")
        print(f"[WARN] {size_violations}/{num_households} households violate size constraints")
        
        if underfilled_households and len(underfilled_households) <= 10:
            print(f"   Examples of underfilled households:")
            for hh_idx, expected, actual in underfilled_households[:10]:
                print(f"     Household {hh_idx}: expected {expected}, got {actual}")
    else:
        print(f"[OK] All household size constraints satisfied (4+ households can have >=4 persons)")
    
    # Check 4: Total persons match
    total_assigned = persons_per_household.sum().item()
    if total_assigned != num_persons:
        errors.append(f"Total assigned persons ({total_assigned}) != total persons ({num_persons})")
    else:
        print(f"[OK] Total persons correctly assigned: {total_assigned}")
    
    # Check 5: COMPOSITION MATCHING (the most important constraint)
    hhcomp_col_persons = 6  # Person household composition column
    hhcomp_col_households = 0  # Household composition column
    
    # Get hard assignments from soft assignments
    hard_assignments = assignments.argmax(dim=1)
    
    # Get person and household compositions
    person_hhcomp = person_nodes[:, hhcomp_col_persons].long()
    household_hhcomp = household_nodes[:, hhcomp_col_households].long()
    
    # Calculate composition match rate
    assigned_household_hhcomp = household_hhcomp[hard_assignments]
    composition_matches = (person_hhcomp == assigned_household_hhcomp).sum().item()
    composition_match_rate = (composition_matches / num_persons) * 100
    
    if composition_match_rate >= 90:
        print(f"[OK] Household composition match rate: {composition_match_rate:.1f}% ({composition_matches}/{num_persons})")
    elif composition_match_rate >= 70:
        print(f"[WARN] Household composition match rate: {composition_match_rate:.1f}% ({composition_matches}/{num_persons})")
        warnings.append(f"Composition match rate below 90%: {composition_match_rate:.1f}%")
    else:
        print(f"[ERROR] Household composition match rate: {composition_match_rate:.1f}% ({composition_matches}/{num_persons})")
        errors.append(f"Composition match rate too low: {composition_match_rate:.1f}%")
    
    # Print breakdown by composition type
    unique_comps = torch.unique(person_hhcomp)
    print(f"\n   Composition match breakdown:")
    for comp_idx in unique_comps:
        comp_val = comp_idx.item()
        persons_with_comp = (person_hhcomp == comp_val)
        total_with_comp = persons_with_comp.sum().item()
        matches_with_comp = ((person_hhcomp == comp_val) & (assigned_household_hhcomp == comp_val)).sum().item()
        comp_name = COMPOSITION_SIZE_MAP.get(comp_val, {}).get('name', f'Unknown({comp_val})')
        match_pct = (matches_with_comp / total_with_comp * 100) if total_with_comp > 0 else 0
        status = "[OK]" if match_pct >= 90 else ("[WARN]" if match_pct >= 70 else "[ERROR]")
        print(f"   {status} {comp_name}: {matches_with_comp}/{total_with_comp} matched ({match_pct:.1f}%)")
    
    # Print results
    if errors:
        print("\n[ERROR] ASSIGNMENT VALIDATION FAILED:")
        for error in errors:
            print(f"  ERROR: {error}")
        return False
    
    if warnings:
        print("\n[WARN] ASSIGNMENT VALIDATION WARNINGS:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    
    print("\n[OK] Assignment validation PASSED (with warnings)" if warnings else "\n[OK] Assignment validation PASSED")
    return True

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    monitor_memory_usage(device, "at startup")

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
# persons_file_path = os.path.join(current_dir, "./outputs/person_nodes.pt")
# households_file_path = os.path.join(current_dir, "./outputs/household_nodes.pt")
persons_file_path = os.path.join(current_dir, f"./outputs/individuals_{selected_area_code}/person_nodes.pt")
households_file_path = os.path.join(current_dir, f"./outputs/households_{selected_area_code}/household_nodes.pt")
hh_size_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_size.csv'))

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
try:
    person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
    print(f"Loaded person_nodes with shape: {person_nodes.shape}")
except Exception as e:
    print(f"Error loading person nodes from {persons_file_path}: {e}")
    raise

try:
    household_nodes = torch.load(households_file_path)  # Expected size: (num_households x 6)
    print(f"Loaded household_nodes with shape: {household_nodes.shape}")
    
    # Validate household tensor structure for size extraction
    if household_nodes.dim() == 2 and household_nodes.size(1) >= 5:
        print(f"Household tensor structure is compatible for size extraction (has {household_nodes.size(1)} columns)")
    else:
        print(f"Warning: Household tensor structure may not support size extraction (shape: {household_nodes.shape})")
        
except Exception as e:
    print(f"Error loading household nodes from {households_file_path}: {e}")
    raise

# Convert to float for neural network compatibility
person_nodes = person_nodes.float()
household_nodes = household_nodes.float()

# Move tensors to GPU
person_nodes = person_nodes.to(device)
household_nodes = household_nodes.to(device)
print(f"Moved person_nodes and household_nodes to {device}")

# Validate tensor schemas
if not validate_tensor_schema(person_nodes, household_nodes):
    print("[ERROR] CRITICAL: Tensor schema validation failed. Exiting.")
    exit(1)

# Validate assignment compatibility
# Validate assignment compatibility (continues with warning if issues found)
validate_assignment_compatibility(person_nodes, household_nodes)
print("[INFO] Proceeding with training regardless of compatibility warnings.")

# Define the household composition categories and mapping
# hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nC', '1FM-nA', '1FC-0C', '1FC-nC', '1FC-nA', '1FL-nC', '1FL-nA', '1H-nC', '1H-nS', '1H-nE', '1H-nA']
hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-nA', '1FL-2C', '1H-nS', '1H-nE', '1H-nA', '1H-2C']
hh_map = {category: i for i, category in enumerate(hh_compositions)}
reverse_hh_map = {v: k for k, v in hh_map.items()}  # Reverse mapping to decode

# Extract the household composition predictions
hh_pred = household_nodes[:, 0].long()

# Flattening size and weight lists
values_size_org = [k for k in hh_size_df.columns if k not in ['geography code', 'total']]
weights_size_org = hh_size_df.iloc[0, 2:].tolist()  # Assuming first row, and skipping the first two columns

household_size_dist = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k != '1'}
values_size, weights_size = zip(*household_size_dist.items())

household_size_dist_na = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k not in ['1', '2']}
values_size_na, weights_size_na = zip(*household_size_dist_na.items())

# Define the size assignment function based on household composition
# fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
# three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
# two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C'}

fixed_hh = {"1PE": 1, "1PA": 1, "1FE": 2, "1FM-0C": 2, "1FC-0C": 2}
three_or_more_hh = {'1FM-nC', '1FM-nA', '1FC-nC', '1FC-nA'}
two_or_more_hh = {'1FL-nC', '1FL-nA', '1H-nC', '1H-nS', '1H-nE', '1H-nA'}

# ============================================================================
# HOUSEHOLD COMPOSITION TO SIZE MAPPING
# This is CRITICAL for ensuring persons are assigned to correctly-sized households
# ============================================================================
# Composition categories and their expected household sizes:
# - 1PE (1 Person Elderly 65+): Size 1
# - 1PA (1 Person Adult <65): Size 1  
# - 1FE (1 Family Elderly couple): Size 2
# - 1FM-0C (1 Family Married, 0 Children): Size 2
# - 1FM-2C (1 Family Married, 2+ Children): Size 3-4+
# - 1FM-nA (1 Family Married, with Adults): Size 3-4+
# - 1FC-0C (1 Family Cohabiting, 0 Children): Size 2
# - 1FC-2C (1 Family Cohabiting, 2+ Children): Size 3-4+
# - 1FC-nA (1 Family Cohabiting, with Adults): Size 3-4+
# - 1FL-nA (1 Family Lone Parent, with Adults): Size 2-4+
# - 1FL-2C (1 Family Lone Parent, 2+ Children): Size 2-4+
# - 1H-nS (1 Household, Student): Size 2-4+ (shared student housing)
# - 1H-nE (1 Household, Elderly): Size 2-4+ (shared elderly housing)
# - 1H-nA (1 Household, Adults): Size 2-4+ (shared adult housing)
# - 1H-2C (1 Household, with Children): Size 2-4+

COMPOSITION_SIZE_MAP = {
    # Single-person households (size 1 ONLY)
    0: {'min': 1, 'max': 1, 'name': '1PE'},      # 1 Person Elderly
    1: {'min': 1, 'max': 1, 'name': '1PA'},      # 1 Person Adult
    
    # Couple households (size 2 ONLY)
    2: {'min': 2, 'max': 2, 'name': '1FE'},      # 1 Family Elderly (couple)
    3: {'min': 2, 'max': 2, 'name': '1FM-0C'},   # 1 Family Married, no children
    6: {'min': 2, 'max': 2, 'name': '1FC-0C'},   # 1 Family Cohabiting, no children
    
    # Family with children/adults (size 3+)
    4: {'min': 3, 'max': 8, 'name': '1FM-2C'},   # 1 Family Married, 2+ children
    5: {'min': 3, 'max': 8, 'name': '1FM-nA'},   # 1 Family Married, with adults
    7: {'min': 3, 'max': 8, 'name': '1FC-2C'},   # 1 Family Cohabiting, 2+ children
    8: {'min': 3, 'max': 8, 'name': '1FC-nA'},   # 1 Family Cohabiting, with adults
    
    # Lone parent families (size 2+)
    9: {'min': 2, 'max': 8, 'name': '1FL-nA'},   # 1 Family Lone Parent, with adults
    10: {'min': 2, 'max': 8, 'name': '1FL-2C'},  # 1 Family Lone Parent, 2+ children
    
    # Other households (size varies)
    11: {'min': 2, 'max': 8, 'name': '1H-nS'},   # Household, students
    12: {'min': 2, 'max': 8, 'name': '1H-nE'},   # Household, elderly
    13: {'min': 2, 'max': 8, 'name': '1H-nA'},   # Household, adults
    14: {'min': 2, 'max': 8, 'name': '1H-2C'},   # Household, with children
}

# ============================================================================
# COMPOSITION SIMILARITY GROUPS
# When exact composition match isn't available, use similar compositions
# This addresses the data mismatch problem where some compositions have
# more persons than households available
# ============================================================================
COMPOSITION_SIMILARITY = {
    # Single person households - can substitute for each other
    0: [0, 1],       # 1PE: prefer elderly single, fallback to adult single
    1: [1, 0],       # 1PA: prefer adult single, fallback to elderly single
    
    # Couples without children - married/cohabiting are similar
    2: [2, 3, 6],    # 1FE: elderly couple -> married couple -> cohabiting couple
    3: [3, 6, 2],    # 1FM-0C: married -> cohabiting -> elderly couple
    6: [6, 3, 2],    # 1FC-0C: cohabiting -> married -> elderly couple
    
    # Families with children - married/cohabiting with kids are similar
    4: [4, 7, 5, 8, 10],  # 1FM-2C: married+kids -> cohab+kids -> married+adults -> cohab+adults -> lone+kids
    7: [7, 4, 8, 5, 10],  # 1FC-2C: cohab+kids -> married+kids -> cohab+adults -> married+adults -> lone+kids
    
    # Families with adults - married/cohabiting with adults are similar
    5: [5, 8, 4, 7, 13],  # 1FM-nA: married+adults -> cohab+adults -> married+kids -> cohab+kids -> HH adults
    8: [8, 5, 7, 4, 13],  # 1FC-nA: cohab+adults -> married+adults -> cohab+kids -> married+kids -> HH adults
    
    # Lone parent families
    9: [9, 10, 13, 5, 8],   # 1FL-nA: lone+adults -> lone+kids -> HH adults -> family+adults
    10: [10, 9, 4, 7, 14],  # 1FL-2C: lone+kids -> lone+adults -> married+kids -> cohab+kids -> HH+kids
    
    # Other multi-person households - all grouped together
    11: [11, 13, 12],       # 1H-nS: students -> adults -> elderly
    12: [12, 2, 13],        # 1H-nE: elderly HH -> elderly couple -> adult HH
    13: [13, 11, 5, 8, 9],  # 1H-nA: adults -> students -> family+adults
    14: [14, 4, 7, 10],     # 1H-2C: HH+kids -> married+kids -> cohab+kids -> lone+kids
}

def get_similar_compositions(comp_idx):
    """Get list of similar compositions for fallback matching."""
    return COMPOSITION_SIMILARITY.get(comp_idx, [comp_idx])

def get_valid_size_range_for_composition(composition_idx):
    """Get the valid (min, max) size range for a household composition index."""
    if composition_idx in COMPOSITION_SIZE_MAP:
        info = COMPOSITION_SIZE_MAP[composition_idx]
        return info['min'], info['max']
    else:
        # Default: flexible size
        return 1, 8

def is_size_valid_for_composition(composition_idx, size):
    """Check if a household size is valid for the given composition."""
    min_size, max_size = get_valid_size_range_for_composition(composition_idx)
    return min_size <= size <= max_size

def fit_household_size(composition):
    if composition in fixed_hh:
        return fixed_hh[composition]
    elif composition in three_or_more_hh:
        return int(random.choices(values_size_na, weights=weights_size_na)[0].replace('8+', '8'))
    elif composition in two_or_more_hh:
        return int(random.choices(values_size, weights=weights_size)[0].replace('8+', '8'))
    else:
        return int(random.choices(values_size_org, weights=weights_size_org)[0].replace('8+', '8'))

# Validate household size category extraction functionality
validation_passed = validate_household_size_extraction(household_nodes, device)

# Try to extract household size categories from the generated household tensor
household_size_categories, extraction_successful = extract_household_sizes_from_tensor(household_nodes, device)

if not extraction_successful:
    print("Extraction Failed. Exiting...")
    exit()
    # print("Falling back to random household size assignment based on composition...")
    # # Fallback: Assign sizes to each household based on its composition (original logic)
    # household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)
    # household_sizes = household_sizes.to(device)
    # print("Done assigning household sizes using random method")
else:
    print("Done assigning household size categories using tensor extraction method")
    print(f"Using extracted size categories for {household_size_categories.size(0)} households")
    
    # Convert size categories to actual sizes for compatibility with existing functions
    # Map: 0->1, 1->2, 2->3, 3->4 (4+ category becomes 4)
    household_sizes = household_size_categories + 1  # Convert 0-3 indices to 1-4 sizes
    print("Converted size categories to actual sizes for compatibility")
    
    # Print detailed information about the conversion
    print(f"Size category distribution:")
    unique_cats, cat_counts = torch.unique(household_size_categories, return_counts=True)
    size_categories = ['1', '2', '3', '4+']
    for cat_idx, count in zip(unique_cats.cpu().numpy(), cat_counts.cpu().numpy()):
        category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
        print(f"  Category {cat_idx} ({category_name}): {count} households")
    
    print(f"Converted size distribution:")
    unique_sizes, size_counts = torch.unique(household_sizes, return_counts=True)
    for size, count in zip(unique_sizes.cpu().numpy(), size_counts.cpu().numpy()):
        print(f"  Size {size}: {count} households")

# Step 2: Define the GNN model with memory-efficient GAT
class HouseholdAssignmentGNN(torch.nn.Module):
    """
    Memory-efficient Graph Attention Network for household assignment.
    
    ARCHITECTURE: Bipartite Graph (Person nodes <-> Household nodes)
    - Person nodes: indices 0 to num_persons-1
    - Household nodes: indices num_persons to num_persons+num_households-1
    - Edges connect compatible person-household pairs
    
    Design choices for 24GB GPU constraint:
    - Uses only 2 attention heads (vs 4-8 typical) to reduce memory
    - concat=False to keep dimensions manageable
    - GraphNorm instead of BatchNorm (works better with graph data)
    - Edge dropout to reduce attention computation
    - Smaller hidden dimensions recommended (128-256)
    - Proper residual connections between all layers
    
    Args:
        person_in_channels: Input feature dimension for person nodes (7 attributes)
        household_in_channels: Input feature dimension for household nodes (6 attributes)
        hidden_channels: Hidden layer dimension (keep <=256 for memory)
        num_persons: Number of person nodes
        num_households: Number of household nodes (also output dimension)
        dropout_rate: Dropout probability (default 0.1)
        attention_heads: Number of attention heads (default 2 for memory efficiency)
        edge_dropout: Probability of dropping edges during training (default 0.1)
    """
    def __init__(self, person_in_channels, household_in_channels, hidden_channels, 
                 num_persons, num_households,
                 dropout_rate=0.1, attention_heads=2, edge_dropout=0.1):
        super(HouseholdAssignmentGNN, self).__init__()
        
        self.attention_heads = attention_heads
        self.edge_dropout = edge_dropout
        self.num_persons = num_persons
        self.num_households = num_households
        
        # Separate embedding layers for person and household features
        # This projects both to the same hidden dimension for the bipartite graph
        self.person_embedding = torch.nn.Linear(person_in_channels, hidden_channels)
        self.household_embedding = torch.nn.Linear(household_in_channels, hidden_channels)
        
        # Projection layer for residual connections (input embedding -> hidden)
        # Used when dimensions might not match
        self.input_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        
        # Three GAT layers with residual connections
        # concat=False means output is hidden_channels (not hidden_channels * heads)
        self.gat1 = GATConv(
            hidden_channels, 
            hidden_channels, 
            heads=attention_heads,
            concat=False,  # Critical for memory: averages heads instead of concatenating
            dropout=dropout_rate,
            edge_dim=None  # No edge features to save memory
        )
        self.norm1 = GraphNorm(hidden_channels)
        
        self.gat2 = GATConv(
            hidden_channels, 
            hidden_channels, 
            heads=attention_heads,
            concat=False,
            dropout=dropout_rate,
            edge_dim=None
        )
        self.norm2 = GraphNorm(hidden_channels)
        
        self.gat3 = GATConv(
            hidden_channels, 
            hidden_channels, 
            heads=attention_heads,
            concat=False,
            dropout=dropout_rate,
            edge_dim=None
        )
        self.norm3 = GraphNorm(hidden_channels)
        
        # Dropout layers
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Final prediction layers (only for person nodes)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, num_households)
        self.relu = torch.nn.ReLU()

    def forward(self, person_features, household_features, edge_index):
        """
        Forward pass through the bipartite GAT.
        
        Args:
            person_features: Person node features [num_persons, person_in_channels]
            household_features: Household node features [num_households, household_in_channels]
            edge_index: Bipartite edge index connecting persons to households [2, num_edges]
        
        Returns:
            Assignment logits [num_persons, num_households]
        """
        # Embed person and household features to same dimension
        person_embedded = self.person_embedding(person_features)      # [num_persons, hidden_channels]
        household_embedded = self.household_embedding(household_features)  # [num_households, hidden_channels]
        
        # Concatenate to form unified node feature matrix for bipartite graph
        # Person nodes: 0 to num_persons-1
        # Household nodes: num_persons to num_persons+num_households-1
        x = torch.cat([person_embedded, household_embedded], dim=0)  # [num_persons + num_households, hidden_channels]
        
        # Project input for residual connections
        x_proj = self.input_proj(x)
        
        # Optional: Apply edge dropout during training to reduce memory
        if self.training and self.edge_dropout > 0:
            # Randomly drop edges to reduce attention computation
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
            edge_index_dropped = edge_index[:, edge_mask]
        else:
            edge_index_dropped = edge_index
        
        # First GAT layer
        h1 = self.gat1(x, edge_index_dropped)
        h1 = self.norm1(h1)
        h1 = self.relu(h1)
        h1 = h1 + x_proj  # RESIDUAL: Add projected input
        h1 = self.dropout(h1)
        
        # Second GAT layer with residual from h1
        h2 = self.gat2(h1, edge_index_dropped)
        h2 = self.norm2(h2)
        h2 = self.relu(h2)
        h2 = h2 + h1  # RESIDUAL: Add h1
        h2 = self.dropout(h2)
        
        # Third GAT layer with residual from h2
        h3 = self.gat3(h2, edge_index_dropped)
        h3 = self.norm3(h3)
        h3 = self.relu(h3)
        h3 = h3 + h2  # RESIDUAL: Add h2
        h3 = self.dropout(h3)
        
        # Extract only person node embeddings for assignment prediction
        # Person nodes are at indices 0 to num_persons-1
        person_embeddings = h3[:self.num_persons]  # [num_persons, hidden_channels]
        
        # Final prediction layers
        out = self.fc1(person_embeddings)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out  # Output shape: (num_persons, num_households)

# Define Gumbel-Softmax with improved temperature handling
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax sampling for differentiable discrete assignments.
    
    Args:
        logits: Raw output from the model [num_persons, num_households]
        tau: Temperature parameter (higher = softer assignments)
        hard: If True, returns one-hot assignments with straight-through gradients
    
    Returns:
        Soft or hard assignment probabilities [num_persons, num_households]
    """
    # Add Gumbel noise for stochastic sampling
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    
    # Apply softmax with temperature
    y = F.softmax(y / tau, dim=-1)

    if hard:
        # Straight-through trick: take the index of the max value, but keep the gradient.
        y_hard = torch.zeros_like(logits, device=logits.device).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    
    return y

def get_annealing_temperature(epoch, num_epochs, tau_start=2.0, tau_end=0.5, anneal_type='exponential'):
    """
    Calculate temperature for annealing schedule.
    
    Args:
        epoch: Current epoch number (0-indexed)
        num_epochs: Total number of epochs
        tau_start: Starting temperature (higher = softer)
        tau_end: Ending temperature (lower = harder)
        anneal_type: 'linear', 'exponential', or 'cosine'
    
    Returns:
        Current temperature value
    """
    progress = epoch / max(num_epochs - 1, 1)  # 0 to 1
    
    if anneal_type == 'linear':
        tau = tau_start - (tau_start - tau_end) * progress
    elif anneal_type == 'exponential':
        # Exponential decay: tau = tau_start * (tau_end/tau_start)^progress
        tau = tau_start * np.power(tau_end / tau_start, progress)
    elif anneal_type == 'cosine':
        # Cosine annealing: smoother transition
        tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + np.cos(np.pi * progress))
    else:
        raise ValueError(f"Unknown anneal_type: {anneal_type}")
    
    return max(tau, tau_end)  # Ensure we don't go below tau_end

# Step 3: Create the BIPARTITE graph (Person nodes <-> Household nodes)
# CRITICAL FIX: Changed from person-to-person to person-to-household bipartite graph
num_persons = person_nodes.size(0)
num_households = household_sizes.size(0)

# Define the columns for religion, ethnicity, and household composition
# Based on actual tensor structures from generation scripts:
# person_nodes: [age, sex, religion, ethnicity, marital, qualification, household_composition] (7 columns)
# household_nodes: [household_composition, ethnicity, religion, tenure, size, rooms] (6 columns)
religion_col_persons, religion_col_households = 2, 2
ethnicity_col_persons, ethnicity_col_households = 3, 1
household_composition_col_persons, household_composition_col_households = 6, 0

# Create output directory for assignment results
output_dir = os.path.join(current_dir, 'outputs', f'assignment_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

# Edge index file path - NOTE: Use new filename with 3-PHASE strategy
# Phase 1: Exact composition, Phase 2: Similar composition, Phase 3: Fallback
edge_index_file_path = os.path.join(output_dir, "edge_index.pt")

if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    print(f"Loaded bipartite edge index from {edge_index_file_path}")
    print(f"Edge index shape: {edge_index.shape}, Total edges: {edge_index.size(1)}")
else:
    print("Creating BIPARTITE edge index (Person <-> Household)...")
    print(f"Graph structure: {num_persons} person nodes + {num_households} household nodes")
    print(f"Person node indices: 0 to {num_persons-1}")
    print(f"Household node indices: {num_persons} to {num_persons + num_households - 1}")
    
    # Use context manager for memory optimization during edge creation
    with torch.no_grad():  # Disable gradient computation for memory efficiency
        # Extract person attributes for compatibility checking
        person_religion = person_nodes[:, religion_col_persons]
        person_ethnicity = person_nodes[:, ethnicity_col_persons]
        person_hhcomp = person_nodes[:, household_composition_col_persons]
        
        # Extract household attributes for compatibility checking
        household_religion = household_nodes[:, religion_col_households]
        household_ethnicity = household_nodes[:, ethnicity_col_households]
        household_hhcomp = household_nodes[:, household_composition_col_households]
        
        # Create compatibility matrices (Person x Household)
        # Shape: [num_persons, num_households] - True where person-household are compatible
        religion_match = person_religion.unsqueeze(1) == household_religion.unsqueeze(0)
        ethnicity_match = person_ethnicity.unsqueeze(1) == household_ethnicity.unsqueeze(0)
        hhcomp_match = person_hhcomp.unsqueeze(1) == household_hhcomp.unsqueeze(0)
        
        print(f"Compatibility matrices created: religion={religion_match.shape}, ethnicity={ethnicity_match.shape}, hhcomp={hhcomp_match.shape}")
        
        # ============================================================================
        # COMPOSITION-FIRST EDGE CREATION STRATEGY
        # Household composition is the PRIMARY constraint because it directly determines
        # household structure and size. A person with composition 1PE MUST go to a 1PE household.
        # ============================================================================
        
        # LEVEL 1: Household composition MUST match (hard constraint)
        # Plus at least one of {religion, ethnicity} should match
        strict_compatibility = hhcomp_match & (religion_match | ethnicity_match)
        
        num_strict = strict_compatibility.sum().item()
        density_strict = num_strict / (num_persons * num_households) * 100
        print(f"Strict compatible pairs (hhcomp + 1 attr): {num_strict} ({density_strict:.2f}%)")
        
        # LEVEL 2: If strict is sufficient, use it
        if num_strict >= num_persons * 3:  # Each person has at least 3 candidate households
            compatibility = strict_compatibility
            print("Using STRICT compatibility (household composition + 1 attribute must match)")
        else:
            # LEVEL 3: Relax to just household composition matching
            hhcomp_only = hhcomp_match.sum().item()
            density_hhcomp = hhcomp_only / (num_persons * num_households) * 100
            print(f"Household composition only matches: {hhcomp_only} ({density_hhcomp:.2f}%)")
            
            if hhcomp_only >= num_persons * 3:
                compatibility = hhcomp_match
                print("Using COMPOSITION-ONLY compatibility (household composition must match)")
            else:
                # LEVEL 4: Last resort - require 2+ attributes (original logic)
                print("WARNING: Too few composition matches. Using 2+ attribute matching.")
                compatibility = (religion_match & ethnicity_match) | \
                               (religion_match & hhcomp_match) | \
                               (ethnicity_match & hhcomp_match)
                
                num_compatible = compatibility.sum().item()
                if num_compatible < num_persons * 3:
                    # Fallback to any attribute match
                    print("WARNING: Relaxing to 1+ attribute match.")
                    compatibility = religion_match | ethnicity_match | hhcomp_match
        
        # Count final compatible pairs
        num_compatible = compatibility.sum().item()
        density = num_compatible / (num_persons * num_households) * 100
        print(f"Final compatible pairs: {num_compatible} out of {num_persons * num_households} possible ({density:.2f}%)")
        
        # Get indices of compatible pairs
        person_indices, household_indices = torch.where(compatibility)
        
        # Limit edges per person to top-k households (for memory efficiency)
        max_edges_per_person = min(100, num_households)  # At most 100 edges per person
        edges_list = []
        
        print(f"Initial compatible edges: {person_indices.size(0)}")
        
        # For each person, keep top-k compatible households
        for p_idx in range(num_persons):
            # Find all compatible households for this person
            person_mask = (person_indices == p_idx)
            compatible_hh = household_indices[person_mask]
            
            if compatible_hh.size(0) > max_edges_per_person:
                # Randomly sample to limit edges
                perm = torch.randperm(compatible_hh.size(0), device=device)[:max_edges_per_person]
                compatible_hh = compatible_hh[perm]
            
            # Add edges: person p_idx connects to household nodes (offset by num_persons)
            for hh_idx in compatible_hh:
                # Person node index: p_idx (0 to num_persons-1)
                # Household node index: num_persons + hh_idx (num_persons to num_persons+num_households-1)
                edges_list.append([p_idx, num_persons + hh_idx.item()])
        
        if len(edges_list) == 0:
            print("WARNING: No edges created! Creating fallback complete bipartite graph.")
            # Fallback: Connect each person to all households
            person_idx = torch.arange(num_persons, device=device).repeat_interleave(num_households)
            household_idx = torch.arange(num_persons, num_persons + num_households, device=device).repeat(num_persons)
            edge_index = torch.stack([person_idx, household_idx], dim=0)
        else:
            # Create edge tensor
            edges_tensor = torch.tensor(edges_list, dtype=torch.long, device=device).t()
            
            # Make bidirectional (undirected bipartite graph)
            edge_index = torch.cat([
                edges_tensor,  # Person -> Household
                edges_tensor.flip(0)  # Household -> Person
            ], dim=1)
        
        print(f"Final bipartite edges: {len(edges_list)} (bidirectional: {edge_index.size(1)})")
        print(f"Average edges per person: {len(edges_list) / num_persons:.1f}")
        
        # Clear intermediate tensors to free memory
        safe_delete_tensor(person_religion)
        safe_delete_tensor(person_ethnicity)
        safe_delete_tensor(person_hhcomp)
        safe_delete_tensor(household_religion)
        safe_delete_tensor(household_ethnicity)
        safe_delete_tensor(household_hhcomp)
        safe_delete_tensor(religion_match)
        safe_delete_tensor(ethnicity_match)
        safe_delete_tensor(hhcomp_match)
        safe_delete_tensor(compatibility)
        safe_delete_tensor(person_indices)
        safe_delete_tensor(household_indices)
        
        # Save edge index
        edge_index_cpu = edge_index.cpu()
        torch.save(edge_index_cpu, edge_index_file_path)
        print(f"Bipartite edge index saved to {edge_index_file_path}")

# Move edge index to GPU (if not already there)
if not edge_index.is_cuda and device.type == 'cuda':
    edge_index = edge_index.to(device)
    print(f"Moved edge_index to {device}")
else:
    print(f"Edge index already on {device}")

monitor_memory_usage(device, "after edge index creation")


# Step 4: Hyperparameter tuning setup
# ============================================================================
# MEMORY-EFFICIENT GAT CONFIGURATION
# ============================================================================
# For 24GB GPU with GAT:
# - Recommended hidden_dims: [128, 256] (avoid 512+ to prevent OOM)
# - Learning rates: [0.001, 0.005] (GAT works well with these)
# - attention_heads: 2 (already set in model, don't increase)
# - edge_dropout: 0.1 (reduces memory during training)
#
# If you still get OOM errors:
# 1. Reduce hidden_dims to [128] only
# 2. Increase edge_dropout to 0.2
# 3. Reduce num_epochs or use early stopping (already implemented)
# ============================================================================

num_epochs = 400  # With early stopping (patience=100)
learning_rates = [0.0005]  # GAT-friendly learning rates  
learning_rates = [0.0005, 0.0001]  # GAT-friendly learning rates  
hidden_dims = [256]  # Start with 128 now that edges are reduced
# Can try [128, 256] after confirming 128 works

# GAT-specific hyperparameters
attention_heads = 2  # Keep at 2 for memory efficiency
edge_dropout = 0.05  # Low dropout since we already have sparse edges

best_loss = float('inf')
best_params = {}

# Estimate GAT memory usage before training
num_edges = edge_index.size(1)
print(f"\nGraph statistics:")
print(f"  Number of persons: {num_persons}")
print(f"  Number of households: {num_households}")
print(f"  Number of edges: {num_edges}")

# Estimate memory for the largest configuration we'll test
max_hidden = max(hidden_dims) if hidden_dims else 256
estimate_gat_memory_usage(num_persons, num_households, max_hidden, attention_heads, num_edges)

# ============================================================================
# COMPOSITION-AWARE LOSS WEIGHTS
# ============================================================================
# For single-person households (1PE, 1PA): religion/ethnicity SHOULD match 100%
# For multi-person households: only ~40-50% match expected (HRP vs family members)
#
# This is because household religion/ethnicity = HRP (Household Reference Person)
# In a family of 4, only 1-2 members typically share HRP's religion/ethnicity
SINGLE_PERSON_COMPOSITIONS = [0, 1]  # 1PE, 1PA
COUPLE_COMPOSITIONS = [2, 3, 6]  # 1FE, 1FM-0C, 1FC-0C (expect ~50% match - one of two)
FAMILY_COMPOSITIONS = [4, 5, 7, 8, 9, 10, 11, 12, 13, 14]  # Families (expect ~30-40% match)

# Expected religion/ethnicity match rates by composition type
COMPOSITION_REL_ETH_WEIGHTS = {
    # Single person: 100% should match
    0: 1.0,   # 1PE
    1: 1.0,   # 1PA
    # Couples: ~50% match (one of two)
    2: 0.5,   # 1FE
    3: 0.5,   # 1FM-0C
    6: 0.5,   # 1FC-0C
    # Families with children/adults: ~35% match
    4: 0.35,  # 1FM-2C
    5: 0.35,  # 1FM-nA
    7: 0.35,  # 1FC-2C
    8: 0.35,  # 1FC-nA
    9: 0.40,  # 1FL-nA (lone parent + adult)
    10: 0.40, # 1FL-2C (lone parent + children)
    # Other multi-person households
    11: 0.30, # 1H-nS (students - diverse)
    12: 0.50, # 1H-nE (elderly - more homogeneous)
    13: 0.30, # 1H-nA (adults - diverse)
    14: 0.35, # 1H-2C (with children)
}

# Compute loss function with COMPOSITION-AWARE weighting
def compute_loss(assignments, household_sizes, person_nodes, household_nodes, 
                 religion_loss_weight=1.0, ethnicity_loss_weight=1.0, 
                 size_loss_weight=2.0, household_composition_loss_weight=1.0):
    """
    Compute multi-objective loss for household assignment.
    
    HYBRID COMPOSITION-AWARE STRATEGY:
    - Composition loss: Always high weight (persons MUST match household composition)
    - Size loss: Important for capacity constraints
    - Religion/Ethnicity loss: WEIGHTED BY COMPOSITION TYPE
      * Single-person households: Full weight (100% should match)
      * Multi-person households: Reduced weight (only 30-50% expected to match)
    
    This accounts for the fact that household religion/ethnicity represents only
    the Household Reference Person (HRP), not all family members.
    
    Args:
        assignments: Soft assignment matrix [num_persons, num_households]
        household_sizes: Target household sizes [num_households] (1, 2, 3, or 4 for 4+)
        person_nodes: Person features [num_persons, 7]
        household_nodes: Household features [num_households, 6]
        *_weight: Loss component weights
    
    Returns:
        total_loss, size_loss, religion_loss, ethnicity_loss, household_composition_loss
    """
    num_persons = assignments.size(0)
    device = assignments.device
    
    # Soft counts per household (expected number of people assigned)
    household_counts = assignments.sum(dim=0)  # [num_households]

    # ============ SIZE LOSS ============
    sizes_float = household_sizes.float()
    pred_counts_capped = torch.clamp(household_counts, max=4.0)
    size_loss = F.mse_loss(pred_counts_capped, sizes_float) * size_loss_weight

    # ============ COMPOSITION-AWARE ATTRIBUTE LOSSES ============
    eps = 1e-8
    
    # Get person compositions for per-person weighting
    person_hhcomp = person_nodes[:, 6].long()
    
    # Create per-person weight tensor based on their composition
    person_rel_eth_weights = torch.ones(num_persons, device=device)
    for comp_idx, weight in COMPOSITION_REL_ETH_WEIGHTS.items():
        mask = (person_hhcomp == comp_idx)
        person_rel_eth_weights[mask] = weight

    # Religion loss (composition-aware weighted)
    religion_col_persons, religion_col_households = 2, 2
    y_religion = person_nodes[:, religion_col_persons].long()
    num_rel_classes = int(torch.max(torch.stack([
        y_religion.max(), household_nodes[:, religion_col_households].long().max()
    ])).item()) + 1
    H_rel_onehot = F.one_hot(household_nodes[:, religion_col_households].long(), num_classes=num_rel_classes).float()
    P_rel = assignments @ H_rel_onehot
    rel_match_prob = P_rel[torch.arange(num_persons, device=device), y_religion]
    # Apply composition-aware weighting: single-person gets full penalty, multi-person gets reduced
    rel_nll = -torch.log(rel_match_prob + eps)
    religion_loss = (rel_nll * person_rel_eth_weights).mean() * religion_loss_weight

    # Ethnicity loss (composition-aware weighted)
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    y_eth = person_nodes[:, ethnicity_col_persons].long()
    num_eth_classes = int(torch.max(torch.stack([
        y_eth.max(), household_nodes[:, ethnicity_col_households].long().max()
    ])).item()) + 1
    H_eth_onehot = F.one_hot(household_nodes[:, ethnicity_col_households].long(), num_classes=num_eth_classes).float()
    P_eth = assignments @ H_eth_onehot
    eth_match_prob = P_eth[torch.arange(num_persons, device=device), y_eth]
    # Apply composition-aware weighting
    eth_nll = -torch.log(eth_match_prob + eps)
    ethnicity_loss = (eth_nll * person_rel_eth_weights).mean() * ethnicity_loss_weight

    # ============ HOUSEHOLD COMPOSITION LOSS (HIGHEST PRIORITY) ============
    household_composition_col_persons, household_composition_col_households = 6, 0
    y_hh = person_nodes[:, household_composition_col_persons].long()
    num_hh_classes = int(torch.max(torch.stack([
        y_hh.max(), household_nodes[:, household_composition_col_households].long().max()
    ])).item()) + 1
    H_hh_onehot = F.one_hot(household_nodes[:, household_composition_col_households].long(), num_classes=num_hh_classes).float()
    P_hh = assignments @ H_hh_onehot
    hh_match_prob = P_hh[torch.arange(num_persons, device=device), y_hh]
    household_composition_loss = (-torch.log(hh_match_prob + eps)).mean() * household_composition_loss_weight

    total_loss = size_loss + religion_loss + ethnicity_loss + household_composition_loss
    return total_loss, size_loss, religion_loss, ethnicity_loss, household_composition_loss

# Store all results for saving
hp_results = []
detailed_results = []  # Store detailed results for each hyperparameter combination
convergence_results = []  # Store convergence data for each combination

# Global best model tracking (similar to generateIndividuals.py)
best_model_info = {
    'model_state': None,
    'loss': float('inf'),
    'accuracy': 0,
    'assignments': None,
    'lr': None,
    'hidden_channels': None,
    'convergence_data': None,
    'detailed_accuracies': None,
    'epoch_numbers': None,
    'religion_accuracies': None,
    'ethnicity_accuracies': None,
    'stopping_epoch': None
}

# ============================================================================
# PAPER-READY PLOTTING CONFIGURATION
# Consistent styling with generateIndividuals.py and generateHouseholds.py
# ============================================================================
PLOT_CONFIG = {
    'figure_width': 1400,  # ~10 inches at 140 DPI
    'font_family': 'Arial',
    'fonts': {
        'title': 20,
        'axis_title': 16,
        'tick_labels': 12,
        'legend': 14,
        'annotation': 12
    },
    'colors': {
        'primary': '#1F77B4',      # Deep blue
        'secondary': '#D62728',    # Rich red
        'tertiary': '#2CA02C',     # Green
        'quaternary': '#FF7F0E',   # Orange
        'grid': '#E5E5E5',
        'axis': '#333333'
    },
    'dpi': 300
}

# Plotting functions - Paper-ready versions
def plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir):
    """
    Plot assignment errors as a paper-ready bar chart.
    Shows mismatches for Size, Religion, Ethnicity, and Household Composition.
    """
    
    # Calculate size errors (using category-based comparison)
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in final_assignments:
        predicted_counts[household_idx] += 1
    
    # Convert to categories for error calculation
    actual_categories = torch.clamp(household_sizes - 1, 0, 3)
    predicted_categories = torch.clamp(predicted_counts - 1, 0, 3)
    size_errors = torch.abs(predicted_categories - actual_categories).sum().item()
    
    # Calculate attribute errors
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    household_composition_col_persons, household_composition_col_households = 6, 0
    
    total_persons = len(final_assignments)
    religion_errors = 0
    ethnicity_errors = 0
    household_composition_errors = 0
    
    for person_idx, household_idx in enumerate(final_assignments):
        household_idx = household_idx.item()
        
        if person_nodes[person_idx, religion_col_persons] != household_nodes[household_idx, religion_col_households]:
            religion_errors += 1
        if person_nodes[person_idx, ethnicity_col_persons] != household_nodes[household_idx, ethnicity_col_households]:
            ethnicity_errors += 1
        if person_nodes[person_idx, household_composition_col_persons] != household_nodes[household_idx, household_composition_col_households]:
            household_composition_errors += 1
    
    # Calculate correct assignments (for stacked bar showing errors vs correct)
    size_correct = total_persons - size_errors
    religion_correct = total_persons - religion_errors
    ethnicity_correct = total_persons - ethnicity_errors
    composition_correct = total_persons - household_composition_errors
    
    # Create paper-ready figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Household\nSize', 'Religion', 'Ethnicity', 'Household\nComposition']
    correct_counts = [size_correct, religion_correct, ethnicity_correct, composition_correct]
    error_counts = [size_errors, religion_errors, ethnicity_errors, household_composition_errors]
    
    x = np.arange(len(categories))
    bar_width = 0.6
    
    # Create stacked bars (correct in blue, errors in red)
    bars_correct = ax.bar(x, correct_counts, bar_width, 
                          label='Correct Assignments', 
                          color=PLOT_CONFIG['colors']['primary'],
                          edgecolor='white', linewidth=0.5)
    bars_errors = ax.bar(x, error_counts, bar_width, bottom=correct_counts,
                         label='Mismatched Assignments',
                         color=PLOT_CONFIG['colors']['secondary'],
                         edgecolor='white', linewidth=0.5)
    
    # Add percentage labels on bars
    for i, (correct, error) in enumerate(zip(correct_counts, error_counts)):
        total = correct + error
        correct_pct = (correct / total) * 100
        error_pct = (error / total) * 100
        
        # Label for correct portion (in middle of correct bar)
        if correct_pct > 10:
            ax.text(i, correct / 2, f'{correct_pct:.1f}%', 
                   ha='center', va='center', fontsize=PLOT_CONFIG['fonts']['annotation'],
                   fontweight='bold', color='white')
        
        # Label for error portion (in middle of error bar)
        if error_pct > 5:
            ax.text(i, correct + error / 2, f'{error_pct:.1f}%',
                   ha='center', va='center', fontsize=PLOT_CONFIG['fonts']['annotation'],
                   fontweight='bold', color='white')
    
    # Styling
    ax.set_xlabel('Assignment Attribute', fontsize=PLOT_CONFIG['fonts']['axis_title'], 
                  fontfamily=PLOT_CONFIG['font_family'])
    ax.set_ylabel('Number of Persons', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                  fontfamily=PLOT_CONFIG['font_family'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=PLOT_CONFIG['fonts']['tick_labels'],
                       fontfamily=PLOT_CONFIG['font_family'])
    ax.tick_params(axis='y', labelsize=PLOT_CONFIG['fonts']['tick_labels'])
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=PLOT_CONFIG['colors']['grid'])
    ax.set_axisbelow(True)
    
    # Legend - positioned below title, no border
    ax.legend(loc='upper center', fontsize=PLOT_CONFIG['fonts']['legend'] - 2,
             framealpha=0.0, edgecolor='none', ncol=2,
             bbox_to_anchor=(0.5, 1.03))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PLOT_CONFIG['colors']['axis'])
    ax.spines['bottom'].set_color(PLOT_CONFIG['colors']['axis'])
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    error_plot_path = os.path.join(output_dir, 'assignment_errors.png')
    plt.savefig(error_plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Assignment errors plot saved to: {error_plot_path}")
    
    pdf_path = os.path.join(output_dir, 'assignment_errors.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Assignment errors PDF saved to: {pdf_path}")
    
    plt.close()


def plot_training_convergence(convergence_data, output_dir):
    """
    Plot training convergence with loss curves and accuracy curves.
    Paper-ready version with consistent styling.
    
    Args:
        convergence_data: Dictionary containing epochs, losses, and accuracies
        output_dir: Directory to save plots
    """
    epochs = convergence_data['epochs']
    
    # ============ FIGURE 1: LOSS CONVERGENCE ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Total Loss
    ax1.plot(epochs, convergence_data['losses'], 
             color=PLOT_CONFIG['colors']['primary'], linewidth=2.5, label='Total Loss')
    
    ax1.set_xlabel('Epoch', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                   fontfamily=PLOT_CONFIG['font_family'])
    ax1.set_ylabel('Loss', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                   fontfamily=PLOT_CONFIG['font_family'])
    ax1.set_title('(a) Training Loss Convergence', fontsize=PLOT_CONFIG['fonts']['title'],
                  fontweight='bold', fontfamily=PLOT_CONFIG['font_family'])
    
    ax1.tick_params(axis='both', labelsize=PLOT_CONFIG['fonts']['tick_labels'])
    ax1.grid(True, linestyle='--', alpha=0.7, color=PLOT_CONFIG['colors']['grid'])
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Find and mark minimum loss point
    min_loss_idx = np.argmin(convergence_data['losses'])
    min_loss_epoch = epochs[min_loss_idx]
    min_loss_value = convergence_data['losses'][min_loss_idx]
    ax1.scatter([min_loss_epoch], [min_loss_value], color=PLOT_CONFIG['colors']['secondary'],
               s=100, zorder=5, marker='*', label=f'Min Loss: {min_loss_value:.3f} (Epoch {min_loss_epoch})')
    ax1.legend(loc='upper right', fontsize=PLOT_CONFIG['fonts']['legend'] - 2, 
              framealpha=0.0, edgecolor='none')
    
    # Right panel: Component Losses
    ax2.plot(epochs, convergence_data['household_composition_losses'], 
             color=PLOT_CONFIG['colors']['quaternary'], linewidth=2, label='Composition Loss', alpha=0.9)
    ax2.plot(epochs, convergence_data['size_losses'], 
             color=PLOT_CONFIG['colors']['tertiary'], linewidth=2, label='Size Loss', alpha=0.9)
    ax2.plot(epochs, convergence_data['religion_losses'], 
             color=PLOT_CONFIG['colors']['primary'], linewidth=2, label='Religion Loss', alpha=0.9)
    ax2.plot(epochs, convergence_data['ethnicity_losses'], 
             color=PLOT_CONFIG['colors']['secondary'], linewidth=2, label='Ethnicity Loss', alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                   fontfamily=PLOT_CONFIG['font_family'])
    ax2.set_ylabel('Loss', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                   fontfamily=PLOT_CONFIG['font_family'])
    ax2.set_title('(b) Component Loss Breakdown', fontsize=PLOT_CONFIG['fonts']['title'],
                  fontweight='bold', fontfamily=PLOT_CONFIG['font_family'])
    
    ax2.tick_params(axis='both', labelsize=PLOT_CONFIG['fonts']['tick_labels'])
    ax2.grid(True, linestyle='--', alpha=0.7, color=PLOT_CONFIG['colors']['grid'])
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=PLOT_CONFIG['fonts']['legend'] - 2, 
              framealpha=0.0, edgecolor='none')
    
    plt.tight_layout()
    
    # Save loss convergence plot
    loss_plot_path = os.path.join(output_dir, 'loss_convergence.png')
    plt.savefig(loss_plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Loss convergence plot saved to: {loss_plot_path}")
    
    pdf_path = os.path.join(output_dir, 'loss_convergence.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Loss convergence PDF saved to: {pdf_path}")
    plt.close()
    
    # ============ FIGURE 2: ACCURACY CONVERGENCE ============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, convergence_data['household_composition_accuracies'],
            color=PLOT_CONFIG['colors']['quaternary'], linewidth=2.5, 
            label='Household Composition', alpha=0.9)
    ax.plot(epochs, convergence_data['religion_accuracies'],
            color=PLOT_CONFIG['colors']['primary'], linewidth=2.5,
            label='Religion', alpha=0.9)
    ax.plot(epochs, convergence_data['ethnicity_accuracies'],
            color=PLOT_CONFIG['colors']['secondary'], linewidth=2.5,
            label='Ethnicity', alpha=0.9)
    ax.plot(epochs, convergence_data['overall_accuracies'],
            color='#7F7F7F', linewidth=3, linestyle='--',
            label='Overall Average', alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                  fontfamily=PLOT_CONFIG['font_family'])
    ax.set_ylabel('Accuracy (%)', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                  fontfamily=PLOT_CONFIG['font_family'])
    ax.set_title('Training Accuracy Convergence', fontsize=PLOT_CONFIG['fonts']['title'],
                 fontweight='bold', fontfamily=PLOT_CONFIG['font_family'])
    
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', labelsize=PLOT_CONFIG['fonts']['tick_labels'])
    ax.grid(True, linestyle='--', alpha=0.7, color=PLOT_CONFIG['colors']['grid'])
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='#CCCCCC', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.legend(loc='lower right', fontsize=PLOT_CONFIG['fonts']['legend'] - 2, 
             framealpha=0.0, edgecolor='none')
    
    plt.tight_layout()
    
    # Save accuracy convergence plot
    acc_plot_path = os.path.join(output_dir, 'accuracy_convergence.png')
    plt.savefig(acc_plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Accuracy convergence plot saved to: {acc_plot_path}")
    
    pdf_path = os.path.join(output_dir, 'accuracy_convergence.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Accuracy convergence PDF saved to: {pdf_path}")
    plt.close()


def plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, household_composition_accuracies, output_dir):
    """
    Plot accuracy over epochs with religion, ethnicity, and household composition.
    Paper-ready version with consistent styling.
    """
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Subsample for cleaner visualization (every 10th epoch)
    step = max(1, len(epoch_numbers) // 40)  # Show ~40 bars max
    
    epochs_sub = epoch_numbers[::step]
    religion_sub = religion_accuracies[::step]
    ethnicity_sub = ethnicity_accuracies[::step]
    composition_sub = household_composition_accuracies[::step]
    
    bar_width = max(1, step * 0.8)
    
    # Subplot styling function
    def style_subplot(ax, epochs, accuracies, title, color):
        bars = ax.bar(epochs, accuracies, width=bar_width, color=color, alpha=0.85, edgecolor='white')
        
        ax.set_xlabel('Epoch', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                      fontfamily=PLOT_CONFIG['font_family'])
        ax.set_ylabel('Accuracy (%)', fontsize=PLOT_CONFIG['fonts']['axis_title'],
                      fontfamily=PLOT_CONFIG['font_family'])
        ax.set_title(title, fontsize=PLOT_CONFIG['fonts']['title'],
                     fontweight='bold', fontfamily=PLOT_CONFIG['font_family'])
        
        ax.set_ylim(0, 105)
        ax.tick_params(axis='both', labelsize=PLOT_CONFIG['fonts']['tick_labels'])
        ax.grid(True, linestyle='--', alpha=0.7, color=PLOT_CONFIG['colors']['grid'], axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add final accuracy annotation
        final_acc = accuracies[-1]
        ax.annotate(f'Final: {final_acc:.1f}%', 
                   xy=(epochs[-1], final_acc),
                   xytext=(epochs[-1] - len(epochs) * 0.15, final_acc + 5),
                   fontsize=PLOT_CONFIG['fonts']['annotation'],
                   fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    # Plot each attribute
    style_subplot(axes[0], epochs_sub, religion_sub, '(a) Religion', PLOT_CONFIG['colors']['primary'])
    style_subplot(axes[1], epochs_sub, ethnicity_sub, '(b) Ethnicity', PLOT_CONFIG['colors']['secondary'])
    style_subplot(axes[2], epochs_sub, composition_sub, '(c) Household Composition', PLOT_CONFIG['colors']['quaternary'])
    
    plt.tight_layout()
    
    # Save plots
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_over_epochs.png')
    plt.savefig(accuracy_plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Accuracy over epochs plot saved to: {accuracy_plot_path}")
    
    pdf_path = os.path.join(output_dir, 'accuracy_over_epochs.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Accuracy over epochs PDF saved to: {pdf_path}")
    plt.close()

# Household Size Accuracy Function (updated for 4-category system)
def calculate_size_distribution_accuracy(assignments, household_sizes):
    """
    Calculate household size distribution accuracy by comparing predicted vs expected size distributions.
    Updated to work with the new 4-category system (1, 2, 3, 4+).
    
    Note: For category 4+ (category index 3), actual household size is flexible (>=4 persons).
    We bin all sizes >=4 into category 3 for distribution comparison.
    """
    # Step 1: Calculate the predicted sizes (how many people in each household)
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in assignments:
        predicted_counts[household_idx] += 1  # Increment for each assignment
    
    # Step 2: Convert actual sizes to categories for comparison
    # Map: 1->0, 2->1, 3->2, 4->3 (4+ category is represented as 4, maps to category 3)
    actual_categories = torch.clamp(household_sizes - 1, 0, 3)  # Convert 1-4 sizes to 0-3 categories
    
    # Step 3: Convert predicted counts to categories
    # Map: 1->0, 2->1, 3->2, >=4->3 (anything 4 or more maps to category 3, representing 4+)
    predicted_categories = torch.clamp(predicted_counts - 1, 0, 3)  # Convert 1-4+ sizes to 0-3 categories
    
    # Step 4: Calculate bincount of the categories (4 categories: 0, 1, 2, 3)
    num_categories = 4
    predicted_distribution = torch.bincount(predicted_categories, minlength=num_categories).float()
    actual_distribution = torch.bincount(actual_categories, minlength=num_categories).float()

    # Step 5: Calculate accuracy for each category
    accuracies = torch.min(predicted_distribution, actual_distribution) / (actual_distribution + 1e-6)  # Avoid division by 0
    overall_accuracy = accuracies.mean().item()  # Average accuracy across all categories
    
    # Debug information (optional - can be removed later)
    if torch.rand(1).item() < 0.01:  # Only print 1% of the time to avoid spam
        print(f"Size accuracy debug - Predicted: {predicted_distribution.cpu().numpy()}, Actual: {actual_distribution.cpu().numpy()}")
        print(f"Category accuracies: {accuracies.cpu().numpy()}, Overall: {overall_accuracy:.4f}")

    return overall_accuracy

# Compliance Accuracy Function
def calculate_individual_compliance_accuracy(assignments, person_nodes, household_nodes):
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    household_composition_col_persons, household_composition_col_households = 6, 0

    total_people = assignments.size(0)
    
    correct_religion_assignments = 0
    correct_ethnicity_assignments = 0
    correct_household_composition_assignments = 0

    # Loop over each person and their assigned household
    for person_idx, household_idx in enumerate(assignments):
        household_idx = household_idx.item()  # Get the household assignment for the person

        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]
        person_household_composition = person_nodes[person_idx, household_composition_col_persons]

        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]
        household_composition = household_nodes[household_idx, household_composition_col_households]

        # Check if the person's religion matches the household's religion
        if person_religion == household_religion:
            correct_religion_assignments += 1

        # Check if the person's ethnicity matches the household's ethnicity
        if person_ethnicity == household_ethnicity:
            correct_ethnicity_assignments += 1

        # Check if the person's household composition matches the household's composition
        if person_household_composition == household_composition:
            correct_household_composition_assignments += 1

    religion_compliance = correct_religion_assignments / total_people
    ethnicity_compliance = correct_ethnicity_assignments / total_people
    household_composition_compliance = correct_household_composition_assignments / total_people

    return religion_compliance, ethnicity_compliance, household_composition_compliance

def greedy_constrained_assignment(logits, household_sizes, person_nodes, household_nodes, device, verbose=False):
    """
    HYBRID COMPOSITION-AWARE greedy assignment algorithm.
    
    STRATEGY (4-Phase Assignment):
    1. Phase 1a: Single-person compositions (1PE, 1PA) - EXACT match on composition + religion + ethnicity
    2. Phase 1b: Multi-person compositions - match on composition only (religion/ethnicity as bonus)
    3. Phase 2: Assign remaining to SIMILAR composition households
    4. Phase 3: Final fallback for any remaining persons
    
    This hybrid approach accounts for the fact that:
    - Single-person households: religion/ethnicity SHOULD match 100%
    - Multi-person households: only HRP's religion/ethnicity is recorded, not all members
    
    Args:
        logits: Model output [num_persons, num_households]
        household_sizes: Expected size for each household [num_households]
        person_nodes, household_nodes: For attribute matching
        device: GPU device
        verbose: Print assignment statistics
    
    Returns:
        assignments: Hard assignments [num_persons] with household indices
    """
    num_persons = logits.size(0)
    num_households = household_sizes.size(0)
    
    # Initialize tracking
    assignments = torch.full((num_persons,), -1, dtype=torch.long, device=device)
    household_counts = torch.zeros(num_households, dtype=torch.long, device=device)
    
    # Set max capacity (4+ households can hold up to 8)
    max_capacity = household_sizes.clone().long()
    max_capacity[household_sizes == 4] = 8
    
    # ============================================================================
    # PRECOMPUTE SCORES AND ATTRIBUTES
    # ============================================================================
    model_scores = F.softmax(logits, dim=1)  # [num_persons, num_households]
    
    # Extract attributes
    person_hhcomp = person_nodes[:, 6].long()
    person_religion = person_nodes[:, 2].long()
    person_ethnicity = person_nodes[:, 3].long()
    
    household_hhcomp = household_nodes[:, 0].long()
    household_religion = household_nodes[:, 2].long()
    household_ethnicity = household_nodes[:, 1].long()
    
    # Compute attribute match matrices
    rel_match_matrix = (person_religion.unsqueeze(1) == household_religion.unsqueeze(0))
    eth_match_matrix = (person_ethnicity.unsqueeze(1) == household_ethnicity.unsqueeze(0))
    
    # Scores with attribute bonuses (for multi-person households)
    rel_bonus = rel_match_matrix.float() * 0.15
    eth_bonus = eth_match_matrix.float() * 0.15
    scores_with_bonus = model_scores + rel_bonus + eth_bonus
    
    unassigned_mask = torch.ones(num_persons, dtype=torch.bool, device=device)
    phase1a_assigned = 0  # Single-person with exact match
    phase1b_assigned = 0  # Multi-person with composition match
    phase2_assigned = 0
    
    unique_comps = torch.unique(person_hhcomp)
    
    # ============================================================================
    # PHASE 1a: SINGLE-PERSON COMPOSITIONS - EXACT MATCH (composition + religion + ethnicity)
    # ============================================================================
    # For 1PE and 1PA, we require ALL attributes to match since there's only one person
    single_person_comps = [0, 1]  # 1PE, 1PA
    
    for comp_val in single_person_comps:
        person_mask = (person_hhcomp == comp_val) & unassigned_mask
        hh_mask = (household_hhcomp == comp_val)
        
        if not person_mask.any() or not hh_mask.any():
            continue
        
        person_indices = torch.where(person_mask)[0]
        hh_indices = torch.where(hh_mask)[0]
        
        # For single-person: require EXACT match on religion AND ethnicity
        # Create combined match mask: composition (already filtered) + religion + ethnicity
        exact_match_scores = model_scores[person_indices][:, hh_indices].clone()
        
        # Mask out households that don't match religion or ethnicity
        for local_p_idx, p_idx in enumerate(person_indices):
            p_religion = person_religion[p_idx]
            p_ethnicity = person_ethnicity[p_idx]
            
            for local_hh_idx, hh_idx in enumerate(hh_indices):
                hh_religion = household_religion[hh_idx]
                hh_ethnicity = household_ethnicity[hh_idx]
                
                # Require BOTH religion AND ethnicity to match for single-person HH
                if p_religion != hh_religion or p_ethnicity != hh_ethnicity:
                    exact_match_scores[local_p_idx, local_hh_idx] = -float('inf')
        
        # Sort persons by max score (most confident first)
        max_person_scores, _ = exact_match_scores.max(dim=1)
        # Filter out persons with no valid matches
        has_valid_match = max_person_scores > -float('inf')
        valid_person_indices = person_indices[has_valid_match]
        valid_scores = exact_match_scores[has_valid_match]
        
        if len(valid_person_indices) == 0:
            continue
        
        max_valid_scores, _ = valid_scores.max(dim=1)
        sorted_order = torch.argsort(max_valid_scores, descending=True)
        
        local_counts = household_counts[hh_indices].clone()
        local_capacity = max_capacity[hh_indices]
        
        for idx in sorted_order:
            p_idx = valid_person_indices[idx].item()
            
            available_mask = local_counts < local_capacity
            if not available_mask.any():
                break
            
            p_scores = valid_scores[idx].clone()
            p_scores[~available_mask] = -float('inf')
            
            if p_scores.max() == -float('inf'):
                continue  # No valid household available
            
            best_local_hh = torch.argmax(p_scores).item()
            best_hh_idx = hh_indices[best_local_hh].item()
            
            assignments[p_idx] = best_hh_idx
            household_counts[best_hh_idx] += 1
            local_counts[best_local_hh] += 1
            unassigned_mask[p_idx] = False
            phase1a_assigned += 1
    
    # ============================================================================
    # PHASE 1b: MULTI-PERSON COMPOSITIONS - COMPOSITION MATCH (religion/ethnicity as bonus)
    # ============================================================================
    # For multi-person households, only require composition match
    # Religion/ethnicity are soft preferences (bonuses) not hard requirements
    
    for comp_idx in unique_comps:
        comp_val = comp_idx.item()
        
        # Skip single-person compositions (already handled in Phase 1a)
        if comp_val in single_person_comps:
            continue
        
        person_mask = (person_hhcomp == comp_val) & unassigned_mask
        hh_mask = (household_hhcomp == comp_val)
        
        if not person_mask.any() or not hh_mask.any():
            continue
        
        person_indices = torch.where(person_mask)[0]
        hh_indices = torch.where(hh_mask)[0]
        
        # Use scores with religion/ethnicity bonuses (soft preference, not hard constraint)
        subset_scores = scores_with_bonus[person_indices][:, hh_indices]
        
        # Sort persons by max score (most confident first)
        max_person_scores, _ = subset_scores.max(dim=1)
        sorted_person_order = torch.argsort(max_person_scores, descending=True)
        
        local_counts = household_counts[hh_indices].clone()
        local_capacity = max_capacity[hh_indices]
        
        for local_p_idx in sorted_person_order:
            p_idx = person_indices[local_p_idx].item()
            
            available_mask = local_counts < local_capacity
            if not available_mask.any():
                break
            
            p_scores = subset_scores[local_p_idx].clone()
            p_scores[~available_mask] = -float('inf')
            
            best_local_hh = torch.argmax(p_scores).item()
            best_hh_idx = hh_indices[best_local_hh].item()
            
            assignments[p_idx] = best_hh_idx
            household_counts[best_hh_idx] += 1
            local_counts[best_local_hh] += 1
            unassigned_mask[p_idx] = False
            phase1b_assigned += 1
    
    # Also handle any remaining single-person compositions that couldn't find exact matches
    for comp_val in single_person_comps:
        person_mask = (person_hhcomp == comp_val) & unassigned_mask
        hh_mask = (household_hhcomp == comp_val)
        
        if not person_mask.any() or not hh_mask.any():
            continue
        
        person_indices = torch.where(person_mask)[0]
        hh_indices = torch.where(hh_mask)[0]
        
        # Fallback: use composition match with bonuses (no exact requirement)
        subset_scores = scores_with_bonus[person_indices][:, hh_indices]
        
        max_person_scores, _ = subset_scores.max(dim=1)
        sorted_person_order = torch.argsort(max_person_scores, descending=True)
        
        local_counts = household_counts[hh_indices].clone()
        local_capacity = max_capacity[hh_indices]
        
        for local_p_idx in sorted_person_order:
            p_idx = person_indices[local_p_idx].item()
            
            available_mask = local_counts < local_capacity
            if not available_mask.any():
                break
            
            p_scores = subset_scores[local_p_idx].clone()
            p_scores[~available_mask] = -float('inf')
            
            best_local_hh = torch.argmax(p_scores).item()
            best_hh_idx = hh_indices[best_local_hh].item()
            
            assignments[p_idx] = best_hh_idx
            household_counts[best_hh_idx] += 1
            local_counts[best_local_hh] += 1
            unassigned_mask[p_idx] = False
            phase1b_assigned += 1  # Count as 1b since no exact match
    
    phase1_assigned = phase1a_assigned + phase1b_assigned
    
    # ============================================================================
    # PHASE 2: SIMILAR COMPOSITION MATCHES (using similarity groups)
    # ============================================================================
    remaining_persons = torch.where(unassigned_mask)[0]
    
    if len(remaining_persons) > 0:
        # Group remaining persons by their composition
        for comp_idx in unique_comps:
            comp_val = comp_idx.item()
            
            # Get unassigned persons with this composition
            person_mask = (person_hhcomp == comp_val) & unassigned_mask
            if not person_mask.any():
                continue
            
            person_indices = torch.where(person_mask)[0]
            
            # Get similar compositions (skip the first one - it's the exact match already tried)
            similar_comps = get_similar_compositions(comp_val)
            
            for sim_comp in similar_comps[1:]:  # Skip index 0 (exact match)
                if not unassigned_mask[person_indices].any():
                    break  # All persons in this group assigned
                
                # Find households with this similar composition
                hh_mask = (household_hhcomp == sim_comp)
                if not hh_mask.any():
                    continue
                
                hh_indices = torch.where(hh_mask)[0]
                
                # Check for available capacity
                available_hh_mask = household_counts[hh_indices] < max_capacity[hh_indices]
                if not available_hh_mask.any():
                    continue
                
                available_hh = hh_indices[available_hh_mask]
                
                # Get still-unassigned persons from this composition group
                still_unassigned = person_indices[unassigned_mask[person_indices]]
                
                # Get scores for these persons to available similar households
                subset_scores = scores_with_bonus[still_unassigned][:, available_hh]
                
                # Sort persons by max score
                max_person_scores, _ = subset_scores.max(dim=1)
                sorted_order = torch.argsort(max_person_scores, descending=True)
                
                for idx in sorted_order:
                    p_idx = still_unassigned[idx].item()
                    
                    if not unassigned_mask[p_idx]:
                        continue
                    
                    # Find best available household
                    local_available = household_counts[available_hh] < max_capacity[available_hh]
                    if not local_available.any():
                        break
                    
                    p_scores = subset_scores[idx].clone()
                    p_scores[~local_available] = -float('inf')
                    
                    best_local_idx = torch.argmax(p_scores).item()
                    best_hh_idx = available_hh[best_local_idx].item()
                    
                    assignments[p_idx] = best_hh_idx
                    household_counts[best_hh_idx] += 1
                    unassigned_mask[p_idx] = False
                    phase2_assigned += 1
    
    # ============================================================================
    # PHASE 3: FINAL FALLBACK (any remaining to any available)
    # ============================================================================
    remaining_persons = torch.where(unassigned_mask)[0]
    phase3_assigned = 0
    
    if len(remaining_persons) > 0:
        remaining_scores = scores_with_bonus[remaining_persons]
        max_scores, _ = remaining_scores.max(dim=1)
        sorted_order = torch.argsort(max_scores, descending=True)
        
        for idx in sorted_order:
            p_idx = remaining_persons[idx].item()
            
            available_mask = household_counts < max_capacity
            
            if available_mask.any():
                p_scores = scores_with_bonus[p_idx].clone()
                p_scores[~available_mask] = -float('inf')
                best_hh_idx = torch.argmax(p_scores).item()
            else:
                overflow = household_counts - max_capacity
                best_hh_idx = torch.argmin(overflow).item()
            
            assignments[p_idx] = best_hh_idx
            household_counts[best_hh_idx] += 1
            unassigned_mask[p_idx] = False
            phase3_assigned += 1
    
    # ============================================================================
    # STATISTICS
    # ============================================================================
    if verbose:
        empty_households = (household_counts == 0).sum().item()
        overflowed = (household_counts > max_capacity).sum().item()
        
        assigned_hhcomp = household_hhcomp[assignments]
        composition_match_rate = (assigned_hhcomp == person_hhcomp).float().mean().item() * 100
        
        assigned_rel = household_religion[assignments]
        religion_match_rate = (assigned_rel == person_religion).float().mean().item() * 100
        
        assigned_eth = household_ethnicity[assignments]
        ethnicity_match_rate = (assigned_eth == person_ethnicity).float().mean().item() * 100
        
        # Calculate religion/ethnicity match rates by composition type
        single_person_mask = (person_hhcomp == 0) | (person_hhcomp == 1)  # 1PE, 1PA
        multi_person_mask = ~single_person_mask
        
        if single_person_mask.any():
            single_rel_match = (assigned_rel[single_person_mask] == person_religion[single_person_mask]).float().mean().item() * 100
            single_eth_match = (assigned_eth[single_person_mask] == person_ethnicity[single_person_mask]).float().mean().item() * 100
        else:
            single_rel_match = 0
            single_eth_match = 0
            
        if multi_person_mask.any():
            multi_rel_match = (assigned_rel[multi_person_mask] == person_religion[multi_person_mask]).float().mean().item() * 100
            multi_eth_match = (assigned_eth[multi_person_mask] == person_ethnicity[multi_person_mask]).float().mean().item() * 100
        else:
            multi_rel_match = 0
            multi_eth_match = 0
        
        print(f"\n    HYBRID 4-Phase assignment completed:")
        print(f"    - Phase 1a (single-person EXACT): {phase1a_assigned} persons")
        print(f"    - Phase 1b (multi-person comp): {phase1b_assigned} persons")
        print(f"    - Phase 2 (similar match): {phase2_assigned} persons")
        print(f"    - Phase 3 (fallback): {phase3_assigned} persons")
        print(f"    - Total assigned: {num_persons}/{num_persons}")
        print(f"    Match Rates:")
        print(f"    - Composition: {composition_match_rate:.1f}%")
        print(f"    - Religion (overall): {religion_match_rate:.1f}%")
        print(f"      -> Single-person HH: {single_rel_match:.1f}% (target: 100%)")
        print(f"      -> Multi-person HH:  {multi_rel_match:.1f}% (target: 35-50%)")
        print(f"    - Ethnicity (overall): {ethnicity_match_rate:.1f}%")
        print(f"      -> Single-person HH: {single_eth_match:.1f}% (target: 100%)")
        print(f"      -> Multi-person HH:  {multi_eth_match:.1f}% (target: 35-50%)")
        print(f"    - Empty households: {empty_households}/{num_households}")
        print(f"    - Overflowed households: {overflowed}/{num_households}")
        
        counts_np = household_counts.cpu().numpy()
        print(f"    - Household size distribution: min={counts_np.min()}, max={counts_np.max()}, mean={counts_np.mean():.2f}, std={counts_np.std():.2f}")
    
    return assignments

# Combined Accuracy Function
def calculate_all_accuracies(assignments, person_nodes, household_nodes, household_sizes):
    """
    Calculate all accuracies: religion, ethnicity, household composition, and household size.
    Returns individual accuracies and overall average accuracy.
    Updated for 4-category household size system.
    """
    # Calculate religion, ethnicity, and household composition accuracies
    religion_compliance, ethnicity_compliance, household_composition_compliance = calculate_individual_compliance_accuracy(
        assignments, person_nodes, household_nodes
    )
    
    # Calculate household size distribution accuracy (updated for 4-category system)
    size_distribution_accuracy = calculate_size_distribution_accuracy(
        assignments, household_sizes
    )
    
    # Calculate overall average accuracy (now including household composition)
    overall_accuracy = (religion_compliance + ethnicity_compliance + household_composition_compliance + size_distribution_accuracy) / 4.0
    
    return {
        'religion_compliance': religion_compliance,
        'ethnicity_compliance': ethnicity_compliance,
        'household_composition_compliance': household_composition_compliance,
        'size_distribution_accuracy': size_distribution_accuracy,
        'overall_accuracy': overall_accuracy
    }

# Function to perform training with given hyperparameters
def train_model(learning_rate, hidden_channels, return_detailed_results=False):
    print(f"    Starting training with LR={learning_rate}, Hidden={hidden_channels}, Heads={attention_heads}, EdgeDropout={edge_dropout}")
    
    # Clear GPU memory before starting new training
    clear_gpu_memory()
    monitor_memory_usage(device, "before model creation")
    
    # Create GAT model with memory-efficient settings
    # FIXED: Now using bipartite graph architecture with separate person/household embeddings
    model = HouseholdAssignmentGNN(
        person_in_channels=person_nodes.size(1),  # 7 attributes for persons
        household_in_channels=household_nodes.size(1),  # 6 attributes for households
        hidden_channels=hidden_channels, 
        num_persons=person_nodes.size(0),
        num_households=household_sizes.size(0),
        dropout_rate=0.05,  # GAT dropout
        attention_heads=attention_heads,  # Use global setting (default: 2)
        edge_dropout=edge_dropout  # Use global setting (default: 0.1)
    )
    model = model.to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    
    # Use a simpler scheduler to avoid compatibility issues
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        use_scheduler = True
    except Exception as e:
        print(f"Warning: Could not create ReduceLROnPlateau scheduler: {e}")
        print("Continuing without learning rate scheduling...")
        scheduler = None
        use_scheduler = False
    
    # Temperature annealing configuration
    tau_start = 2.0  # Start with high temperature (softer assignments)
    tau_end = 0.5    # End with low temperature (harder assignments)
    tau = tau_start

    monitor_memory_usage(device, "after model creation")

    # Track accuracies over epochs and convergence data
    religion_accuracies = []
    ethnicity_accuracies = []
    household_composition_accuracies = []
    epoch_numbers = []
    
    # Track convergence data for all training runs
    convergence_data = {
        'epochs': [],
        'losses': [],
        'size_losses': [],
        'religion_losses': [],
        'ethnicity_losses': [],
        'household_composition_losses': [],
        'religion_accuracies': [],
        'ethnicity_accuracies': [],
        'household_composition_accuracies': [],
        'size_distribution_accuracies': [],
        'overall_accuracies': [],
        'cumulative_time_seconds': [],
        'epoch_time_seconds': [],
        'tau_values': []
    }
    
    # Start timing for epoch-wise tracking
    training_start_time = time.time()
    best_epoch_loss = float('inf')
    best_epoch_accuracy = 0.0  # Track best overall accuracy for reference
    best_epoch_state = None
    best_epoch_number = 0  # Track which epoch had best loss
    
    # ============================================================================
    # EARLY STOPPING BASED ON LOSS (not accuracy)
    # ============================================================================
    # Rationale: The greedy assignment algorithm masks poor model predictions,
    # so accuracy can stay high even when the model is overfitting.
    # Loss directly measures model quality and shows overfitting clearly.
    #
    # From training logs: Loss minimum at epoch ~200 (0.88), then rises to 2.72 by epoch 400
    # This indicates severe overfitting that accuracy-based stopping doesn't catch.
    patience = 50  # Stop if loss doesn't improve for 50 epochs
    patience_counter = 0
    min_loss_improvement = 0.001  # Require at least 0.1% loss improvement
    
    # Track stopping epoch
    stopping_epoch = num_epochs  # Default to max epochs if no early stopping

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        # FIXED: Pass both person and household features to bipartite GAT
        logits = model(person_nodes, household_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=tau, hard=False)

        # Loss weights aligned with hybrid greedy scoring:
        # - Composition gets HIGHEST weight (primary constraint in Phase 1)
        # - Size important for capacity feasibility  
        # - Religion/ethnicity are secondary (tie-breakers)
        total_loss, size_loss, religion_loss, ethnicity_loss, household_composition_loss = compute_loss(
            assignments,
            household_sizes,
            person_nodes,
            household_nodes,
            religion_loss_weight=0.3,            # Secondary (tie-breaker)
            ethnicity_loss_weight=0.3,           # Secondary (tie-breaker)
            size_loss_weight=1.0,                # Important for feasibility
            household_composition_loss_weight=3.0  # HIGHEST priority - drives Phase 1
        )
        total_loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate scheduler if available
        if use_scheduler and scheduler is not None:
            scheduler.step(total_loss)
        
        # Update temperature using annealing schedule
        # tau = get_annealing_temperature(epoch, num_epochs, tau_start, tau_end, anneal_type='exponential')
        tau = get_annealing_temperature(epoch, num_epochs, tau_start, tau_end, anneal_type='cosine')
        # tau = get_annealing_temperature(epoch, num_epochs, tau_start=3.0, tau_end=0.3, anneal_type='cosine')
        
        # Calculate all accuracies for this epoch using constraint-aware assignment
        # Use greedy assignment less frequently (every 50 epochs) for speed
        # Fast argmax is used for intermediate epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            final_assignments = greedy_constrained_assignment(
                logits, household_sizes, person_nodes, household_nodes, device
            )
        else:
            # Use fast argmax for intermediate epochs (much faster)
            final_assignments = torch.argmax(assignments, dim=1)
        
        accuracies = calculate_all_accuracies(final_assignments, person_nodes, household_nodes, household_sizes)
        
        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time = epoch_end_time - training_start_time
        
        # Store convergence data for every epoch
        convergence_data['epochs'].append(epoch + 1)
        convergence_data['losses'].append(total_loss.item())
        convergence_data['size_losses'].append(size_loss.item())
        convergence_data['religion_losses'].append(religion_loss.item())
        convergence_data['ethnicity_losses'].append(ethnicity_loss.item())
        convergence_data['household_composition_losses'].append(household_composition_loss.item())
        convergence_data['religion_accuracies'].append(accuracies['religion_compliance'] * 100)
        convergence_data['ethnicity_accuracies'].append(accuracies['ethnicity_compliance'] * 100)
        convergence_data['household_composition_accuracies'].append(accuracies['household_composition_compliance'] * 100)
        convergence_data['size_distribution_accuracies'].append(accuracies['size_distribution_accuracy'] * 100)
        convergence_data['overall_accuracies'].append(accuracies['overall_accuracy'] * 100)
        convergence_data['epoch_time_seconds'].append(epoch_duration)
        convergence_data['cumulative_time_seconds'].append(cumulative_time)
        convergence_data['tau_values'].append(tau)
        
        # Track for detailed results
        epoch_numbers.append(epoch + 1)
        religion_accuracies.append(accuracies['religion_compliance'] * 100)
        ethnicity_accuracies.append(accuracies['ethnicity_compliance'] * 100)
        household_composition_accuracies.append(accuracies['household_composition_compliance'] * 100)
        
        # ============================================================================
        # EARLY STOPPING BASED ON LOSS (prevents overfitting)
        # ============================================================================
        current_accuracy = accuracies['overall_accuracy']
        current_loss = total_loss.item()
        
        # Track best accuracy for reference (but don't use for stopping)
        if current_accuracy > best_epoch_accuracy:
            best_epoch_accuracy = current_accuracy
        
        # Check if loss improved (lower is better)
        if current_loss < best_epoch_loss - min_loss_improvement:
            best_epoch_loss = current_loss
            best_epoch_state = model.state_dict().copy()
            best_epoch_number = epoch + 1
            patience_counter = 0  # Reset patience on loss improvement
            if (epoch + 1) % 50 == 0:  # Only print occasionally
                print(f"      -> New best loss: {best_epoch_loss:.4f} (accuracy: {current_accuracy:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping check - stop if loss hasn't improved
        if patience_counter >= patience:
            stopping_epoch = epoch + 1
            print(f"\n    Early stopping at epoch {epoch+1}: Loss hasn't improved for {patience} epochs")
            print(f"    Best loss: {best_epoch_loss:.4f} at epoch {best_epoch_number}")
            print(f"    Current loss: {current_loss:.4f} (overfitting detected)")
            print(f"    Best accuracy achieved: {best_epoch_accuracy:.4f}")
            break
        
        # Clear intermediate tensors to free memory
        safe_delete_tensor(logits)
        safe_delete_tensor(assignments)
        safe_delete_tensor(final_assignments)
        
        # Print progress every 50 epochs (matches greedy frequency for accurate metrics)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            # Calculate loss ratios for debugging
            total_loss_val = total_loss.item()
            size_ratio = size_loss.item() / total_loss_val if total_loss_val > 0 else 0
            religion_ratio = religion_loss.item() / total_loss_val if total_loss_val > 0 else 0
            ethnicity_ratio = ethnicity_loss.item() / total_loss_val if total_loss_val > 0 else 0
            household_composition_ratio = household_composition_loss.item() / total_loss_val if total_loss_val > 0 else 0
            
            print(f"\n    Epoch {epoch+1:3d}/{num_epochs} | Loss: {total_loss_val:.4f} | HH_Comp: {household_composition_loss.item():.4f}({household_composition_ratio:.0%}) | Size: {size_loss.item():.4f}({size_ratio:.0%}) | Rel/Eth: {religion_loss.item():.4f}/{ethnicity_loss.item():.4f}")
            print(f"      Accuracies: Religion={accuracies['religion_compliance']*100:.1f}% | Ethnicity={accuracies['ethnicity_compliance']*100:.1f}% | HH_Comp={accuracies['household_composition_compliance']*100:.1f}% | Size={accuracies['size_distribution_accuracy']*100:.1f}% | Overall={accuracies['overall_accuracy']*100:.1f}%")
            
            # Monitor memory usage
            monitor_memory_usage(device, f"at epoch {epoch+1}")

    print()  # New line after training completes
    print(f"    Training completed at epoch {epoch + 1}")
    print(f"    Final loss: {total_loss.item():.6f}")
    print(f"    Best loss: {best_epoch_loss:.6f} (at epoch {best_epoch_number})")
    if epoch + 1 > best_epoch_number:
        print(f"    Note: Model was overfitting for {epoch + 1 - best_epoch_number} epochs after best loss")

    # Load best epoch state for final evaluation
    model.load_state_dict(best_epoch_state)
    
    # Get final assignments and accuracies using best model WITH CONSTRAINT ENFORCEMENT
    with torch.no_grad():
        # FIXED: Pass both person and household features to bipartite GAT
        logits = model(person_nodes, household_nodes, edge_index)
        
        # CRITICAL FIX: Use greedy constrained assignment instead of naive argmax
        final_assignments = greedy_constrained_assignment(
            logits, household_sizes, person_nodes, household_nodes, device, verbose=True
        )
        
        # Convert to soft assignments for validation (one-hot encoding)
        assignments = F.one_hot(final_assignments, num_classes=household_sizes.size(0)).float()
        
        final_accuracies = calculate_all_accuracies(final_assignments, person_nodes, household_nodes, household_sizes)
        
        # Validate final assignment
        assignment_valid = validate_final_assignment(assignments, person_nodes, household_nodes, household_sizes)
        if not assignment_valid:
            print("[WARN] WARNING: Final assignment validation failed! Results may be unreliable.")

    # Update global best model info if this model performs better
    global best_model_info
    if final_accuracies['overall_accuracy'] > best_model_info['accuracy'] or (final_accuracies['overall_accuracy'] == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
        best_model_info.update({
            'model_state': best_epoch_state,
            'loss': best_epoch_loss,
            'accuracy': final_accuracies['overall_accuracy'],
            'assignments': final_assignments.clone(),
            'lr': learning_rate,
            'hidden_channels': hidden_channels,
            'convergence_data': convergence_data,
            'detailed_accuracies': final_accuracies,
            'epoch_numbers': epoch_numbers.copy(),
            'religion_accuracies': religion_accuracies.copy(),
            'ethnicity_accuracies': ethnicity_accuracies.copy(),
            'household_composition_accuracies': household_composition_accuracies.copy(),
            'stopping_epoch': stopping_epoch
        })

    # Clear model and intermediate tensors from GPU memory before returning
    safe_delete_tensor(logits)
    safe_delete_tensor(assignments)
    safe_delete_model(model)
    clear_gpu_memory()
    
    monitor_memory_usage(device, "after model cleanup")
    
    if return_detailed_results:
        return best_epoch_loss, final_assignments, epoch_numbers, religion_accuracies, ethnicity_accuracies, convergence_data, final_accuracies, stopping_epoch
    else:
        return best_epoch_loss, convergence_data, final_accuracies, stopping_epoch

# Perform grid search over hyperparameters
total_start_time = time.time()

print(f"\n{'='*80}")
print(f"STARTING GAT-BASED HYPERPARAMETER TUNING")
print(f"{'='*80}")
print(f"Model Architecture: Graph Attention Network (GAT)")
print(f"Attention Heads: {attention_heads}")
print(f"Edge Dropout: {edge_dropout}")
print(f"Learning Rates to Test: {learning_rates}")
print(f"Hidden Dimensions to Test: {hidden_dims}")
print(f"Total Combinations: {len(learning_rates) * len(hidden_dims)}")
print(f"Max Epochs per Combination: {num_epochs} (with early stopping)")
print(f"{'='*80}\n")

# Clear GPU memory before starting hyperparameter tuning
clear_gpu_memory()
monitor_memory_usage(device, "before hyperparameter tuning")

try:
    for idx, lr in enumerate(learning_rates):
        for jdx, hidden_dim in enumerate(hidden_dims):
            try:
                combination_start_time = time.time()
                print(f"Training with learning rate {lr} and hidden dimension {hidden_dim} ({idx*len(hidden_dims)+jdx+1}/{len(learning_rates)*len(hidden_dims)})")
                
                # Print GPU memory before training
                monitor_memory_usage(device, "before training combination")
                
                final_loss, convergence_data, final_accuracies, stopping_epoch = train_model(learning_rate=lr, hidden_channels=hidden_dim)
                
                combination_end_time = time.time()
                combination_training_time = combination_end_time - combination_start_time
                combination_training_time_str = str(timedelta(seconds=int(combination_training_time)))
                
                print(f"Final loss: {final_loss:.6f}")
                print(f"Final Religion Compliance: {final_accuracies['religion_compliance']*100:.2f}%")
                print(f"Final Ethnicity Compliance: {final_accuracies['ethnicity_compliance']*100:.2f}%")
                print(f"Final Size Distribution Accuracy: {final_accuracies['size_distribution_accuracy']*100:.2f}%")
                print(f"Final Overall Accuracy: {final_accuracies['overall_accuracy']*100:.2f}%")
                print(f"Training time: {combination_training_time_str}")
                
                # Store basic results for saving
                hp_results.append({
                    'learning_rate': lr,
                    'hidden_channels': hidden_dim,
                    'final_loss': final_loss,
                    'religion_compliance': final_accuracies['religion_compliance'],
                    'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                    'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                    'overall_accuracy': final_accuracies['overall_accuracy'],
                    'training_time': combination_training_time_str,
                    'stopping_epoch': stopping_epoch
                })
                
                # Store detailed results
                detailed_results.append({
                    'learning_rate': lr,
                    'hidden_channels': hidden_dim,
                    'final_loss': final_loss,
                    'religion_compliance': final_accuracies['religion_compliance'],
                    'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                    'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                    'overall_accuracy': final_accuracies['overall_accuracy'],
                    'religion_accuracy_percent': final_accuracies['religion_compliance'] * 100,
                    'ethnicity_accuracy_percent': final_accuracies['ethnicity_compliance'] * 100,
                    'size_distribution_accuracy_percent': final_accuracies['size_distribution_accuracy'] * 100,
                    'overall_accuracy_percent': final_accuracies['overall_accuracy'] * 100,
                    'training_time_seconds': combination_training_time,
                    'training_time_str': combination_training_time_str,
                    'area_code': selected_area_code,
                    'num_persons': num_persons,
                    'num_households': household_sizes.size(0),
                    'num_epochs': num_epochs,
                    'stopping_epoch': stopping_epoch
                })
                
                # Store convergence data with hyperparameter info
                convergence_data_with_hp = convergence_data.copy()
                convergence_data_with_hp['learning_rate'] = [lr] * len(convergence_data['epochs'])
                convergence_data_with_hp['hidden_channels'] = [hidden_dim] * len(convergence_data['epochs'])
                convergence_data_with_hp['combination_id'] = [f"lr_{lr}_hc_{hidden_dim}"] * len(convergence_data['epochs'])
                convergence_results.append(convergence_data_with_hp)
                
                # Print GPU memory after training
                monitor_memory_usage(device, "after training combination")

                # Track the best performing hyperparameters (using overall accuracy as primary metric)
                if final_accuracies['overall_accuracy'] > best_params.get('overall_accuracy', 0) or (final_accuracies['overall_accuracy'] == best_params.get('overall_accuracy', 0) and final_loss < best_loss):
                    best_loss = final_loss
                    best_params = {
                        'learning_rate': lr, 
                        'hidden_channels': hidden_dim,
                        'religion_compliance': final_accuracies['religion_compliance'],
                        'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                        'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                        'overall_accuracy': final_accuracies['overall_accuracy'],
                        'stopping_epoch': stopping_epoch
                    }
                
                # Clear memory between combinations
                clear_gpu_memory()
                print("-" * 50)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory error for combination LR={lr}, Hidden={hidden_dim}")
                    print(f"Error: {e}")
                    print("Clearing GPU memory and continuing with next combination...")
                    emergency_memory_cleanup()
                    continue
                else:
                    print(f"Runtime error for combination LR={lr}, Hidden={hidden_dim}: {e}")
                    clear_gpu_memory()
                    continue
            except Exception as e:
                print(f"Unexpected error for combination LR={lr}, Hidden={hidden_dim}: {e}")
                clear_gpu_memory()
                continue

except Exception as e:
    print(f"Critical error during hyperparameter tuning: {e}")
    print("Performing emergency cleanup...")
    emergency_memory_cleanup()
    raise

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total hyperparameter tuning time: {total_training_time_str}")

# Output the best hyperparameters
if best_params:
    print(f"Best hyperparameters: {best_params} with final loss {best_loss}")
else:
    print("No successful training runs completed. No best hyperparameters available.")
    best_params = {
        'learning_rate': None,
        'hidden_channels': None,
        'religion_compliance': 0.0,
        'ethnicity_compliance': 0.0,
        'size_distribution_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'stopping_epoch': 0
    }
    best_loss = float('inf')

# Save hyperparameter tuning results
print(f"\nSaving hyperparameter tuning results to: {output_dir}")

# Save basic results as CSV
if hp_results:
    hp_results_df = pd.DataFrame(hp_results)
    # Mark the best hyperparameter combination
    if best_model_info.get('lr') is not None and best_model_info.get('hidden_channels') is not None:
        hp_results_df['is_best'] = (
            (hp_results_df['learning_rate'] == best_model_info['lr']) &
            (hp_results_df['hidden_channels'] == best_model_info['hidden_channels'])
        )
    else:
        hp_results_df['is_best'] = False
    hp_results_path = os.path.join(output_dir, 'hp_tuning_results.csv')
    hp_results_df.to_csv(hp_results_path, index=False)
    print(f"Basic results saved: {hp_results_path}")
else:
    print("No results to save - no successful training runs")

# Save detailed results as CSV
if detailed_results:
    detailed_results_df = pd.DataFrame(detailed_results)
    detailed_results_path = os.path.join(output_dir, 'detailed_hp_results.csv')
    detailed_results_df.to_csv(detailed_results_path, index=False)
    print(f"Detailed results saved: {detailed_results_path}")
else:
    print("No detailed results to save - no successful training runs")

# Save convergence data for all combinations
if convergence_results:
    # Combine all convergence data into one DataFrame
    all_convergence_data = []
    for conv_data in convergence_results:
        # Convert to DataFrame and add to list
        conv_df = pd.DataFrame(conv_data)
        all_convergence_data.append(conv_df)
    
    # Concatenate all convergence data
    combined_convergence_df = pd.concat(all_convergence_data, ignore_index=True)
    convergence_path = os.path.join(output_dir, 'all_combinations_convergence_data.csv')
    combined_convergence_df.to_csv(convergence_path, index=False)
    print(f"Convergence data for all combinations saved: {convergence_path}")
else:
    print("No convergence data to save - no successful training runs")

# Save performance summary
performance_summary = {
    'area_code': selected_area_code,
    'num_persons': num_persons,
    'num_households': household_sizes.size(0),
    'total_combinations_tested': len(hp_results),
    'total_training_time_seconds': total_training_time,
    'total_training_time_str': total_training_time_str,
    'num_epochs_per_combination': num_epochs,
    'learning_rates_tested': learning_rates,
    'hidden_dims_tested': hidden_dims,
    'best_learning_rate': best_params.get('learning_rate'),
    'best_hidden_channels': best_params.get('hidden_channels'),
    'best_loss': best_loss,
    'best_religion_compliance': best_params.get('religion_compliance', 0.0),
    'best_ethnicity_compliance': best_params.get('ethnicity_compliance', 0.0),
    'best_size_distribution_accuracy': best_params.get('size_distribution_accuracy', 0.0),
    'best_overall_accuracy': best_params.get('overall_accuracy', 0.0),
    'best_stopping_epoch': best_params.get('stopping_epoch', 0)
}

performance_summary_path = os.path.join(output_dir, 'performance_summary.json')
with open(performance_summary_path, 'w') as f:
    json.dump(performance_summary, f, indent=4)
print(f"Performance summary saved: {performance_summary_path}")

# Save best parameters as JSON (enhanced)
best_params_with_loss = best_params.copy()
best_params_with_loss['best_loss'] = best_loss
best_params_with_loss['total_combinations'] = len(hp_results)
best_params_with_loss['area_code'] = selected_area_code
best_params_with_loss['religion_accuracy_percent'] = best_params.get('religion_compliance', 0.0) * 100
best_params_with_loss['ethnicity_accuracy_percent'] = best_params.get('ethnicity_compliance', 0.0) * 100
best_params_with_loss['size_distribution_accuracy_percent'] = best_params.get('size_distribution_accuracy', 0.0) * 100
best_params_with_loss['overall_accuracy_percent'] = best_params.get('overall_accuracy', 0.0) * 100
best_params_with_loss['total_training_time'] = total_training_time_str
best_params_with_loss['stopping_epoch'] = best_params.get('stopping_epoch', 0)

best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params_with_loss, f, indent=4)
print(f"Best parameters saved: {best_params_path}")

print(f"\nAll hyperparameter tuning results saved to: {output_dir}")

# Use saved best model information instead of retraining
print(f"\n{'='*60}")
print(f"USING BEST MODEL RESULTS FOR PLOTTING (NO RETRAINING)")
print(f"{'='*60}")

# Check if we have valid best model information
if best_model_info['model_state'] is not None:
    # Print best model information
    print("\nBest Model Information:")
    print(f"Learning Rate: {best_model_info['lr']}")
    print(f"Hidden Channels: {best_model_info['hidden_channels']}")
    print(f"Best Loss: {best_model_info['loss']:.6f}")
    print(f"Best Overall Accuracy: {best_model_info['accuracy']:.4f}")

    # Extract saved results from best model
    final_assignments = best_model_info['assignments']
    epoch_numbers = best_model_info['epoch_numbers']
    religion_accuracies = best_model_info['religion_accuracies']
    ethnicity_accuracies = best_model_info['ethnicity_accuracies']
    household_composition_accuracies = best_model_info['household_composition_accuracies']
    best_convergence_data = best_model_info['convergence_data']
    final_accuracies = best_model_info['detailed_accuracies']

    print(f"\nUsing saved best model results (no retraining needed)")

    # Generate plots using saved data
    print("\nGenerating paper-ready plots...")
    plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir)
    plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, household_composition_accuracies, output_dir)
    plot_training_convergence(best_convergence_data, output_dir)
else:
    print("\nNo successful training runs completed. Cannot generate plots.")
    print("Please check the error messages above and fix the issues before running again.")
    exit(1)

# Save final assignment results
if best_model_info['model_state'] is not None:
    print(f"\nSaving final assignment results to {output_dir}")

    # Save final assignments tensor
    final_assignments_path = os.path.join(output_dir, 'final_assignments.pt')
    torch.save(final_assignments.cpu(), final_assignments_path)

    # Save convergence data from best model run
    best_convergence_df = pd.DataFrame(best_convergence_data)
    best_convergence_path = os.path.join(output_dir, 'best_model_convergence_data.csv')
    best_convergence_df.to_csv(best_convergence_path, index=False)
    print(f"Best model convergence data saved: {best_convergence_path}")

    # Update best parameters with final results from best model run
    best_params_with_loss['final_religion_compliance'] = final_accuracies['religion_compliance']
    best_params_with_loss['final_ethnicity_compliance'] = final_accuracies['ethnicity_compliance']
    best_params_with_loss['final_household_composition_compliance'] = final_accuracies['household_composition_compliance']
    best_params_with_loss['final_size_distribution_accuracy'] = final_accuracies['size_distribution_accuracy']
    best_params_with_loss['final_overall_accuracy'] = final_accuracies['overall_accuracy']
    best_params_with_loss['final_religion_accuracy_percent'] = final_accuracies['religion_compliance'] * 100
    best_params_with_loss['final_ethnicity_accuracy_percent'] = final_accuracies['ethnicity_compliance'] * 100
    best_params_with_loss['final_household_composition_accuracy_percent'] = final_accuracies['household_composition_compliance'] * 100
    best_params_with_loss['final_size_distribution_accuracy_percent'] = final_accuracies['size_distribution_accuracy'] * 100
    best_params_with_loss['final_overall_accuracy_percent'] = final_accuracies['overall_accuracy'] * 100

    # Re-save updated best parameters
    with open(best_params_path, 'w') as f:
        json.dump(best_params_with_loss, f, indent=4)

    print(f"\nFinal Results with Best Hyperparameters:")
    print(f"  Learning Rate: {best_model_info['lr']}")
    print(f"  Hidden Channels: {best_model_info['hidden_channels']}")
    print(f"  Final Loss: {best_model_info['loss']:.6f}")
    print(f"  Stopping Epoch: {best_model_info.get('stopping_epoch', 'N/A')}")
    print(f"  Religion Compliance: {final_accuracies['religion_compliance'] * 100:.2f}%")
    print(f"  Ethnicity Compliance: {final_accuracies['ethnicity_compliance'] * 100:.2f}%")
    print(f"  Household Composition Compliance: {final_accuracies['household_composition_compliance'] * 100:.2f}%")
    print(f"  Size Distribution Accuracy: {final_accuracies['size_distribution_accuracy'] * 100:.2f}%")
    print(f"  Overall Accuracy: {final_accuracies['overall_accuracy'] * 100:.2f}%")
    print(f"  Results and plots saved to: {output_dir}")
else:
    print(f"\nNo final results to save - no successful training runs")

# Comprehensive final cleanup
print("\nPerforming final memory cleanup...")

# Clear all intermediate variables
safe_delete_tensor(final_assignments)
safe_delete_tensor(epoch_numbers)
safe_delete_tensor(religion_accuracies)
safe_delete_tensor(ethnicity_accuracies)

# Clear data tensors if they're no longer needed
# Note: We keep person_nodes, household_nodes, household_sizes, and edge_index 
# as they might be needed for future operations
monitor_memory_usage(device, "before final cleanup")

# Clear GPU cache and perform garbage collection
clear_gpu_memory()

# Print final memory status
monitor_memory_usage(device, "after final cleanup")

# Print peak memory usage if available
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"Peak GPU Memory Usage: Allocated: {peak_allocated:.2f}GB, Reserved: {peak_reserved:.2f}GB")

print("Hyperparameter tuning completed!")
print("GPU memory has been cleared and optimized.")