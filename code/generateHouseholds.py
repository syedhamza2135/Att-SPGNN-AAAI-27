import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm, GATConv
from torch_geometric.utils import add_self_loops
import random
import json
import time
from datetime import timedelta
import numpy as np
from collections import Counter
import shutil  # For directory operations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import argparse
import geopandas as gpd  # Added for geo plotting
import plotly.express as px  # Added for geo plotting

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic households using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Generation for area: {selected_area_code}")

# Set display options permanently
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=torch.inf)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_geo_plot_trace(selected_area_code, current_dir):
    """
    Create a geo plot trace showing all areas in white except the selected area code which is shaded.
    Returns the geo traces that can be added to a subplot.
    
    Parameters:
    selected_area_code - The area code to highlight
    current_dir - The current directory path for file loading
    """
    try:
        # Define paths relative to the code directory
        BASE = os.path.join(current_dir, '../data/')
        PERSONS_DIR = os.path.join(current_dir, '../data/preprocessed-data/individuals')
        
        # Load age data and rename the geography code
        age_file = os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv")
        if not os.path.exists(age_file):
            print(f"Warning: Age data file not found at {age_file}")
            return [], {}
            
        age = pd.read_csv(age_file)
        age = age.rename(columns={"geography code": "MSOA21CD"})
        
        # Load shapefiles
        msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
        red_fp = os.path.join(BASE, "geodata", "boundary.geojson")
        
        if not os.path.exists(msoa_fp) or not os.path.exists(red_fp):
            print(f"Warning: Geodata files not found. Expected at {msoa_fp} and {red_fp}")
            return [], {}
        
        # Read in spatial data
        gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
        red_bnd = gpd.read_file(red_fp).to_crs(4326)
        
        # Merge population totals
        gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
        gdf_msoa["total"] = gdf_msoa["total"].fillna(0)
        
        # Manually remove unwanted MSOAs
        exclude_codes = [
            "E02005939", "E02005979", "E02005963", "E02005959"
        ]
        gdf_msoa = gdf_msoa[~gdf_msoa["MSOA21CD"].isin(exclude_codes)]
        
        # Clip to red boundary
        red_union = red_bnd.unary_union
        gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()
        
        # Create color column: selected area gets color 1, others get color 0
        gdf_clip["color_value"] = gdf_clip["MSOA21CD"].apply(
            lambda x: 1 if x == selected_area_code else 0
        )
        
        # Compute accurate centroids for labels
        proj_crs = 27700
        gdf_proj = gdf_clip.to_crs(proj_crs)
        centroids = gdf_proj.geometry.centroid.to_crs(4326)
        gdf_clip["lon"] = centroids.x
        gdf_clip["lat"] = centroids.y
        
        # Create traces list
        traces = []
        
        # Add choropleth trace
        choropleth_trace = go.Choropleth(
            geojson=json.loads(gdf_clip.to_json()),
            locations=gdf_clip["MSOA21CD"],
            featureidkey="properties.MSOA21CD",
            z=gdf_clip["color_value"],
            colorscale=[[0, "white"], [1, "lightblue"]],  # White for 0, light blue for selected
            showscale=False,
            hovertemplate="<b>%{location}</b><extra></extra>",
            name="Areas"
        )
        traces.append(choropleth_trace)
        
        # Add red boundary outline
        for poly in red_bnd.geometry.explode(index_parts=False):
            boundary_trace = go.Scattergeo(
                lon=list(poly.exterior.coords.xy[0]),
                lat=list(poly.exterior.coords.xy[1]),
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo="skip"
            )
            traces.append(boundary_trace)
        
        # Calculate bounds for the geo layout with reduced padding for bigger geo plots
        # Use red boundary bounds instead of just clipped areas for better coverage
        red_bounds = red_bnd.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = red_bounds
        
        # Reduce padding to make geo plots bigger within their allocated space
        lat_padding = (lat_max - lat_min) * 0.05  # Further reduced to 5% padding for larger geo plot
        lon_padding = (lon_max - lon_min) * 0.05  # Further reduced to 5% padding for larger geo plot
        
        geo_layout = {
            'visible': False,
            'lataxis_range': [lat_min - lat_padding, lat_max + lat_padding],
            'lonaxis_range': [lon_min - lon_padding, lon_max + lon_padding],
            'projection_type': 'mercator'
        }
        
        return traces, geo_layout
        
    except Exception as e:
        print(f"Warning: Could not create geo plot trace: {e}")
        return [], {}

def get_target_tensors_2way(cross_table, feature_1_categories, feature_1_map, feature_2_categories, feature_2_map):
    """
    Get target tensors for 2-way crosstables (e.g., hhcomp×ethnicity, hhcomp×religion)
    """
    y_feature_1 = torch.zeros(num_households, dtype=torch.long, device=device)
    y_feature_2 = torch.zeros(num_households, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross-table and feature categories
    household_idx = 0

    for _, row in cross_table.iterrows():
        for feature_1 in feature_1_categories:  # First attribute (e.g., hhcomp)
            for feature_2 in feature_2_categories:  # Second attribute (e.g., ethnicity/religion)
                col_name = f'{feature_1} {feature_2}'
                count = int(row.get(col_name, 0))
                for _ in range(count):
                    if household_idx < num_households:
                        y_feature_1[household_idx] = feature_1_map.get(feature_1, -1)
                        y_feature_2[household_idx] = feature_2_map.get(feature_2, -1)
                        household_idx += 1

    return (y_feature_1, y_feature_2)

def get_target_tensors_3way(cross_table, feature_1_categories, feature_1_map, feature_2_categories, feature_2_map, feature_3_categories, feature_3_map):
    """
    Get target tensors for 3-way crosstables (e.g., tenure×size×rooms)
    """
    y_feature_1 = torch.zeros(num_households, dtype=torch.long, device=device)
    y_feature_2 = torch.zeros(num_households, dtype=torch.long, device=device)
    y_feature_3 = torch.zeros(num_households, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross-table and feature categories
    household_idx = 0

    for _, row in cross_table.iterrows():
        for feature_1 in feature_1_categories:  # First attribute (e.g., tenure)
            for feature_2 in feature_2_categories:  # Second attribute (e.g., size)
                for feature_3 in feature_3_categories:  # Third attribute (e.g., rooms)
                    col_name = f'{feature_1} {feature_2} {feature_3}'
                    count = int(row.get(col_name, 0))
                    for _ in range(count):
                        if household_idx < num_households:
                            y_feature_1[household_idx] = feature_1_map.get(feature_1, -1)
                            y_feature_2[household_idx] = feature_2_map.get(feature_2, -1)
                            y_feature_3[household_idx] = feature_3_map.get(feature_3, -1)
                            household_idx += 1

    return (y_feature_1, y_feature_2, y_feature_3)

# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
# NOTE: ethnicity_df and religion_df are PERSON-level data, NOT household-level!
# They are loaded for reference/comparison but NOT used for household generation.
# Household-level distributions are derived from crosstables instead.
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Households.csv'))
hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Updated.csv'))
# New attribute tables
tenure_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Tenure_Updated.csv'))
hh_size_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_size_Updated.csv'))
rooms_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Number_of_rooms_Updated.csv'))

# Load crosstables
# hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity.csv'))
# hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion.csv'))
hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion_Updated.csv'))
hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_Updated.csv'))
# New crosstable
tenure_by_size_by_rooms_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/Tenure_by_household_size_by_number_of_rooms_Updated.csv'))

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")

ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']
# New attribute categories
tenure_categories = ['OW', 'SO', 'SR', 'PR']  # You may need to adjust based on your data
size_categories = ['1', '2', '3', '4+']
rooms_categories = ['1', '2', '3', '4', '5', '6+']  # You may need to adjust based on your data

# Filter the DataFrame for the specified Oxford areas
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
hhcomp_df = hhcomp_df[hhcomp_df['geography code'].isin(oxford_areas)]
# Filter new dataframes
tenure_df = tenure_df[tenure_df['geography code'].isin(oxford_areas)]
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]
rooms_df = rooms_df[rooms_df['geography code'].isin(oxford_areas)]
# Filter crosstables
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'].isin(oxford_areas)]
hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'].isin(oxford_areas)]
tenure_by_size_by_rooms_df = tenure_by_size_by_rooms_df[tenure_by_size_by_rooms_df['geography code'].isin(oxford_areas)]

# Drop unnecessary columns from new crosstable
tenure_by_size_by_rooms_df = tenure_by_size_by_rooms_df.drop(columns = ['total', 'geography code'])

num_households = int(hhcomp_df['total'].iloc[0])
print(f"Number of households: {num_households}")

# Preprocess household composition data
# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-1C'] + hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-1C'] + hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-1C'] + hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-1C'] + hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(['1FM', '1FC', '1FL'], axis=1)

# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-nC', '1FC-nC', '1FL-nC', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(columns=['total', 'geography code'], inplace=True)

hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-nA', '1FL-2C', '1H-nS', '1H-nE', '1H-nA', '1H-2C']
# hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-nC', '1FM-nA','1FC-0C','1FC-nC','1FC-nA','1FL-nA','1FL-nC','1H-nS','1H-nE','1H-nA', '1H-nC']

# Filter and preprocess columns
# filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]
# filtered_columns = [col for col in hhcomp_by_religion_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_religion_df = hhcomp_by_religion_df[filtered_columns]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns = ['total', 'geography code'])
hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns = ['total', 'geography code'])

# Encode the categories to indices
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
hh_map = {category: i for i, category in enumerate(hh_compositions)}
# New attribute maps
tenure_map = {category: i for i, category in enumerate(tenure_categories)}
size_map = {category: i for i, category in enumerate(size_categories)}
rooms_map = {category: i for i, category in enumerate(rooms_categories)}

# Create household nodes with unique IDs
households_nodes = torch.arange(num_households).view(num_households, 1).to(device)

# Create nodes for ethnicity and religion categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Create nodes for new attributes
tenure_nodes = torch.tensor([[tenure_map[tenure]] for tenure in tenure_categories], dtype=torch.float).to(device)
size_nodes = torch.tensor([[size_map[size]] for size in size_categories], dtype=torch.float).to(device)
rooms_nodes = torch.tensor([[rooms_map[rooms]] for rooms in rooms_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([households_nodes, ethnicity_nodes, religion_nodes, tenure_nodes, size_nodes, rooms_nodes], dim=0).to(device)

# ============================================================================
# HOUSEHOLD-LEVEL MARGINAL DISTRIBUTIONS
# ============================================================================
# CRITICAL: Ethnicity and Religion distributions must be derived from CROSSTABLES,
# not from person-level data. Person-level distributions are weighted by household
# size and will incorrectly overrepresent ethnicities/religions with larger households.

def derive_marginal_from_crosstable(crosstable_df, attribute_categories, hh_compositions):
    """
    Derive HOUSEHOLD-LEVEL marginal distribution of an attribute from HHcomp x Attribute crosstable.
    
    This is the CORRECT way to get household ethnicity/religion distributions because:
    - Person-level data counts each PERSON once (weighted by household size)
    - Crosstable data counts each HOUSEHOLD once (HRP = Household Reference Person)
    
    Crosstable columns are formatted as: "{HHcomp} {Attribute}" (e.g., "1PE W1")
    
    Args:
        crosstable_df: DataFrame with columns like "1PE W1", "1PE W2", etc.
        attribute_categories: List of attribute categories (e.g., ['W1', 'W2', ...])
        hh_compositions: List of household compositions (e.g., ['1PE', '1PA', ...])
    
    Returns:
        List of probabilities for each attribute category (sums to 1.0)
    """
    marginal_counts = {}
    missing_combinations = []
    
    for attr in attribute_categories:
        # Sum all columns that end with this attribute across all compositions
        attr_cols = []
        for comp in hh_compositions:
            col_name = f"{comp} {attr}"
            if col_name in crosstable_df.columns:
                attr_cols.append(col_name)
            else:
                missing_combinations.append(col_name)
        
        if attr_cols:
            marginal_counts[attr] = crosstable_df[attr_cols].iloc[0].sum()
        else:
            marginal_counts[attr] = 0
    
    # Report missing combinations (helps debug data issues)
    if missing_combinations and len(missing_combinations) <= 20:
        print(f"  Note: {len(missing_combinations)} composition-attribute combinations not in crosstable")
    
    total = sum(marginal_counts.values())
    if total > 0:
        prob_list = [marginal_counts[attr] / total for attr in attribute_categories]
    else:
        # Fallback to uniform distribution if no data
        print(f"  WARNING: No counts found in crosstable, using uniform distribution")
        prob_list = [1.0 / len(attribute_categories)] * len(attribute_categories)
    
    return prob_list, marginal_counts

def validate_attribute_coverage(marginal_counts, attribute_categories, attribute_name):
    """
    Validate that all attribute categories have non-zero counts in household data.
    
    This catches the ethnicity range mismatch issue where some categories may
    have zero households but non-zero persons.
    """
    zero_categories = [cat for cat, count in marginal_counts.items() if count == 0]
    
    if zero_categories:
        print(f"\n⚠️  WARNING: {len(zero_categories)} {attribute_name} categories have ZERO households:")
        for cat in zero_categories:
            print(f"    - {cat}: 0 households (persons with this {attribute_name} cannot be matched!)")
        print(f"    This will cause matching failures in the assignment module.")
        return False
    else:
        print(f"✓ All {len(attribute_categories)} {attribute_name} categories have household coverage")
        return True

# Derive CORRECT household-level distributions from crosstables
print("\n=== DERIVING HOUSEHOLD-LEVEL MARGINAL DISTRIBUTIONS ===")
print("(Using crosstables instead of person-level data)")

print("\nDeriving household ethnicity distribution from HHcomp_by_ethnicity crosstable...")
ethnicity_prob_list, ethnicity_marginal_counts = derive_marginal_from_crosstable(
    hhcomp_by_ethnicity_df, ethnicity_categories, hh_compositions
)
validate_attribute_coverage(ethnicity_marginal_counts, ethnicity_categories, "ethnicity")

print("\nDeriving household religion distribution from HHcomp_by_religion crosstable...")
religion_prob_list, religion_marginal_counts = derive_marginal_from_crosstable(
    hhcomp_by_religion_df, religion_categories, hh_compositions
)
validate_attribute_coverage(religion_marginal_counts, religion_categories, "religion")

# Print comparison: Person-level vs Household-level distributions
print("\n=== DISTRIBUTION COMPARISON: Person-level vs Household-level ===")
print("(This shows why using person-level data for households is incorrect)")

# Person-level ethnicity (from individuals/Ethnicity.csv) - for comparison only
person_ethnicity_counts = ethnicity_df[ethnicity_categories].iloc[0].to_dict()
person_ethnicity_total = sum(person_ethnicity_counts.values())

print("\nEthnicity distribution comparison:")
print(f"{'Category':<8} {'Persons':>10} {'Person%':>10} {'Households':>12} {'HH%':>10} {'Ratio':>8}")
print("-" * 60)
for cat in ethnicity_categories:
    p_count = person_ethnicity_counts.get(cat, 0)
    p_pct = (p_count / person_ethnicity_total * 100) if person_ethnicity_total > 0 else 0
    h_count = ethnicity_marginal_counts.get(cat, 0)
    h_pct = (h_count / num_households * 100) if num_households > 0 else 0
    ratio = (p_pct / h_pct) if h_pct > 0 else float('inf')
    # Only print if there's a significant difference or it's a major category
    if abs(ratio - 1.0) > 0.1 or p_count > person_ethnicity_total * 0.05:
        print(f"{cat:<8} {p_count:>10} {p_pct:>9.1f}% {h_count:>12} {h_pct:>9.1f}% {ratio:>7.2f}x")

print("\n(Ratio > 1 means ethnicity is overrepresented in person data due to larger households)")

# These use actual household-level data (correct)
tenure_prob_list    = (tenure_df[tenure_categories].iloc[0]       / num_households).tolist()
size_prob_list      = (hh_size_df[size_categories].iloc[0]        / num_households).tolist()
rooms_prob_list     = (rooms_df[rooms_categories].iloc[0]         / num_households).tolist()

print("\n✓ Household attribute probability distributions ready")

# Edge index generation
def generate_edge_index(num_households):
    edge_index = []
    num_ethnicities = len(ethnicity_map)
    num_religions = len(religion_map)
    num_tenures = len(tenure_map)
    num_sizes = len(size_map)
    num_rooms = len(rooms_map)

    ethnicity_start_idx = num_households
    religion_start_idx = ethnicity_start_idx + num_ethnicities
    tenure_start_idx = religion_start_idx + num_religions
    size_start_idx = tenure_start_idx + num_tenures
    rooms_start_idx = size_start_idx + num_sizes

    for i in range(num_households):
        # Weighted sampling based on area marginals
        ethnicity_category = random.choices(
            range(ethnicity_start_idx, ethnicity_start_idx + num_ethnicities),
            weights=ethnicity_prob_list, k=1
        )[0]
        religion_category = random.choices(
            range(religion_start_idx, religion_start_idx + num_religions),
            weights=religion_prob_list, k=1
        )[0]
        tenure_category = random.choices(
            range(tenure_start_idx, tenure_start_idx + num_tenures),
            weights=tenure_prob_list, k=1
        )[0]
        size_category = random.choices(
            range(size_start_idx, size_start_idx + num_sizes),
            weights=size_prob_list, k=1
        )[0]
        rooms_category = random.choices(
            range(rooms_start_idx, rooms_start_idx + num_rooms),
            weights=rooms_prob_list, k=1
        )[0]
        
        # Append edges for the selected categories
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])
        edge_index.append([i, tenure_category])
        edge_index.append([i, size_category])
        edge_index.append([i, rooms_category])

    # Convert edge_index to a tensor and transpose for PyTorch Geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index
edge_index = generate_edge_index(num_households)

# Make edges bidirectional and add self-loops
total_nodes = node_features.size(0)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
edge_index, _ = add_self_loops(edge_index, num_nodes=total_nodes)

# Create the data object for PyTorch Geometric with unique node IDs as features
data = Data(x=torch.arange(total_nodes, device=device), edge_index=edge_index).to(device)

# Enhanced GNN Model
class AttentiveMultiTaskGNN(torch.nn.Module):
    """
    Multi–head Graph‑Attention network for synthetic‑household generation.
    * node_ids  : long tensor (same as your current x)                 [N]
    * edge_index: bidirectional edges household ↔ attribute categories    [2,E]
    The first `num_households` nodes are households – identical assumption
    to your existing code, so nothing else in the script has to change.
    """
    def __init__(
        self,
        num_nodes: int,
        num_households: int,
        embed_dim: int,
        hidden_dim: int,
        heads: int,
        out_dims: dict[str, int],      # {"hhcomp":15, "ethnicity":18, ...}
        dropout: float = 0.8
    ):
        super().__init__()
        self.num_households = num_households
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 1️⃣  Learnable embedding for every node‑id  --------------------------
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
        
        # Projection layer to match dimensions for residual connections (if needed)
        self.proj = torch.nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else None

        # 2️⃣  Three GAT layers with residual connections  --------------------
        self.gat1 = GATConv(embed_dim,  hidden_dim, heads=heads,
                            concat=False, dropout=dropout)
        self.norm1 = GraphNorm(hidden_dim)

        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads,
                            concat=False, dropout=dropout)
        self.norm2 = GraphNorm(hidden_dim)

        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=heads,
                            concat=False, dropout=dropout)
        self.norm3 = GraphNorm(hidden_dim)

        # 3️⃣  Shared MLP trunk (optional but helps)  -------------------------
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

        # 4️⃣  Light task‑specific heads  -------------------------------------
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(hidden_dim, out_dim)
            for name, out_dim in out_dims.items()
        })

    # -------------------------------------------------------------------------
    def forward(self, data):
        x = self.embedding(data.x.squeeze().long())      # [N, embed_dim]
        
        # Project embeddings to hidden_dim if dimensions don't match
        if self.proj is not None:
            x_proj = self.proj(x)  # [N, hidden_dim]
        else:
            x_proj = x  # Dimensions already match

        h1 = F.relu(self.norm1(self.gat1(x, data.edge_index)))
        h1 = h1 + x_proj                                 # residual (now dimensions match)

        h2 = F.relu(self.norm2(self.gat2(h1, data.edge_index)))
        h2 = h2 + h1                                     # residual

        h3 = F.relu(self.norm3(self.gat3(h2, data.edge_index)))
        h   = h3 + h2                                    # residual

        # Use only *household* node embeddings for prediction
        z = self.trunk(h[:self.num_households])

        return {name: head(z) for name, head in self.heads.items()}

# Get target tensors
targets = []

# 2-way targets using appropriate function
targets.append(
    (
        ('hhcomp', 'ethnicity'), 
        get_target_tensors_2way(hhcomp_by_ethnicity_df, hh_compositions, hh_map, ethnicity_categories, ethnicity_map)
    )
)

targets.append(
    (
        ('hhcomp', 'religion'), 
        get_target_tensors_2way(hhcomp_by_religion_df, hh_compositions, hh_map, religion_categories, religion_map)
    )
)

# 3-way target using appropriate function
targets.append(
    (
        ('tenure', 'size', 'rooms'), 
        get_target_tensors_3way(tenure_by_size_by_rooms_df, tenure_categories, tenure_map, size_categories, size_map, rooms_categories, rooms_map)
    )
)

# Hyperparameter Tuning
# ============================================================================
# IMPORTANT: GAT-based architecture requires lower learning rates than SAGE!
# Anti-overfitting configurations:
#   - For quick testing:  learning_rates = [0.001], hidden_channel_options = [128]
#   - For full search:    learning_rates = [0.001, 0.0005], hidden_channel_options = [64, 128]
#   - Avoid LR > 0.001 with GAT (causes instability)
#   - Smaller hidden dimensions (64, 128) reduce overfitting compared to larger ones (256)
# ============================================================================
# learning_rates = [0.001, 0.0005]  # Stable for GAT (reduced from 0.005)
# learning_rates = [0.0005]  # Stable for GAT (reduced from 0.005)
learning_rates = [0.001, 0.0005, 0.0001]
hidden_channel_options = [128, 256]  # Smaller dimensions to prevent overfitting (reduced from [128, 256])
# learning_rates = [0.0005]
# hidden_channel_options = [256]  # Smaller dimensions to prevent overfitting (reduced from [128, 256])
mlp_hidden_dim = 256
num_epochs = 2000  # With early stopping (patience=100), typically stops around 300-700 epochs

# Results storage
results = []
time_results = []
best_model_info = {
    'model_state': None,
    'loss': float('inf'),
    'accuracy': 0,
    'predictions': None,
    'lr': None,
    'hidden_channels': None,
    'training_time': None
}

# Function to calculate R-squared accuracy
def calculate_r2_accuracy(generated_counts, target_counts):
    """
    Simple R² measure comparing two distributions:
    R² = 1 - (SSE / SST), with SSE = sum of squared errors
    """
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)

    sse = np.sum((gen_vals - tgt_vals) ** 2)
    sst = np.sum((tgt_vals - tgt_vals.mean()) ** 2)

    return 1.0 - sse / sst if sst > 1e-12 else 1.0

# Optimized GPU-friendly accuracy function for multi-task learning
def calculate_distribution_task_accuracy(pred_1, pred_2, target_combination, actual_crosstable, pred_3=None):
    """
    Fast GPU-optimized distribution-based accuracy calculation.
    Uses tensor operations instead of pandas for speed during training.
    Handles both 2-way and 3-way combinations.
    """
    if pred_3 is None:
        # 2-way calculation
        categories_1, categories_2 = target_combination
        
        # Map attribute names to category counts
        category_sizes = {
            'hhcomp': len(hh_compositions),
            'ethnicity': len(ethnicity_categories),
            'religion': len(religion_categories),
            'tenure': len(tenure_categories),
            'size': len(size_categories),
            'rooms': len(rooms_categories)
        }
        
        size_1 = category_sizes[categories_1]
        size_2 = category_sizes[categories_2]
        
        # Create predicted counts tensor (keep on GPU)
        combo_indices = pred_1 * size_2 + pred_2
        total_combinations = size_1 * size_2
        
        # Count occurrences efficiently on GPU
        predicted_counts = torch.bincount(combo_indices, minlength=total_combinations).float()
        
        # Pre-compute actual counts tensor (do this only once, not every epoch)
        cache_key = f'actual_counts_{categories_1}_{categories_2}'
        if not hasattr(calculate_distribution_task_accuracy, cache_key):
            # Extract actual counts and convert to tensor format
            actual_counts_tensor = torch.zeros(total_combinations, dtype=torch.float, device=device)
            
            category_map = {
                'hhcomp': hh_compositions,
                'ethnicity': ethnicity_categories,
                'religion': religion_categories,
                'tenure': tenure_categories,
                'size': size_categories,
                'rooms': rooms_categories
            }
            
            cats_1 = category_map[categories_1]
            cats_2 = category_map[categories_2]
            
            # Order: household compositions first, then ethnicity/religion
            for i1, cat1 in enumerate(cats_1):
                for i2, cat2 in enumerate(cats_2):
                    original_col = f'{cat1} {cat2}'
                    if original_col in actual_crosstable.columns:
                        combo_idx = i1 * size_2 + i2
                        actual_counts_tensor[combo_idx] = actual_crosstable[original_col].iloc[0]
            
            # Cache the result to avoid recomputation
            setattr(calculate_distribution_task_accuracy, cache_key, actual_counts_tensor)
        
        actual_counts = getattr(calculate_distribution_task_accuracy, cache_key)
        
    else:
        # 3-way calculation
        categories_1, categories_2, categories_3 = target_combination
        
        # Map attribute names to category counts
        category_sizes = {
            'hhcomp': len(hh_compositions),
            'ethnicity': len(ethnicity_categories),
            'religion': len(religion_categories),
            'tenure': len(tenure_categories),
            'size': len(size_categories),
            'rooms': len(rooms_categories)
        }
        
        size_1 = category_sizes[categories_1]
        size_2 = category_sizes[categories_2]
        size_3 = category_sizes[categories_3]
        
        # Create predicted counts tensor (keep on GPU) - using correct ordering: tenure -> size -> rooms
        combo_indices = pred_1 * (size_2 * size_3) + pred_2 * size_3 + pred_3
        total_combinations = size_1 * size_2 * size_3
        
        # Count occurrences efficiently on GPU
        predicted_counts = torch.bincount(combo_indices, minlength=total_combinations).float()
        
        # Pre-compute actual counts tensor (do this only once, not every epoch)
        cache_key = f'actual_counts_{categories_1}_{categories_2}_{categories_3}'
        if not hasattr(calculate_distribution_task_accuracy, cache_key):
            # Extract actual counts and convert to tensor format
            actual_counts_tensor = torch.zeros(total_combinations, dtype=torch.float, device=device)
            
            category_map = {
                'hhcomp': hh_compositions,
                'ethnicity': ethnicity_categories,
                'religion': religion_categories,
                'tenure': tenure_categories,
                'size': size_categories,
                'rooms': rooms_categories
            }
            
            cats_1 = category_map[categories_1]
            cats_2 = category_map[categories_2]
            cats_3 = category_map[categories_3]
            
            # Order: feature_1 first, then feature_2, then feature_3 (tenure -> size -> rooms)
            for i1, cat1 in enumerate(cats_1):
                for i2, cat2 in enumerate(cats_2):
                    for i3, cat3 in enumerate(cats_3):
                        original_col = f'{cat1} {cat2} {cat3}'
                        if original_col in actual_crosstable.columns:
                            combo_idx = i1 * (size_2 * size_3) + i2 * size_3 + i3
                            actual_counts_tensor[combo_idx] = actual_crosstable[original_col].iloc[0]
            
            # Cache the result to avoid recomputation
            setattr(calculate_distribution_task_accuracy, cache_key, actual_counts_tensor)
        
        actual_counts = getattr(calculate_distribution_task_accuracy, cache_key)
    
    # Calculate R² efficiently on GPU
    actual_mean = actual_counts.mean()
    ss_tot = torch.sum((actual_counts - actual_mean) ** 2)
    ss_res = torch.sum((actual_counts - predicted_counts) ** 2)
    
    if ss_tot > 1e-12:
        r2 = 1.0 - (ss_res / ss_tot)
        return max(0.0, r2.item())  # Ensure non-negative and convert to Python float
    else:
        return 1.0

# Function to calculate RMSE
def calculate_rmse(generated_counts, target_counts):
    """
    Calculate RMSE between two distributions (dicts of category: count).
    """
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)
    mse = np.mean((gen_vals - tgt_vals) ** 2)
    return np.sqrt(mse)

# Custom loss function for 3-way targets
def custom_loss_function_3way(first_out, second_out, third_out, y_first, y_second, y_third):
    loss_first = F.cross_entropy(first_out, y_first)
    loss_second = F.cross_entropy(second_out, y_second)
    loss_third = F.cross_entropy(third_out, y_third)
    total_loss = loss_first + loss_second + loss_third
    return total_loss

# Custom loss function for 2-way targets
def custom_loss_function_2way(first_out, second_out, y_first, y_second):
    loss_first = F.cross_entropy(first_out, y_first)
    loss_second = F.cross_entropy(second_out, y_second)
    total_loss = loss_first + loss_second
    return total_loss

# Function to train model
def train_model(lr, hidden_channels, num_epochs, data, targets,
                embed_dim=128, heads=4, dropout=0.8):
    """
    Train the GAT-based multi-task GNN for household generation.
    
    Improvements for stability and preventing overfitting:
    - Balanced dropout at 0.8 (prevents overfitting while maintaining performance)
    - Separate embed_dim (128) from hidden_channels (prevents tiny embeddings)
    - Added learning rate scheduler (ReduceLROnPlateau)
    - Added gradient clipping (max_norm=1.0)
    - Added early stopping (patience=100 epochs) for faster overfitting detection
    - Increased weight decay (1e-4) for stronger regularization
    
    Args:
        lr: Learning rate (recommend 0.001 or 0.0005 for GAT)
        hidden_channels: GAT hidden dimension size
        num_epochs: Maximum training epochs
        data: PyTorch Geometric Data object
        targets: List of target combinations and tensors
        embed_dim: Node embedding dimension (default 128, stable across configs)
        heads: Number of attention heads (default 4)
        dropout: Dropout rate (default 0.8, balanced for regularization)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Build output‑dimension dictionary once
    # ─────────────────────────────────────────────────────────────────────────
    out_dims = {
        "hhcomp": len(hh_compositions),
        "ethnicity": len(ethnicity_categories),
        "religion": len(religion_categories),
        "tenure": len(tenure_categories),
        "size": len(size_categories),
        "rooms": len(rooms_categories)
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Instantiate model + optimizer
    # ─────────────────────────────────────────────────────────────────────────
    model = AttentiveMultiTaskGNN(
        num_nodes=node_features.size(0),
        num_households=num_households,
        embed_dim=128,  # Keep embeddings stable, independent of hidden_channels
        hidden_dim=hidden_channels,
        heads=heads,
        out_dims=out_dims,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Increased from 1e-5 for stronger L2 regularization
    
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30  # Reduced from 50 for faster adaptation
    )

    # Track best epoch weights
    best_epoch_loss = float('inf')
    best_epoch_state = None

    # Convergence tracking
    convergence_data = {
        'epochs': [], 'losses': [], 'accuracies': [],
        'cumulative_time_seconds': [], 'epoch_time_seconds': []
    }
    training_start = time.time()
    epoch_acc_history = []
    
    # Early stopping parameters
    patience = 100  # Stop if no improvement for 100 epochs (reduced from 150 to catch overfitting faster)
    patience_counter = 0
    min_improvement = 1e-4  # Minimum loss improvement to reset patience

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        ep_start = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        outputs = {k: v[:num_households] for k, v in outputs.items()}

        # Multi‑task loss
        loss = 0.0
        for target_tuple in targets:
            target_attrs, target_tensors = target_tuple
            if len(target_attrs) == 2:  # 2-way targets
                a1, a2 = target_attrs
                y1, y2 = target_tensors
                loss += custom_loss_function_2way(
                    outputs[a1], outputs[a2], y1, y2
                )
            elif len(target_attrs) == 3:  # 3-way targets
                a1, a2, a3 = target_attrs
                y1, y2, y3 = target_tensors
                loss += custom_loss_function_3way(
                    outputs[a1], outputs[a2], outputs[a3], y1, y2, y3
                )

        # Accuracy every 100 epochs for speed
        if (epoch + 1) % 100 == 0:
            task_acc = []
            name_map = {
                0: "Household‑Ethnicity",
                1: "Household‑Religion", 
                2: "Tenure‑Size‑Rooms"
            }

            for i, (target_attrs, _) in enumerate(targets):
                if len(target_attrs) == 2:  # 2-way targets
                    a1, a2 = target_attrs
                    p1, p2 = [outputs[a].argmax(1) for a in (a1, a2)]
                    actual_ct = [
                        hhcomp_by_ethnicity_df,
                        hhcomp_by_religion_df,
                        tenure_by_size_by_rooms_df
                    ][i]

                    acc = calculate_distribution_task_accuracy(
                            p1, p2, target_attrs, actual_ct)
                elif len(target_attrs) == 3:  # 3-way targets
                    a1, a2, a3 = target_attrs
                    p1, p2, p3 = [outputs[a].argmax(1) for a in (a1, a2, a3)]
                    actual_ct = [
                        hhcomp_by_ethnicity_df,
                        hhcomp_by_religion_df,
                        tenure_by_size_by_rooms_df
                    ][i]

                    acc = calculate_distribution_task_accuracy(
                            p1, p2, target_attrs, actual_ct, p3)
                task_acc.append(acc)

            avg_acc = sum(task_acc) / len(task_acc)
            epoch_acc_history.append(avg_acc)

            print(f"\nEpoch {epoch+1:>4}/{num_epochs} "
                  f" | loss: {loss.item():.4f} | dist‑acc: {avg_acc:.4f}")

        # Track best epoch and early stopping
        current_loss = loss.item()
        if current_loss < best_epoch_loss - min_improvement:
            best_epoch_loss = current_loss
            best_epoch_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience on improvement
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}: No improvement for {patience} epochs")
            print(f"Best loss: {best_epoch_loss:.4f}")
            break

        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate based on loss plateau
        scheduler.step(loss)

        # Calculate epoch timing
        ep_end = time.time()
        ep_duration = ep_end - ep_start
        cumulative_time = ep_end - training_start

        # Store convergence data
        convergence_data['epochs'].append(epoch + 1)
        convergence_data['losses'].append(loss.item())
        convergence_data['epoch_time_seconds'].append(ep_duration)
        convergence_data['cumulative_time_seconds'].append(cumulative_time)
        
        # Store accuracy data (only when calculated)
        if (epoch + 1) % 100 == 0:
            convergence_data['accuracies'].append(avg_acc)
        else:
            convergence_data['accuracies'].append(None)  # Placeholder for missing accuracy

    # Calculate average accuracy across all epochs
    average_accuracy = sum(epoch_acc_history) / len(epoch_acc_history) if epoch_acc_history else 0

    # Load best epoch state for evaluation
    model.load_state_dict(best_epoch_state)
    
    # Evaluate predictions
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        outputs = {k: v[:num_households] for k, v in outputs.items()}
        
        # Get the predicted class for each attribute by taking argmax of output logits
        hh_pred = outputs['hhcomp'].argmax(dim=1)
        ethnicity_pred = outputs['ethnicity'].argmax(dim=1)
        religion_pred = outputs['religion'].argmax(dim=1)
        tenure_pred = outputs['tenure'].argmax(dim=1)
        size_pred = outputs['size'].argmax(dim=1)
        rooms_pred = outputs['rooms'].argmax(dim=1)
        
        # Calculate distribution-based accuracy across all tasks
        net_accuracy = 0
        final_task_accuracies = {}
        
        for i in range(len(targets)):
            target_attrs, _ = targets[i]
            
            if len(target_attrs) == 2:  # 2-way targets
                a1, a2 = target_attrs
                pred_1 = outputs[a1].argmax(dim=1)
                pred_2 = outputs[a2].argmax(dim=1)
                
                # Get the corresponding actual cross-table
                actual_crosstable = [
                    hhcomp_by_ethnicity_df,
                    hhcomp_by_religion_df,
                    tenure_by_size_by_rooms_df
                ][i]
                
                # Calculate distribution-based accuracy (R²)
                task_distribution_accuracy = calculate_distribution_task_accuracy(
                    pred_1, pred_2, target_attrs, actual_crosstable
                )
            elif len(target_attrs) == 3:  # 3-way targets
                a1, a2, a3 = target_attrs
                pred_1 = outputs[a1].argmax(dim=1)
                pred_2 = outputs[a2].argmax(dim=1)
                pred_3 = outputs[a3].argmax(dim=1)
                
                # Get the corresponding actual cross-table
                actual_crosstable = [
                    hhcomp_by_ethnicity_df,
                    hhcomp_by_religion_df,
                    tenure_by_size_by_rooms_df
                ][i]
                
                # Calculate distribution-based accuracy (R²)
                task_distribution_accuracy = calculate_distribution_task_accuracy(
                    pred_1, pred_2, target_attrs, actual_crosstable, pred_3
                )
            
            # Accumulate accuracy across all target combinations
            net_accuracy += task_distribution_accuracy
            task_name = '_'.join(target_attrs)
            final_task_accuracies[task_name] = task_distribution_accuracy * 100
        
        final_accuracy = net_accuracy / len(targets)
        
        # Print final task accuracies
        print(f"\n=== DISTRIBUTION-BASED ACCURACY RESULTS ===")
        for task, acc in final_task_accuracies.items():
            print(f"{task} distribution accuracy (R²): {acc:.2f}%")
        print(f"Overall distribution accuracy: {final_accuracy*100:.2f}%")
        
        # Calculate RMSE
        net_rmse = 0
        final_task_rmses = {}
        for i in range(len(targets)):
            target_attrs, _ = targets[i]
            
            if len(target_attrs) == 2:  # 2-way targets
                a1, a2 = target_attrs
                pred_1 = outputs[a1].argmax(dim=1)
                pred_2 = outputs[a2].argmax(dim=1)
                
                actual_crosstable = [
                    hhcomp_by_ethnicity_df,
                    hhcomp_by_religion_df,
                    tenure_by_size_by_rooms_df
                ][i]
                
                # Calculate RMSE for 2-way combination
                size_1 = len(hh_compositions if a1=='hhcomp' else (ethnicity_categories if a1=='ethnicity' else religion_categories))
                size_2 = len(religion_categories if a2=='religion' else ethnicity_categories)
                
                pred_counts = torch.bincount(pred_1 * size_2 + pred_2, minlength=size_1*size_2).cpu().numpy()
                
                # Build actual counts
                actual_counts = []
                cats_1 = hh_compositions if a1=='hhcomp' else (ethnicity_categories if a1=='ethnicity' else religion_categories)
                cats_2 = religion_categories if a2=='religion' else ethnicity_categories
                
                for cat1 in cats_1:
                    for cat2 in cats_2:
                        col = f'{cat1} {cat2}'
                        actual_counts.append(actual_crosstable[col].iloc[0] if col in actual_crosstable.columns else 0)
                
                pred_dict = {j: pred_counts[j] for j in range(len(pred_counts))}
                actual_dict = {j: actual_counts[j] for j in range(len(actual_counts))}
                task_rmse = calculate_rmse(pred_dict, actual_dict)
                
            elif len(target_attrs) == 3:  # 3-way targets
                a1, a2, a3 = target_attrs
                pred_1 = outputs[a1].argmax(dim=1)
                pred_2 = outputs[a2].argmax(dim=1)
                pred_3 = outputs[a3].argmax(dim=1)
                
                actual_crosstable = [
                    hhcomp_by_ethnicity_df,
                    hhcomp_by_religion_df,
                    tenure_by_size_by_rooms_df
                ][i]
                
                # Calculate RMSE for 3-way combination
                size_1 = len(tenure_categories if a1=='tenure' else (size_categories if a1=='size' else rooms_categories))
                size_2 = len(size_categories if a2=='size' else (tenure_categories if a2=='tenure' else rooms_categories))
                size_3 = len(rooms_categories if a3=='rooms' else (tenure_categories if a3=='tenure' else size_categories))
                
                # Use correct ordering: tenure -> size -> rooms (pred_1 -> pred_2 -> pred_3)
                pred_counts = torch.bincount(pred_1 * (size_2 * size_3) + pred_2 * size_3 + pred_3, minlength=size_1*size_2*size_3).cpu().numpy()
                
                # Build actual counts
                actual_counts = []
                cats_1 = tenure_categories if a1=='tenure' else (size_categories if a1=='size' else rooms_categories)
                cats_2 = size_categories if a2=='size' else (tenure_categories if a2=='tenure' else rooms_categories)
                cats_3 = rooms_categories if a3=='rooms' else (tenure_categories if a3=='tenure' else size_categories)
                
                # Order: cat1 first, then cat2, then cat3 (tenure -> size -> rooms)
                for cat1 in cats_1:
                    for cat2 in cats_2:
                        for cat3 in cats_3:
                            col = f'{cat1} {cat2} {cat3}'
                            actual_counts.append(actual_crosstable[col].iloc[0] if col in actual_crosstable.columns else 0)
                
                pred_dict = {j: pred_counts[j] for j in range(len(pred_counts))}
                actual_dict = {j: actual_counts[j] for j in range(len(actual_counts))}
                task_rmse = calculate_rmse(pred_dict, actual_dict)
            
            net_rmse += task_rmse
            task_name = '_'.join(target_attrs)
            final_task_rmses[task_name] = task_rmse
        
        overall_rmse = net_rmse / len(targets)
        # Print RMSE results
        print(f"\n=== RMSE RESULTS ===")
        for task, rmse in final_task_rmses.items():
            print(f"{task} RMSE: {rmse:.2f}")
        print(f"Overall RMSE: {overall_rmse:.2f}")
        
        # Update best model info if this model performs better
        global best_model_info
        if final_accuracy > best_model_info['accuracy'] or (final_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': final_accuracy,
                'predictions': (hh_pred, ethnicity_pred, religion_pred, tenure_pred, size_pred, rooms_pred),
                'lr': lr,
                'hidden_channels': hidden_channels,
                'convergence_data': convergence_data,
                'rmse': overall_rmse,
                'task_rmses': final_task_rmses
            })
    
    return best_epoch_loss, average_accuracy, final_accuracy, (hh_pred, ethnicity_pred, religion_pred, tenure_pred, size_pred, rooms_pred), convergence_data

# Run grid search over hyperparameters
total_start_time = time.time()

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, average_accuracy, final_accuracy, predictions, convergence_data = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            'average_accuracy': final_accuracy,
            'training_time': train_time_str,
            'rmse': best_model_info.get('rmse', None)
        })
        
        # Store timing results
        time_results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'training_time': train_time_str
        })

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}, Average Distribution Accuracy: {average_accuracy:.4f}, Final Distribution Accuracy: {final_accuracy:.4f}")
        print(f"Training time: {train_time_str}")

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total training time: {total_training_time_str}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("\nHyperparameter tuning results:")
print(results_df)

# Add best_model column to results_df using best_model_info
results_df['best_model'] = (
    (results_df['learning_rate'] == best_model_info['lr']) &
    (results_df['hidden_channels'] == best_model_info['hidden_channels'])
)

# Print best model information
print("\nBest Model Information:")
print(f"Learning Rate: {best_model_info['lr']}")
print(f"Hidden Channels: {best_model_info['hidden_channels']}")
print(f"Best Loss: {best_model_info['loss']:.4f}")
print(f"Best Distribution Accuracy (R²): {best_model_info['accuracy']:.4f}")

# Create output directory if it doesn't exist
# output_dir = os.path.join(current_dir, 'outputs')
output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

# Save best model information and results
# output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
# os.makedirs(output_dir, exist_ok=True)

# Save best model predictions
best_predictions = {
    'household_pred': best_model_info['predictions'][0].cpu().numpy(),
    'ethnicity_pred': best_model_info['predictions'][1].cpu().numpy(),
    'religion_pred': best_model_info['predictions'][2].cpu().numpy(),
    'tenure_pred': best_model_info['predictions'][3].cpu().numpy(),
    'size_pred': best_model_info['predictions'][4].cpu().numpy(),
    'rooms_pred': best_model_info['predictions'][5].cpu().numpy()
}

# Save hyperparameter results
results_df.to_csv(os.path.join(output_dir, 'generateHouseholds_results.csv'), index=False)

# Save convergence data from best model
if 'convergence_data' in best_model_info:
    convergence_df = pd.DataFrame(best_model_info['convergence_data'])
    convergence_df.to_csv(os.path.join(output_dir, 'convergence_data.csv'), index=False)

# Save performance data
performance_data = {
    'area_code': selected_area_code,
    'num_households': num_households,
    'training_time_seconds': total_training_time,
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'final_accuracy': best_model_info['accuracy'],
    'rmse': best_model_info.get('rmse', None)
}
performance_df = pd.DataFrame([performance_data])
performance_df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)

# Save best model configuration
best_config = {
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'loss': best_model_info['loss'],
    'accuracy': best_model_info['accuracy']
}

# Extract the best model's predictions for visualization
hh_pred, ethnicity_pred, religion_pred, tenure_pred, size_pred, rooms_pred = best_model_info['predictions']

# ============================================================================
# COMPOSITION-SIZE CONSISTENCY ENFORCEMENT
# ============================================================================
# Household composition DETERMINES valid size ranges:
# - 1PE, 1PA (single person): Size MUST be 1 (category 0)
# - 1FE, 1FM-0C, 1FC-0C (couples): Size MUST be 2 (category 1)
# - Others: Size 2+ is valid
#
# This fixes the issue where model may predict inconsistent composition-size pairs
# (e.g., 1PE household with size 4+, which is impossible)

COMPOSITION_SIZE_CONSTRAINTS = {
    # Single person households: size must be 1 (category index 0)
    0: 0,   # 1PE -> size 1
    1: 0,   # 1PA -> size 1
    # Couple households: size must be 2 (category index 1)
    2: 1,   # 1FE -> size 2
    3: 1,   # 1FM-0C -> size 2
    6: 1,   # 1FC-0C -> size 2
    # All other compositions: no constraint (size 2+ is valid)
}

def enforce_composition_size_consistency(hh_pred, size_pred):
    """
    Enforce that household size is consistent with composition.
    
    For example:
    - 1PE (single person elderly) MUST have size 1
    - 1FM-0C (married couple, no children) MUST have size 2
    
    Args:
        hh_pred: Household composition predictions [num_households]
        size_pred: Size category predictions [num_households]
    
    Returns:
        Corrected size predictions
    """
    size_pred_corrected = size_pred.clone()
    corrections_made = 0
    
    for i in range(len(hh_pred)):
        comp_idx = hh_pred[i].item()
        if comp_idx in COMPOSITION_SIZE_CONSTRAINTS:
            expected_size = COMPOSITION_SIZE_CONSTRAINTS[comp_idx]
            if size_pred_corrected[i].item() != expected_size:
                size_pred_corrected[i] = expected_size
                corrections_made += 1
    
    if corrections_made > 0:
        print(f"\n✓ Composition-size consistency: Corrected {corrections_made}/{len(hh_pred)} households")
        print(f"  (e.g., 1PE households forced to size=1, couples forced to size=2)")
    else:
        print(f"\n✓ Composition-size consistency: All {len(hh_pred)} households already consistent")
    
    return size_pred_corrected

print("\n=== ENFORCING COMPOSITION-SIZE CONSISTENCY ===")
size_pred_corrected = enforce_composition_size_consistency(hh_pred, size_pred)

# Create household tensor with attributes matching original format
# Expected format: [household_composition, ethnicity, religion, tenure, size, rooms] (6 columns)
# Where hh_composition is at index 0, ethnicity is at index 1, religion is at index 2, tenure is at index 3, size is at index 4, rooms is at index 5
household_nodes_tensor = torch.stack([
    hh_pred,              # Column 0: household composition
    ethnicity_pred,       # Column 1: ethnicity
    religion_pred,        # Column 2: religion
    tenure_pred,          # Column 3: tenure
    size_pred_corrected,  # Column 4: size (CORRECTED for consistency)
    rooms_pred            # Column 5: rooms
], dim=1)

# Validate composition-size distribution after corrections
print("\n=== FINAL HOUSEHOLD COMPOSITION-SIZE DISTRIBUTION ===")
comp_size_distribution = {}
for i in range(len(hh_pred)):
    comp_idx = hh_pred[i].item()
    size_idx = size_pred_corrected[i].item()
    key = (comp_idx, size_idx)
    comp_size_distribution[key] = comp_size_distribution.get(key, 0) + 1

# Show distribution for key compositions
print(f"{'Composition':<12} {'Size 1':>8} {'Size 2':>8} {'Size 3':>8} {'Size 4+':>8}")
print("-" * 48)
for comp_idx in range(min(15, len(hh_compositions))):
    comp_name = hh_compositions[comp_idx] if comp_idx < len(hh_compositions) else f"Comp{comp_idx}"
    row = [comp_name]
    for size_idx in range(4):
        count = comp_size_distribution.get((comp_idx, size_idx), 0)
        row.append(str(count) if count > 0 else "-")
    print(f"{row[0]:<12} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

# Save household tensor
household_tensor_path = os.path.join(output_dir, 'household_nodes.pt')
torch.save(household_nodes_tensor.cpu(), household_tensor_path)
print(f"\nBest model outputs saved to {output_dir}")

# Get the predicted household compositions, ethnicities, and religions
hh_comp_pred_indices = hh_pred.cpu().numpy()
ethnicity_pred_indices = ethnicity_pred.cpu().numpy()
religion_pred_indices = religion_pred.cpu().numpy()
tenure_pred_indices = tenure_pred.cpu().numpy()
size_pred_indices = size_pred.cpu().numpy()
rooms_pred_indices = rooms_pred.cpu().numpy()

# Convert indices to category names
hh_comp_pred_names = [hh_compositions[i] for i in hh_comp_pred_indices]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred_indices]
religion_pred_names = [religion_categories[i] for i in religion_pred_indices]
tenure_pred_names = [tenure_categories[i] for i in tenure_pred_indices]
size_pred_names = [size_categories[i] for i in size_pred_indices]
rooms_pred_names = [rooms_categories[i] for i in rooms_pred_indices]

# Save a human-readable table of generated households and print a sample
generated_households_df = pd.DataFrame({
    'household_id': np.arange(num_households),
    'composition': hh_comp_pred_names,
    'ethnicity': ethnicity_pred_names,
    'religion': religion_pred_names,
    'tenure': tenure_pred_names,
    'size': size_pred_names,
    'rooms': rooms_pred_names
})

gen_households_csv = os.path.join(output_dir, 'generated_households.csv')
generated_households_df.to_csv(gen_households_csv, index=False)
print(f"Saved generated households table to: {gen_households_csv}")

print("\nSample of generated households (first 20):")
print(generated_households_df.head(20).to_string(index=False))

print("\nHousehold composition distribution (counts):")
print(generated_households_df['composition'].value_counts().sort_index())

# Calculate counts of actual categories from the original data
hh_comp_actual = {}
for hh_comp in hh_compositions:
    hh_comp_actual[hh_comp] = hhcomp_df[hh_comp].iloc[0]

# Fix ethnicity and religion extraction based on column structure in the datasets
ethnicity_actual = {}
for eth in ethnicity_categories:
    ethnicity_actual[eth] = ethnicity_df[eth].iloc[0]

religion_actual = {}
for rel in religion_categories:
    religion_actual[rel] = religion_df[rel].iloc[0]

# Add actual data extraction for new attributes
tenure_actual = {}
for ten in tenure_categories:
    tenure_actual[ten] = tenure_df[ten].iloc[0]

size_actual = {}
for sz in size_categories:
    size_actual[sz] = hh_size_df[sz].iloc[0]

rooms_actual = {}
for rm in rooms_categories:
    rooms_actual[rm] = rooms_df[rm].iloc[0]

# Calculate counts of predicted categories
hh_comp_pred = dict(Counter(hh_comp_pred_names))
ethnicity_pred = dict(Counter(ethnicity_pred_names))
religion_pred = dict(Counter(religion_pred_names))
tenure_pred = dict(Counter(tenure_pred_names))
size_pred = dict(Counter(size_pred_names))
rooms_pred = dict(Counter(rooms_pred_names))

# Normalize the actual distributions to match the total number of households in predictions
# This ensures fair comparison of relative proportions
total_actual_ethnicity = sum(ethnicity_actual.values())
total_actual_religion = sum(religion_actual.values())
total_actual_tenure = sum(tenure_actual.values())
total_actual_size = sum(size_actual.values())
total_actual_rooms = sum(rooms_actual.values())
total_pred = num_households

if total_actual_ethnicity > 0:
    ethnicity_actual = {k: v * total_pred / total_actual_ethnicity for k, v in ethnicity_actual.items()}
if total_actual_religion > 0:
    religion_actual = {k: v * total_pred / total_actual_religion for k, v in religion_actual.items()}
if total_actual_tenure > 0:
    tenure_actual = {k: v * total_pred / total_actual_tenure for k, v in tenure_actual.items()}
if total_actual_size > 0:
    size_actual = {k: v * total_pred / total_actual_size for k, v in size_actual.items()}
if total_actual_rooms > 0:
    rooms_actual = {k: v * total_pred / total_actual_rooms for k, v in rooms_actual.items()}

# Create crosstable dataframes for visualization - only for combinations that actually exist in the data
# Reshape actual crosstables to match our format
hh_by_ethnicity_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Extract the actual counts from the real crosstable dataframes
for hh in hh_compositions:
    for eth in ethnicity_categories:
        col_name = f'{hh} {eth}'
        if col_name in hhcomp_by_ethnicity_df.columns:
            hh_by_ethnicity_actual_reshaped.loc[hh, eth] = hhcomp_by_ethnicity_df[col_name].iloc[0]
    
    for rel in religion_categories:
        col_name = f'{hh} {rel}'
        if col_name in hhcomp_by_religion_df.columns:
            hh_by_religion_actual_reshaped.loc[hh, rel] = hhcomp_by_religion_df[col_name].iloc[0]

# Create predicted crosstables from our predictions - only for real combinations
hh_by_ethnicity_pred = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_pred = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Fill the predicted crosstables based on our model predictions
for i in range(len(hh_comp_pred_names)):
    hh = hh_comp_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    
    hh_by_ethnicity_pred.loc[hh, eth] += 1
    hh_by_religion_pred.loc[hh, rel] += 1

# Plotly version of individual attribute distribution plots
def plotly_attribute_distributions(attribute_dicts, categories_dict, use_log=False, filter_zero_bars=False, max_cols=2, save_path=None):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for multiple attributes.
    Now includes a geo plot in the top right corner showing the selected area.
    
    Parameters:
    attribute_dicts - Dictionary of attribute names to (actual, predicted) count dictionaries
    categories_dict - Dictionary of attribute names to lists of categories
    use_log - Whether to use log scale for y-axis
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    max_cols - Maximum number of columns in the subplot grid
    save_path - Optional path to save the plot as HTML
    """
    attrs = list(attribute_dicts.keys())
    num_plots = len(attrs)
    
    # Add one extra column for the geo plot
    num_cols = min(num_plots, max_cols) + 1
    num_rows = math.ceil(num_plots / (num_cols - 1))  # Exclude geo column from calculation
    
    # Pre-calculate accuracy for each attribute
    accuracy_data = {}
    for attr_name in attrs:
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Calculate R² accuracy
        r2 = calculate_r2_accuracy(
            {cat: predicted_dict.get(cat, 0) for cat in categories},
            {cat: actual_dict.get(cat, 0) for cat in categories}
        )
        accuracy_data[attr_name] = r2 * 100.0
    
    # Create subplot specifications with geo plot spanning multiple rows
    specs = []
    for row in range(num_rows):
        row_specs = []
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Top right corner for geo plot
                row_specs.append({"type": "geo", "rowspan": min(num_rows, 3)})  # Span up to 3 rows
            elif row > 0 and row < min(num_rows, 3) and col == num_cols - 1:  # Skip cells for geo plot span
                row_specs.append(None)
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Create complete subplot titles with accuracy information, accounting for rowspan
    subplot_titles = []
    attr_idx = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Geo plot position (first row)
                subplot_titles.append("")  # No title for geo plot
            elif row > 0 and row < min(num_rows, 3) and col == num_cols - 1:  # Geo plot spanned rows
                subplot_titles.append(None)  # None for spanned cells
            elif attr_idx < len(attrs):  # Main plot positions
                attr_name = attrs[attr_idx]
                accuracy = accuracy_data[attr_name]
                subplot_titles.append(f"{attr_name} - Accuracy:{accuracy:.2f}%")
                attr_idx += 1
            else:  # Empty positions
                subplot_titles.append("")
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.20
    )
    
    # Add attribute distribution plots
    for idx, attr_name in enumerate(attrs):
        row = (idx // (num_cols - 1)) + 1  # Exclude geo column from calculation
        col = (idx % (num_cols - 1)) + 1   # Exclude geo column from calculation
        
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Convert to arrays
        actual_counts = np.array([actual_dict.get(cat, 0) for cat in categories])
        predicted_counts = np.array([predicted_dict.get(cat, 0) for cat in categories])
        
        # Optional log transform
        if use_log:
            actual_counts = np.log1p(actual_counts)
            predicted_counts = np.log1p(predicted_counts)
        
        # Add traces
        actual_trace = go.Bar(
            x=categories,
            y=actual_counts,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=categories,
            y=predicted_counts,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
    
    # Add geo plot in top right corner
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=num_cols)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=num_cols
        )
    
    # Update layout with increased height for larger geo plot
    fig.update_layout(
        height=350 * num_rows,  # Increased height to accommodate larger geo plot
        width=450 * num_cols,  # Fixed width
        title_text="Individual Attributes: Actual vs. Predicted",
        showlegend=True,
        plot_bgcolor="white",
        barmode='group',
        margin=dict(l=40, r=40, t=80, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column
            bgcolor='rgba(255,255,255,0.9)'
        )
    )

    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Household attributes plot saved to: {save_path}")
    
    # Display the plot
    # fig.show()

# Plotly version of crosstable plots - Paper-ready version
def plotly_crosstable_comparison(
    actual_dfs, 
    predicted_dfs, 
    titles, 
    show_keys=False, 
    num_cols=1, 
    filter_zero_bars=True,
    save_path=None
):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for crosstables.
    Paper-ready version with optimized sizing, fonts, and static image export.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    show_keys - Whether to show full category key combinations (True) or numeric indices (False)
    num_cols - Number of columns in the subplot grid
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    save_path - Optional path to save the plot (supports .html, .png, .pdf, .svg)
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # ============ PAPER-READY CONFIGURATION ============
    # Figure dimensions optimized for academic papers
    PAPER_WIDTH = 1400  # ~10 inches at 140 DPI - suitable for full-page width
    SUBPLOT_HEIGHT = 320  # Height per subplot row (increased for larger text)
    
    # Publication-quality font sizes - LARGER for paper readability
    FONT_CONFIG = {
        'title': 24,           # Subplot titles (increased)
        'axis_title': 18,      # Axis labels (increased)
        'tick_labels': 14,     # Tick labels (increased)
        'legend': 16,          # Legend text (increased)
        'annotation': 14       # Annotations (increased)
    }
    
    # Colors optimized for print (colorblind-friendly)
    COLORS = {
        'actual': '#D62728',      # Rich red
        'predicted': '#1F77B4',   # Deep blue
        'grid': '#E5E5E5',        # Light gray grid
        'axis': '#333333'         # Dark gray axis
    }
    # ===================================================
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    rmse_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays for bar charts
        actual_vals = []
        predicted_vals = []
        
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                threshold = 5
                should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                
                if not filter_zero_bars or not should_filter:
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
        
        # Calculate R² accuracy
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
        
        # Calculate RMSE
        rmse_val = calculate_rmse(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        rmse_data[idx] = rmse_val
    
    # Calculate number of rows
    crosstable_rows = (num_plots + num_cols - 1) // num_cols
    total_rows = crosstable_rows
    
    # Create subplot specifications
    specs = []
    for row in range(crosstable_rows):
        row_specs = []
        for col in range(num_cols):
            row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Create formatted titles with accuracy and RMSE
    all_titles = []
    main_plot_idx = 0
    for i in range(crosstable_rows):
        for j in range(num_cols):
            if main_plot_idx < len(titles):
                accuracy = accuracy_data[main_plot_idx]
                rmse = rmse_data[main_plot_idx]
                # Cleaner title format for papers
                all_titles.append(f"{titles[main_plot_idx]} (R²: {accuracy:.1f}%, RMSE: {rmse:.1f})")
                main_plot_idx += 1
            else:
                all_titles.append("")
    
    fig = make_subplots(
        rows=total_rows,
        cols=num_cols,
        subplot_titles=all_titles,
        specs=specs,
        vertical_spacing=0.10,  # Reduced vertical spacing between subplots
        horizontal_spacing=0.10
    )

    # Style subplot titles
    if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if ann.text:
                ann.font = dict(size=FONT_CONFIG['title'], color='#333333', family='Arial')
    
    for idx, crosstable_key in enumerate(keys_list):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten data
        actual_vals = []
        predicted_vals = []
        category_labels = []
        
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                threshold = 5
                should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                
                if not filter_zero_bars or not should_filter:
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
                    category_labels.append(f"{row_idx} {col_idx}")
        
        # For dense figures, sample bars to reduce visual clutter
        num_total = len(actual_vals)
        if num_total > 300:
            bar_step = 3  # Show every 3rd bar for very dense plots
        elif num_total > 150:
            bar_step = 2  # Show every 2nd bar for moderately dense plots
        else:
            bar_step = 1  # Show all bars for smaller plots
        
        # Sample the data if needed
        if bar_step > 1:
            sampled_indices = list(range(0, num_total, bar_step))
            actual_vals_plot = [actual_vals[i] for i in sampled_indices]
            predicted_vals_plot = [predicted_vals[i] for i in sampled_indices]
            continuous_positions = list(range(1, len(actual_vals_plot) + 1))
        else:
            actual_vals_plot = actual_vals
            predicted_vals_plot = predicted_vals
            continuous_positions = list(range(1, len(actual_vals) + 1))
        
        # Create bar traces with publication-quality styling
        actual_trace = go.Bar(
            x=continuous_positions,
            y=actual_vals_plot,
            name='Actual (Census)' if idx == 0 else None,
            marker=dict(
                color=COLORS['actual'],
                line=dict(width=0)  # Clean edges
            ),
            opacity=0.85,
            showlegend=idx == 0
        )
        
        predicted_trace = go.Bar(
            x=continuous_positions,
            y=predicted_vals_plot,
            name='Predicted (Model)' if idx == 0 else None,
            marker=dict(
                color=COLORS['predicted'],
                line=dict(width=0)
            ),
            opacity=0.85,
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Adaptive x-axis labeling - show more labels for better readability
        num_points = len(continuous_positions)
        # Calculate step size to show approximately 15-25 labels
        if num_points > 200:
            step_size = num_points // 20  # Show ~20 labels
        elif num_points > 100:
            step_size = num_points // 18  # Show ~18 labels
        elif num_points > 50:
            step_size = num_points // 15  # Show ~15 labels
        elif num_points > 25:
            step_size = max(2, num_points // 12)
        else:
            step_size = 1  # Show all labels for small plots

        step_size = max(1, step_size)  # Ensure at least 1
        visible_positions = [p for p in continuous_positions if (p - 1) % step_size == 0 or p == 1]
        # Don't add the last index as it overlaps with nearby labels
        visible_labels = [str(p) for p in visible_positions]

        fig.update_xaxes(
            tickmode='array',  # Force use of specified tick values only
            ticktext=visible_labels,
            tickvals=visible_positions,
            tickangle=0,  # Horizontal for cleaner look
            tickfont=dict(size=FONT_CONFIG['tick_labels'], family='Arial'),
            title_text="Category Index",
            title_font=dict(size=FONT_CONFIG['axis_title'], family='Arial'),
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['axis'],
            linewidth=1.5,
            mirror=True,
            row=row,
            col=col
        )
        
        fig.update_yaxes(
            title_text="Household Count",
            title_font=dict(size=FONT_CONFIG['axis_title'], family='Arial'),
            tickfont=dict(size=FONT_CONFIG['tick_labels'], family='Arial'),
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['axis'],
            linewidth=1.5,
            mirror=True,
            ticks="outside",
            tickcolor=COLORS['axis'],
            row=row,
            col=col
        )
    
    # Calculate total figure height - compact layout
    total_height = SUBPLOT_HEIGHT * crosstable_rows + 80  # Reduced extra space
    
    # Update layout with paper-ready settings
    fig.update_layout(
        width=PAPER_WIDTH,
        height=total_height,
        showlegend=True,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family='Arial', size=FONT_CONFIG['tick_labels'], color='#333333'),
        margin=dict(
            b=60,   # Reduced bottom margin
            t=70,   # Increased top margin to fit legend above plots
            l=80,   # Increased left margin for larger y-axis labels
            r=30    # Reduced right margin
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.000,  # Slightly lower, closer to top subplot
            xanchor="right",
            x=0.99,  # Aligned to the right (next to subplot title)
            bgcolor='rgba(255,255,255,0.0)',  # Transparent background
            borderwidth=0,  # No border
            font=dict(size=FONT_CONFIG['legend'] + 2, family='Arial')  # Larger font
        )
    )
    
    # Save the plot
    if save_path:
        # Save as HTML
        fig.write_html(save_path)
        print(f"Household crosstable comparison plot saved to: {save_path}")
        
        # Also save as high-resolution PNG for paper use
        png_path = save_path.replace('.html', '.png')
        try:
            fig.write_image(png_path, scale=2, width=PAPER_WIDTH, height=total_height)
            print(f"High-resolution PNG saved to: {png_path}")
        except Exception as e:
            print(f"Note: PNG export requires kaleido package. Install with: pip install kaleido")
        
        # Also save as PDF for vector graphics
        pdf_path = save_path.replace('.html', '.pdf')
        try:
            fig.write_image(pdf_path, width=PAPER_WIDTH, height=total_height)
            print(f"PDF saved to: {pdf_path}")
        except Exception as e:
            print(f"Note: PDF export requires kaleido package. Install with: pip install kaleido")
    
    # Display the plot
    fig.show()

# Plotly version of radar crosstable comparison
def plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles, save_path=None):
    """
    Creates radar chart subplots comparing actual vs. predicted distributions for crosstables.
    Uses numeric indices instead of category labels and shows aggregated actual vs predicted lines.
    Now includes a geo plot showing the selected area.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    save_path - Optional path to save the plot as HTML
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays with new ordering (household compositions first, then ethnicity/religion)
        actual_vals = actual_df.values.flatten()  # No transpose needed - using original order
        predicted_vals = predicted_df.values.flatten()  # No transpose needed - using original order
        
        # Calculate R² accuracy using the same method as training
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
    
    # Set to two columns: one for radar charts, one for geo plot
    num_cols = 2
    num_rows = num_plots
    
    # Create subplot specifications
    specs = []
    for row in range(num_rows):
        row_specs = [{'type': 'polar'}]  # Radar chart column
        if row == 0:  # Only add geo to first row
            row_specs.append({'type': 'geo'})  # Geo plot column
        else:
            row_specs.append(None)  # Empty subplot for other rows
        specs.append(row_specs)
    
    # Create complete titles with accuracy information
    extended_titles = []
    for i, title in enumerate(titles):
        accuracy = accuracy_data[i]
        extended_titles.append(f"{title} - Accuracy:{accuracy:.2f}%")
        
        if i == 0:  # Add geo title only for first row
            extended_titles.append("")  # Removed geo plot title
        else:
            extended_titles.append("")  # Empty title for other rows
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=extended_titles,
        specs=specs,
        vertical_spacing=0.1,  # Increased vertical spacing between subplots
        horizontal_spacing=0.2,
        column_widths=[0.7, 0.3]  # 70% radar charts, 30% geo plot
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = idx + 1
        col = 1  # Always put radar charts in first column
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays with new ordering (household compositions first, then ethnicity/religion)
        actual_vals = actual_df.values.flatten()  # No transpose needed - using original order
        predicted_vals = predicted_df.values.flatten()  # No transpose needed - using original order
        
        # Create numeric indices for the categories
        num_points = len(actual_vals)
        
        # Determine step size for labels based on number of points
        if num_points > 400:
            step_size = 16
        elif num_points > 200:
            step_size = 8
        elif num_points > 40:
            step_size = 4
        elif num_points > 30:
            step_size = 3
        elif num_points > 20:
            step_size = 2
        else:
            step_size = 1
            
        # Create labels with appropriate step size
        theta = []
        for i in range(num_points):
            if i % step_size == 0:
                theta.append(f"{i+1}")
            else:
                theta.append("")
        
        # Add the first value again to close the polygon
        actual_vals = np.append(actual_vals, actual_vals[0])
        predicted_vals = np.append(predicted_vals, predicted_vals[0])
        theta = theta + [theta[0]]
        
        # Get pre-calculated accuracy
        r2_accuracy = accuracy_data[idx]
        
        # Create traces
        actual_trace = go.Scatterpolar(
            r=actual_vals,
            theta=theta,
            name='Actual' if idx == 0 else None,
            line=dict(color='red', width=2),
            showlegend=idx == 0
        )
        
        predicted_trace = go.Scatterpolar(
            r=predicted_vals,
            theta=theta,
            name=f'Predicted (Acc: {r2_accuracy:.1f}%)' if idx == 0 else None,
            line=dict(color='blue', width=2),
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
    
    # Add geo plot in top right (first row, second column)
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=2)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=2
        )
    
    # Update layout with fixed dimensions
    fig.update_layout(
        height=450 * num_rows,  # Slightly increased height to accommodate title spacing
        width=1200,  # Fixed width
        title_text="Radar Chart Comparison: Actual vs. Predicted",
        title_font_size=18,  # Main title size
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column (for 70/30 layout)
            bgcolor='rgba(255,255,255,0.9)'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(
                    actual_df.values.max(),
                    predicted_df.values.max()
                )]
            )
        ),
        margin=dict(t=120, b=80, l=100, r=100)  # Increased top margin for titles
    )
    
    # Update polar axes for each subplot
    for i in range(1, num_rows + 1):
        fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced radial axis font size
                ),
                angularaxis=dict(
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced angular axis font size
                    rotation=90,
                    direction="clockwise"
                )
            ),
            row=i,
            col=1  # Only apply to radar chart column
        )
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Household radar chart comparison plot saved to: {save_path}")
    
    # Display the plot
    fig.show()

# Create Plotly individual attribute plots
attribute_dicts = {
    'Composition': (hh_comp_actual, hh_comp_pred),
    'Ethnicity': (ethnicity_actual, ethnicity_pred),
    'Religion': (religion_actual, religion_pred),
    'Tenure': (tenure_actual, tenure_pred),
    'Size': (size_actual, size_pred),
    'Rooms': (rooms_actual, rooms_pred)
}

categories_dict = {
    'Composition': hh_compositions,
    'Ethnicity': ethnicity_categories,
    'Religion': religion_categories,
    'Tenure': tenure_categories,
    'Size': size_categories,
    'Rooms': rooms_categories
}

household_attributes_save_path = os.path.join(output_dir, 'household_attributes_comparison.html')
plotly_attribute_distributions(attribute_dicts, categories_dict, filter_zero_bars=True, save_path=household_attributes_save_path)

# Create 3-way crosstable for tenure x size x rooms (similar to generateIndividuals.py)
tenure_size_rooms_actual = pd.DataFrame(0, index=range(len(tenure_categories) * len(size_categories) * len(rooms_categories)), columns=['combination', 'actual_count'])
tenure_size_rooms_pred = pd.DataFrame(0, index=range(len(tenure_categories) * len(size_categories) * len(rooms_categories)), columns=['combination', 'predicted_count'])

# Fill actual 3-way crosstable - using correct order: tenure -> size -> rooms
idx = 0
for ten in tenure_categories:  # tenure first (matches get_target_tensors_3way ordering)
    for sz in size_categories:  # size second  
        for rm in rooms_categories:  # rooms third
            col_name = f'{ten} {sz} {rm}'
            actual_count = tenure_by_size_by_rooms_df[col_name].iloc[0] if col_name in tenure_by_size_by_rooms_df.columns else 0
            tenure_size_rooms_actual.loc[idx] = [f'{ten} {sz} {rm}', actual_count]
            idx += 1

# Fill predicted 3-way crosstable  
idx = 0
tenure_size_rooms_pred_counts = {}
for i in range(len(tenure_pred_names)):
    ten = tenure_pred_names[i]
    sz = size_pred_names[i]
    rm = rooms_pred_names[i]
    combo = f'{ten} {sz} {rm}'
    tenure_size_rooms_pred_counts[combo] = tenure_size_rooms_pred_counts.get(combo, 0) + 1

for ten in tenure_categories:  # tenure first (matches get_target_tensors_3way ordering)
    for sz in size_categories:  # size second  
        for rm in rooms_categories:  # rooms third
            combo = f'{ten} {sz} {rm}'
            pred_count = tenure_size_rooms_pred_counts.get(combo, 0)
            tenure_size_rooms_pred.loc[idx] = [combo, pred_count]
            idx += 1

# Create Plotly crosstable plots
actual_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_actual_reshaped,
    'Household_by_Religion': hh_by_religion_actual_reshaped,
    'Tenure_Size_Rooms': tenure_size_rooms_actual.set_index('combination')['actual_count'].to_frame().T
}

predicted_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_pred,
    'Household_by_Religion': hh_by_religion_pred,
    'Tenure_Size_Rooms': tenure_size_rooms_pred.set_index('combination')['predicted_count'].to_frame().T
}

titles = [
    'Household Comp x Ethnicity',
    'Household Comp x Religion',
    'Tenure x Size x Rooms'
]

household_crosstable_save_path = os.path.join(output_dir, 'household_crosstable_comparison.html')
plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True, save_path=household_crosstable_save_path)

# Create Plotly radar crosstable plots
household_radar_save_path = os.path.join(output_dir, 'household_radar_crosstable_comparison.html')
# plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles, save_path=household_radar_save_path)