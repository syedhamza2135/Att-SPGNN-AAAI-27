import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import argparse
import plotly.io as pio

# ============================================================================
# PAPER-READY PLOTTING CONFIGURATION
# Consistent styling with generateIndividuals.py, generateHouseholds.py, and assignHouseholds.py
# ============================================================================
PLOT_CONFIG = {
    'figure_width': 1600,
    'figure_height': 900,
    'font_family': 'Arial',
    'fonts': {
        'title': 26,           # Main title (increased)
        'subplot_title': 22,   # Subplot titles (increased)
        'axis_title': 18,      # Axis labels (increased)
        'tick_labels': 14,     # Tick labels (increased)
        'legend': 14,          # Legend text (increased)
        'annotation': 13       # Annotations (increased)
    },
    'colors': {
        'individuals': ['#1F77B4', '#4A90D9', '#7EB3E8', '#A8CCF0', '#D1E5F7'],  # Blues
        'households': ['#D62728', '#E85A5B', '#F08080', '#F5A6A6', '#FACBCB'],   # Reds
        'grid': '#E5E5E5',
        'axis': '#333333'
    },
    'dpi': 300
}

# Extended color palettes for multiple areas
AREA_COLORS_INDIVIDUALS = [
    '#1F77B4', '#2E86C1', '#3498DB', '#5DADE2', '#85C1E9',
    '#1A5276', '#21618C', '#2874A6', '#2980B9', '#5499C7'
]

AREA_COLORS_HOUSEHOLDS = [
    '#D62728', '#E74C3C', '#EC7063', '#F1948A', '#F5B7B1',
    '#922B21', '#A93226', '#C0392B', '#CD6155', '#D98880'
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot convergence and performance data for all areas')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Output directory containing area folders (default: ../outputs)')
    parser.add_argument('--plot_type', type=str, choices=['individuals', 'households', 'both'], 
                       default='both', help='Type of plots to generate (default: both)')
    return parser.parse_args()

def find_area_folders(output_dir):
    """
    Find all individuals_{area_code} and households_{area_code} folders
    Returns dictionaries with area_code as key and folder path as value
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(current_dir, output_dir)
    
    individuals_folders = {}
    households_folders = {}
    
    if os.path.exists(base_output_dir):
        # Find individuals folders
        individual_pattern = os.path.join(base_output_dir, 'individuals_*')
        for folder in glob.glob(individual_pattern):
            if os.path.isdir(folder):
                area_code = os.path.basename(folder).replace('individuals_', '')
                individuals_folders[area_code] = folder
        
        # Find households folders
        household_pattern = os.path.join(base_output_dir, 'households_*')
        for folder in glob.glob(household_pattern):
            if os.path.isdir(folder):
                area_code = os.path.basename(folder).replace('households_', '')
                households_folders[area_code] = folder
    
    return individuals_folders, households_folders

def load_convergence_data(folder_path):
    """Load convergence data from a folder if it exists"""
    convergence_file = os.path.join(folder_path, 'convergence_data.csv')
    if os.path.exists(convergence_file):
        return pd.read_csv(convergence_file)
    return None

def load_performance_data(folder_path):
    """Load performance data from a folder if it exists"""
    performance_file = os.path.join(folder_path, 'performance_data.csv')
    if os.path.exists(performance_file):
        return pd.read_csv(performance_file)
    return None

def create_convergence_plots(individuals_folders, households_folders, plot_type='both'):
    """
    Create paper-ready convergence plots for loss and accuracy.
    Single compact 2x2 layout with individuals (top) and households (bottom).
    """
    
    # Line styles for better differentiation
    LINE_STYLES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
    
    # Always create 2x2 layout for compact single figure
    subplot_titles = [
        '(a) Individuals - Loss Convergence',
        '(b) Individuals - Accuracy Convergence', 
        '(c) Households - Loss Convergence',
        '(d) Households - Accuracy Convergence'
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )
    
    # Style subplot titles - larger for visibility
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=PLOT_CONFIG['fonts']['subplot_title'], 
                                  family=PLOT_CONFIG['font_family'],
                                  color='#333333')
    
    # Plot individuals convergence (top row)
    if individuals_folders:
        for idx, (area_code, folder_path) in enumerate(sorted(individuals_folders.items())):
            convergence_data = load_convergence_data(folder_path)
            if convergence_data is not None:
                color = AREA_COLORS_INDIVIDUALS[idx % len(AREA_COLORS_INDIVIDUALS)]
                line_style = LINE_STYLES[idx % len(LINE_STYLES)]
                
                epochs = convergence_data['epochs']
                losses = convergence_data['losses']
                
                # Subsample for cleaner lines
                step = max(1, len(epochs) // 40)
                sampled_epochs = list(epochs[::step])
                sampled_losses = list(losses[::step])
                
                # Always include last point
                if len(epochs) > 0 and epochs.iloc[-1] not in sampled_epochs:
                    sampled_epochs.append(epochs.iloc[-1])
                    sampled_losses.append(losses.iloc[-1])
                
                # Shorten area code for legend
                short_code = area_code.replace('E0200', '')
                
                # Loss plot (row 1, col 1)
                fig.add_trace(
                    go.Scatter(
                        x=sampled_epochs,
                        y=sampled_losses,
                        mode='lines+markers',
                        name=f'{short_code}',
                        line=dict(color=color, width=2.5, dash=line_style),
                        marker=dict(size=6, symbol='circle', color=color),
                        opacity=0.9,
                        showlegend=True,
                        legendgroup='individuals',
                        legendgrouptitle_text='Individuals',
                        hovertemplate=f'<b>{area_code}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Accuracy plot (row 1, col 2)
                valid_accuracy_data = convergence_data.dropna(subset=['accuracies'])
                if not valid_accuracy_data.empty:
                    acc_epochs = valid_accuracy_data['epochs']
                    accuracies = valid_accuracy_data['accuracies']
                    
                    sampled_acc_epochs = list(acc_epochs[::step])
                    sampled_accuracies = list(accuracies[::step])
                    
                    if len(acc_epochs) > 0 and acc_epochs.iloc[-1] not in sampled_acc_epochs:
                        sampled_acc_epochs.append(acc_epochs.iloc[-1])
                        sampled_accuracies.append(accuracies.iloc[-1])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sampled_acc_epochs,
                            y=sampled_accuracies,
                            mode='lines+markers',
                            name=f'{short_code}',
                            line=dict(color=color, width=2.5, dash=line_style),
                            marker=dict(size=6, symbol='circle', color=color),
                            opacity=0.9,
                            showlegend=False,
                            legendgroup='individuals',
                            hovertemplate=f'<b>{area_code}</b><br>Epoch: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
                        ),
                        row=1, col=2
                    )
    
    # Plot households convergence (bottom row)
    if households_folders:
        for idx, (area_code, folder_path) in enumerate(sorted(households_folders.items())):
            convergence_data = load_convergence_data(folder_path)
            if convergence_data is not None:
                color = AREA_COLORS_HOUSEHOLDS[idx % len(AREA_COLORS_HOUSEHOLDS)]
                line_style = LINE_STYLES[idx % len(LINE_STYLES)]
                
                epochs = convergence_data['epochs']
                losses = convergence_data['losses']
                
                step = max(1, len(epochs) // 40)
                sampled_epochs = list(epochs[::step])
                sampled_losses = list(losses[::step])
                
                if len(epochs) > 0 and epochs.iloc[-1] not in sampled_epochs:
                    sampled_epochs.append(epochs.iloc[-1])
                    sampled_losses.append(losses.iloc[-1])
                
                short_code = area_code.replace('E0200', '')
                
                # Loss plot (row 2, col 1)
                fig.add_trace(
                    go.Scatter(
                        x=sampled_epochs,
                        y=sampled_losses,
                        mode='lines+markers',
                        name=f'{short_code}',
                        line=dict(color=color, width=2.5, dash=line_style),
                        marker=dict(size=6, symbol='circle', color=color),
                        opacity=0.9,
                        showlegend=True,
                        legendgroup='households',
                        legendgrouptitle_text='Households',
                        hovertemplate=f'<b>{area_code}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Accuracy plot (row 2, col 2)
                valid_accuracy_data = convergence_data.dropna(subset=['accuracies'])
                if not valid_accuracy_data.empty:
                    acc_epochs = valid_accuracy_data['epochs']
                    accuracies = valid_accuracy_data['accuracies']
                    
                    sampled_acc_epochs = list(acc_epochs[::step])
                    sampled_accuracies = list(accuracies[::step])
                    
                    if len(acc_epochs) > 0 and acc_epochs.iloc[-1] not in sampled_acc_epochs:
                        sampled_acc_epochs.append(acc_epochs.iloc[-1])
                        sampled_accuracies.append(accuracies.iloc[-1])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sampled_acc_epochs,
                            y=sampled_accuracies,
                            mode='lines+markers',
                            name=f'{short_code}',
                            line=dict(color=color, width=2.5, dash=line_style),
                            marker=dict(size=6, symbol='circle', color=color),
                            opacity=0.9,
                            showlegend=False,
                            legendgroup='households',
                            hovertemplate=f'<b>{area_code}</b><br>Epoch: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
                        ),
                        row=2, col=2
                    )
    
    # Update axes with paper-ready styling
    axis_style = dict(
        showgrid=True,
        gridcolor=PLOT_CONFIG['colors']['grid'],
        gridwidth=1,
        showline=True,
        linewidth=1.5,
        linecolor=PLOT_CONFIG['colors']['axis'],
        mirror=True,
        tickfont=dict(size=PLOT_CONFIG['fonts']['tick_labels'], family=PLOT_CONFIG['font_family'])
    )
    
    # Row 1 (Individuals)
    fig.update_xaxes(title_text="Epoch", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=1, col=1, **axis_style)
    fig.update_xaxes(title_text="Epoch", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=1, col=2, **axis_style)
    fig.update_yaxes(title_text="Loss", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=1, col=1, **axis_style)
    fig.update_yaxes(title_text="Accuracy (%)", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=1, col=2, **axis_style)
    
    # Row 2 (Households)
    fig.update_xaxes(title_text="Epoch", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=2, col=1, **axis_style)
    fig.update_xaxes(title_text="Epoch", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=2, col=2, **axis_style)
    fig.update_yaxes(title_text="Loss", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=2, col=1, **axis_style)
    fig.update_yaxes(title_text="Accuracy (%)", title_font=dict(size=PLOT_CONFIG['fonts']['axis_title']), 
                     row=2, col=2, **axis_style)
    
    # Calculate required height based on number of legend items
    num_ind = len(individuals_folders) if individuals_folders else 0
    num_hh = len(households_folders) if households_folders else 0
    total_legend_items = num_ind + num_hh + 2  # +2 for group titles
    
    # Ensure figure is tall enough for all legend items (approx 25px per item)
    min_height_for_legend = total_legend_items * 28 + 150
    fig_height = max(900, min_height_for_legend)
    
    # Calculate gap between legend groups to align households with bottom row
    # Each legend item is approximately 22px, individuals section should span top half
    individuals_legend_height = (num_ind + 1) * 22  # +1 for group title
    # Gap should push households legend to align with row 2 (bottom half of plot)
    # Row 2 starts at approximately y=0.45 in the figure
    # Increased multiplier from 0.35 to 0.45 for more vertical separation
    legend_gap = max(95, int(fig_height * 0.52) - individuals_legend_height)
    
    # Update layout - LARGER plot with legend on the right, NO SCROLLING
    fig.update_layout(
        height=fig_height,
        width=1700,  # Slightly wider for legend space
        font=dict(family=PLOT_CONFIG['font_family'], size=PLOT_CONFIG['fonts']['tick_labels']),
        showlegend=True,
        legend=dict(
            orientation="v",  # Vertical legend
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,  # Position to the right of plots
            font=dict(size=PLOT_CONFIG['fonts']['legend'] - 1),  # Slightly smaller to fit all
            bgcolor='rgba(255,255,255,0.0)',  # Transparent background
            borderwidth=0,
            tracegroupgap=legend_gap,  # Dynamic gap to align households with bottom row
            groupclick="toggleitem",
            itemsizing="constant",
            title=dict(font=dict(size=PLOT_CONFIG['fonts']['legend'])),
            itemwidth=30
        ),
        margin=dict(l=90, r=200, t=90, b=80),  # More right margin for legend
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_performance_plots(individuals_folders, households_folders, plot_type='both'):
    """Create performance plots showing relationship between population and training efficiency"""
    
    # Determine subplot configuration based on plot type
    if plot_type == 'individuals':
        rows, cols = 2, 1
        subplot_titles = [
            'Individuals - Population vs Avg Epoch Time',
            'Individuals - Population vs Total Training Time'
        ]
    elif plot_type == 'households':
        rows, cols = 2, 1
        subplot_titles = [
            'Households - Population vs Avg Epoch Time',
            'Households - Population vs Total Training Time'
        ]
    else:  # both
        rows, cols = 2, 2
        subplot_titles = [
            'Individuals - Population vs Avg Epoch Time',
            'Households - Population vs Avg Epoch Time',
            'Individuals - Population vs Total Training Time', 
            'Households - Population vs Total Training Time'
        ]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    # Collect individuals performance data
    individuals_data = []
    for area_code, folder_path in individuals_folders.items():
        performance_data = load_performance_data(folder_path)
        convergence_data = load_convergence_data(folder_path)
        
        if performance_data is not None and convergence_data is not None:
            # Calculate average epoch time if available
            avg_epoch_time = convergence_data['epoch_time_seconds'].mean() if 'epoch_time_seconds' in convergence_data.columns else None
            
            individuals_data.append({
                'area_code': area_code,
                'population': performance_data['num_persons'].iloc[0],
                'total_time_seconds': performance_data['training_time_seconds'].iloc[0],
                'avg_epoch_time': avg_epoch_time,
                'accuracy': performance_data['final_accuracy'].iloc[0],
                'num_epochs': len(convergence_data) if convergence_data is not None else 0
            })
    
    # Collect households performance data
    households_data = []
    for area_code, folder_path in households_folders.items():
        performance_data = load_performance_data(folder_path)
        convergence_data = load_convergence_data(folder_path)
        
        if performance_data is not None and convergence_data is not None:
            # Calculate average epoch time if available
            avg_epoch_time = convergence_data['epoch_time_seconds'].mean() if 'epoch_time_seconds' in convergence_data.columns else None
            
            households_data.append({
                'area_code': area_code,
                'population': performance_data['num_households'].iloc[0],
                'total_time_seconds': performance_data['training_time_seconds'].iloc[0],
                'avg_epoch_time': avg_epoch_time,
                'accuracy': performance_data['final_accuracy'].iloc[0],
                'num_epochs': len(convergence_data) if convergence_data is not None else 0
            })
    
    # Plot individuals - Population vs Average Epoch Time
    if plot_type in ['individuals', 'both'] and individuals_data:
        individuals_df = pd.DataFrame(individuals_data)
        individuals_df = individuals_df.dropna(subset=['avg_epoch_time'])  # Remove rows without epoch time data
        
        if not individuals_df.empty:
            # Calculate efficiency metrics
            individuals_df['persons_per_second'] = individuals_df['population'] / individuals_df['avg_epoch_time']
            
            fig.add_trace(
                go.Scatter(
                    x=individuals_df['population'],
                    y=individuals_df['avg_epoch_time'],
                    mode='markers+text',
                    name='Individuals (Epoch Time)',
                    text=individuals_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=individuals_df['accuracy'],
                        colorscale='Viridis',
                        colorbar=dict(
                            title="Accuracy",
                            x=0.48,
                            len=0.4,
                            y=0.75
                        ),
                        showscale=True,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Avg Epoch Time: %{y:.2f}s<br>' +
                                'Efficiency: %{customdata:.1f} persons/sec<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=individuals_df['persons_per_second']
                ),
                row=1, col=(1 if plot_type == 'individuals' else 1)
            )
            
            # Add trend line for epoch time
            if len(individuals_df) > 1:
                z = np.polyfit(individuals_df['population'], individuals_df['avg_epoch_time'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(individuals_df['population'].min(), individuals_df['population'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name=f'Trend (slope: {z[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=1, col=(1 if plot_type == 'individuals' else 1)
                )
            
            # Plot individuals - Population vs Total Training Time
            fig.add_trace(
                go.Scatter(
                    x=individuals_df['population'],
                    y=individuals_df['total_time_seconds'],
                    mode='markers+text',
                    name='Individuals (Total Time)',
                    text=individuals_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=individuals_df['accuracy'],
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Total Time: %{y:.1f}s<br>' +
                                'Time per Person: %{customdata:.3f}s<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=individuals_df['total_time_seconds'] / individuals_df['population']
                ),
                row=2, col=(1 if plot_type == 'individuals' else 1)
            )
            
            # Add trend line for total time
            if len(individuals_df) > 1:
                z2 = np.polyfit(individuals_df['population'], individuals_df['total_time_seconds'], 1)
                p2 = np.poly1d(z2)
                y_trend2 = p2(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend2,
                        mode='lines',
                        name=f'Trend (slope: {z2[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=2, col=(1 if plot_type == 'individuals' else 1)
                )
    
        # Plot households - Population vs Average Epoch Time  
    if plot_type in ['households', 'both'] and households_data:
        households_df = pd.DataFrame(households_data)
        households_df = households_df.dropna(subset=['avg_epoch_time'])  # Remove rows without epoch time data
        
        if not households_df.empty:
            # Calculate efficiency metrics
            households_df['households_per_second'] = households_df['population'] / households_df['avg_epoch_time']
            
            fig.add_trace(
                go.Scatter(
                    x=households_df['population'],
                    y=households_df['avg_epoch_time'],
                    mode='markers+text',
                    name='Households (Epoch Time)',
                    text=households_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=households_df['accuracy'],
                        colorscale='Plasma',
                        colorbar=dict(
                            title="Accuracy",
                            x=1.02,
                            len=0.4,
                            y=0.75
                        ),
                        showscale=True,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Avg Epoch Time: %{y:.2f}s<br>' +
                                'Efficiency: %{customdata:.1f} households/sec<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=households_df['households_per_second']
                ),
                row=1, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add trend line for epoch time
            if len(households_df) > 1:
                z3 = np.polyfit(households_df['population'], households_df['avg_epoch_time'], 1)
                p3 = np.poly1d(z3)
                x_trend3 = np.linspace(households_df['population'].min(), households_df['population'].max(), 100)
                y_trend3 = p3(x_trend3)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend3,
                        y=y_trend3,
                        mode='lines',
                        name=f'Trend (slope: {z3[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=1, col=(1 if plot_type == 'households' else 2)
                )
            
            # Plot households - Population vs Total Training Time
            fig.add_trace(
                go.Scatter(
                    x=households_df['population'],
                    y=households_df['total_time_seconds'],
                    mode='markers+text',
                    name='Households (Total Time)',
                    text=households_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=households_df['accuracy'],
                        colorscale='Plasma',
                        showscale=False,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Total Time: %{y:.1f}s<br>' +
                                'Time per Household: %{customdata:.3f}s<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=households_df['total_time_seconds'] / households_df['population']
                ),
                row=2, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add trend line for total time
            if len(households_df) > 1:
                z4 = np.polyfit(households_df['population'], households_df['total_time_seconds'], 1)
                p4 = np.poly1d(z4)
                y_trend4 = p4(x_trend3)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend3,
                        y=y_trend4,
                        mode='lines',
                        name=f'Trend (slope: {z4[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=2, col=(1 if plot_type == 'households' else 2)
                )
    
    # Update axes labels based on plot type
    if plot_type == 'individuals':
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Individuals", row=2, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
    elif plot_type == 'households':
        fig.update_xaxes(title_text="Number of Households", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=2, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
    else:  # both
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=1, col=2)
        fig.update_xaxes(title_text="Number of Individuals", row=2, col=1)
        fig.update_xaxes(title_text="Number of Households", row=2, col=2)
        
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=2)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=2)
    
    # Update layout based on plot type
    width = 700 if plot_type in ['individuals', 'households'] else 1400
    title_suffix = plot_type.title() if plot_type != 'both' else 'All Types'
    
    fig.update_layout(
        height=800,
        width=width,
        title_text=f"Training Performance vs Population Size Analysis - {title_suffix}",
        title_font_size=18,
        showlegend=False,  # Too many traces, legend would be cluttered
        margin=dict(l=80, r=150, t=100, b=80)
    )
    
    return fig

def create_training_progress_plots(individuals_folders, households_folders, plot_type='both'):
    """Create training progress histograms: Population/Households vs Training Time"""
    
    # Determine subplot configuration based on plot type
    if plot_type == 'individuals':
        rows, cols = 1, 1
        subplot_titles = ['Individuals - Population vs Total Training Time']
    elif plot_type == 'households':
        rows, cols = 1, 1
        subplot_titles = ['Households - Number vs Total Training Time']
    else:  # both
        rows, cols = 1, 2
        subplot_titles = [
            'Individuals - Population vs Total Training Time',
            'Households - Number vs Total Training Time'
        ]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15
    )
    
    # Collect individuals data for histogram
    if plot_type in ['individuals', 'both'] and individuals_folders:
        populations = []
        training_times = []
        area_codes = []
        
        for area_code, folder_path in individuals_folders.items():
            performance_data = load_performance_data(folder_path)
            
            if performance_data is not None:
                populations.append(performance_data['num_persons'].iloc[0])
                training_times.append(performance_data['training_time_seconds'].iloc[0])
                area_codes.append(area_code)
        
        if populations:
            fig.add_trace(
                go.Bar(
                    x=populations,
                    y=training_times,
                    orientation='v',
                    name='Individuals',
                    width=[max(populations) * 0.002] * len(populations),  # Make bars thinner
                    marker=dict(color='red', opacity=0.7),
                    hovertemplate='<b>%{customdata}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Training Time: %{y:.1f}s<br>' +
                                '<extra></extra>',
                    customdata=area_codes,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add text annotations manually for better control
            for i, (pop, time, code) in enumerate(zip(populations, training_times, area_codes)):
                fig.add_annotation(
                    x=pop,
                    y=time + max(training_times) * 0.15,  # Position further above bar
                    text=code,
                    textangle=90,
                    font=dict(size=8, color='black'),
                    showarrow=False,
                    row=1, col=1
                )
    
    # Collect households data for histogram
    if plot_type in ['households', 'both'] and households_folders:
        num_households = []
        training_times = []
        area_codes = []
        
        for area_code, folder_path in households_folders.items():
            performance_data = load_performance_data(folder_path)
            
            if performance_data is not None:
                num_households.append(performance_data['num_households'].iloc[0])
                training_times.append(performance_data['training_time_seconds'].iloc[0])
                area_codes.append(area_code)
        
        if num_households:
            fig.add_trace(
                go.Bar(
                    x=num_households,
                    y=training_times,
                    orientation='v',
                    name='Households',
                    width=[max(num_households) * 0.002] * len(num_households),  # Make bars thinner
                    marker=dict(color='red', opacity=0.7),
                    hovertemplate='<b>%{customdata}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Training Time: %{y:.1f}s<br>' +
                                '<extra></extra>',
                    customdata=area_codes,
                    showlegend=False
                ),
                row=1, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add text annotations manually for better control
            for i, (hh, time, code) in enumerate(zip(num_households, training_times, area_codes)):
                fig.add_annotation(
                    x=hh,
                    y=time + max(training_times) * 0.15,  # Position further above bar
                    text=code,
                    textangle=90,
                    font=dict(size=8, color='black'),
                    showarrow=False,
                    row=1, col=(1 if plot_type == 'households' else 2)
                )
    
    # Update axes based on plot type
    if plot_type == 'individuals':
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
    elif plot_type == 'households':
        fig.update_xaxes(title_text="Number of Households", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
    else:  # both
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=1, col=2)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=2)
    

    
    # Update layout based on plot type
    width = 2300 if plot_type in ['individuals', 'households'] else 2600
    title_suffix = plot_type.title() if plot_type != 'both' else 'All Areas'
    
    fig.update_layout(
        height=600,
        width=width,
        title_text=f"Population vs Training Time Analysis - {title_suffix}",
        title_font_size=18,
        showlegend=False,
        margin=dict(l=80, r=150, t=100, b=80),
        plot_bgcolor="white"
    )
    
    # Style the axes
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
    
    return fig

def main():
    args = parse_arguments()
    
    print("=" * 60)
    print("CONVERGENCE PLOT GENERATOR - Paper-Ready Version")
    print("=" * 60)
    
    print("\nSearching for existing output folders...")
    individuals_folders, households_folders = find_area_folders(args.output_dir)
    
    print(f"\nFound {len(individuals_folders)} individuals folders:")
    for area_code in sorted(individuals_folders.keys()):
        print(f"  - individuals_{area_code}")
    
    print(f"\nFound {len(households_folders)} households folders:")
    for area_code in sorted(households_folders.keys()):
        print(f"  - households_{area_code}")
    
    if not individuals_folders and not households_folders:
        print("\nNo output folders found. Please run generateIndividuals.py and/or generateHouseholds.py first.")
        return
    
    # Create convergence plots
    if individuals_folders or households_folders:
        print(f"\nCreating paper-ready convergence plots...")
        convergence_fig = create_convergence_plots(individuals_folders, households_folders, args.plot_type)
        
        # Save convergence plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(current_dir, args.output_dir)
        os.makedirs(output_base, exist_ok=True)
        
        # Save as HTML (interactive)
        html_path = os.path.join(output_base, 'convergence_plots.html')
        convergence_fig.write_html(html_path)
        print(f"  HTML saved to: {html_path}")
        
        # Save as PNG (high-resolution for paper)
        try:
            png_path = os.path.join(output_base, 'convergence_plots.png')
            # Use figure's actual dimensions
            fig_width = convergence_fig.layout.width or 1700
            fig_height = convergence_fig.layout.height or 900
            convergence_fig.write_image(png_path, scale=2, width=fig_width, height=fig_height)
            print(f"  PNG saved to: {png_path}")
        except Exception as e:
            print(f"  Note: PNG export requires kaleido. Install with: pip install kaleido")
        
        # Save as PDF (vector graphics for paper)
        try:
            pdf_path = os.path.join(output_base, 'convergence_plots.pdf')
            convergence_fig.write_image(pdf_path, width=fig_width, height=fig_height)
            print(f"  PDF saved to: {pdf_path}")
        except Exception as e:
            print(f"  Note: PDF export requires kaleido. Install with: pip install kaleido")
        
        # Show the plot
        convergence_fig.show()
    
    # Create performance plots
    # if individuals_folders or households_folders:
        # print(f"\nCreating performance plots for {args.plot_type}...")
        # performance_fig = create_performance_plots(individuals_folders, households_folders, args.plot_type)
        # performance_fig.show()
        
        # Save performance plot
        # performance_plot_path = os.path.join(current_dir, args.output_dir, f'performance_plots{plot_type_suffix}.html')
        # performance_fig.write_html(performance_plot_path)
        # print(f"Performance plots saved to: {performance_plot_path}")
    
    # Create training progress plots
    # if individuals_folders or households_folders:
    #     print(f"\nCreating training progress plots for {args.plot_type}...")
    #     training_progress_fig = create_training_progress_plots(individuals_folders, households_folders, args.plot_type)
    #     training_progress_fig.show()
        
    #     # Save training progress plot
    #     training_progress_plot_path = os.path.join(current_dir, args.output_dir, f'training_progress_plots{plot_type_suffix}.html')
    #     training_progress_fig.write_html(training_progress_plot_path)
    #     print(f"Training progress plots saved to: {training_progress_plot_path}")
    
    # Create summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    if individuals_folders:
        print("\nIndividuals:")
        for area_code, folder_path in individuals_folders.items():
            performance_data = load_performance_data(folder_path)
            convergence_data = load_convergence_data(folder_path)
            if performance_data is not None:
                pop = performance_data['num_persons'].iloc[0]
                time_sec = performance_data['training_time_seconds'].iloc[0]
                accuracy = performance_data['final_accuracy'].iloc[0]
                
                # Add epoch timing info if available
                avg_epoch_time = "N/A"
                if convergence_data is not None and 'epoch_time_seconds' in convergence_data.columns:
                    avg_epoch_time = f"{convergence_data['epoch_time_seconds'].mean():.2f}s"
                
                print(f"  {area_code}: {pop:,} persons, {time_sec:.1f}s total, {avg_epoch_time} avg/epoch, {accuracy:.3f} accuracy")
    
    if households_folders:
        print("\nHouseholds:")
        for area_code, folder_path in households_folders.items():
            performance_data = load_performance_data(folder_path)
            convergence_data = load_convergence_data(folder_path)
            if performance_data is not None:
                hh = performance_data['num_households'].iloc[0]
                time_sec = performance_data['training_time_seconds'].iloc[0]
                accuracy = performance_data['final_accuracy'].iloc[0]
                
                # Add epoch timing info if available
                avg_epoch_time = "N/A"
                if convergence_data is not None and 'epoch_time_seconds' in convergence_data.columns:
                    avg_epoch_time = f"{convergence_data['epoch_time_seconds'].mean():.2f}s"
                
                print(f"  {area_code}: {hh:,} households, {time_sec:.1f}s total, {avg_epoch_time} avg/epoch, {accuracy:.3f} accuracy")

if __name__ == "__main__":
    main() 