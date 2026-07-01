"""
Assignment Error Comparison Plot Generator

Creates paper-ready comparison plots between different assignment methods:
- Original GNN (GraphSAGE) from previous research
- Proposed GAT-based framework from this research

Uses Plotly for interactive plots, consistent with other plots in this research.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAPER-READY PLOTTING CONFIGURATION
# Consistent styling with other plots in this research
# ============================================================================
PLOT_CONFIG = {
    'figure_width': 1000,
    'figure_height': 600,
    'font_family': 'Arial',
    'fonts': {
        'title': 24,
        'axis_title': 18,
        'tick_labels': 14,
        'legend': 14,
        'bar_labels': 14
    },
    'colors': {
        'graphsage': '#D62728',      # Red for original/baseline
        'gat': '#1F77B4',            # Blue for proposed method
        'improvement_positive': '#2CA02C',  # Green for positive improvement
        'improvement_negative': '#D62728',  # Red for negative
        'grid': '#E5E5E5',
        'axis': '#333333'
    },
    'dpi': 300
}

# ============================================================================
# DATA: Assignment Errors for Area Code E02005940
# Total Population: 8842
# ============================================================================
TOTAL_POPULATION = 8842

# Original GraphSAGE Paper Results
GRAPHSAGE_ERRORS = {
    'Size': 4049,
    'Religion': 907,
    'Ethnicity': 857,
    # No Household Composition in original paper
}

# Proposed GAT Framework Results
GAT_ERRORS = {
    'Size': 121,
    'Religion': 990,
    'Ethnicity': 1050,
    'Household Composition': 871,
}

def calculate_error_rates(errors_dict, total_population):
    """Convert error counts to percentages."""
    return {k: (v / total_population) * 100 for k, v in errors_dict.items()}


def create_comparison_plot(output_dir=None):
    """
    Create a grouped bar chart comparing assignment errors between methods.
    Uses Plotly for interactive visualization.
    
    Similar to the LaTeX tikzpicture style with:
    - Grouped bars for each error type
    - Percentage labels above bars
    - Clean legend at bottom
    - Grid lines
    """
    
    # Calculate error rates as percentages
    graphsage_rates = calculate_error_rates(GRAPHSAGE_ERRORS, TOTAL_POPULATION)
    gat_rates = calculate_error_rates(GAT_ERRORS, TOTAL_POPULATION)
    
    # Categories to compare
    all_categories = ['Household Size', 'Religion', 'Ethnicity', 'Household Composition']
    
    # Prepare data
    graphsage_values = [
        graphsage_rates.get('Size', 0),
        graphsage_rates.get('Religion', 0),
        graphsage_rates.get('Ethnicity', 0),
        0  # N/A for Household Composition
    ]
    gat_values = [
        gat_rates.get('Size', 0),
        gat_rates.get('Religion', 0),
        gat_rates.get('Ethnicity', 0),
        gat_rates.get('Household Composition', 0)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add GraphSAGE bars
    fig.add_trace(go.Bar(
        name='Original GNN (GraphSAGE)',
        x=all_categories,
        y=graphsage_values,
        marker_color=PLOT_CONFIG['colors']['graphsage'],
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.85,
        text=[f'{v:.1f}%' if v > 0 else 'N/A' for v in graphsage_values],
        textposition='outside',
        textfont=dict(size=PLOT_CONFIG['fonts']['bar_labels'], family=PLOT_CONFIG['font_family'])
    ))
    
    # Add GAT bars
    fig.add_trace(go.Bar(
        name='Proposed Framework (GAT)',
        x=all_categories,
        y=gat_values,
        marker_color=PLOT_CONFIG['colors']['gat'],
        marker_line_color='black',
        marker_line_width=1,
        marker_pattern_shape='/',  # Hatching pattern
        opacity=0.85,
        text=[f'{v:.1f}%' for v in gat_values],
        textposition='outside',
        textfont=dict(size=PLOT_CONFIG['fonts']['bar_labels'], family=PLOT_CONFIG['font_family'])
    ))
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Error Type', font=dict(size=PLOT_CONFIG['fonts']['axis_title'])),
            tickfont=dict(size=PLOT_CONFIG['fonts']['tick_labels'], family=PLOT_CONFIG['font_family']),
            showline=True,
            linewidth=1.5,
            linecolor=PLOT_CONFIG['colors']['axis']
        ),
        yaxis=dict(
            title=dict(text='Assignment Error Rate (%)', font=dict(size=PLOT_CONFIG['fonts']['axis_title'])),
            tickfont=dict(size=PLOT_CONFIG['fonts']['tick_labels'], family=PLOT_CONFIG['font_family']),
            showgrid=True,
            gridcolor=PLOT_CONFIG['colors']['grid'],
            gridwidth=1,
            showline=True,
            linewidth=1.5,
            linecolor=PLOT_CONFIG['colors']['axis'],
            range=[0, max(max(graphsage_values), max(gat_values)) * 1.2]
        ),
        barmode='group',
        bargap=0.20,
        bargroupgap=0.1,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=PLOT_CONFIG['fonts']['legend'], family=PLOT_CONFIG['font_family']),
            bgcolor='rgba(255,255,255,0)',
            borderwidth=0
        ),
        width=PLOT_CONFIG['figure_width'],
        height=PLOT_CONFIG['figure_height'],
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=70, b=60),
        font=dict(family=PLOT_CONFIG['font_family'])
    )
    
    # Save plots
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, '..', 'outputs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HTML
    html_path = os.path.join(output_dir, 'assignment_error_comparison.html')
    fig.write_html(html_path)
    print(f"HTML saved to: {html_path}")
    
    # Save as PNG
    try:
        png_path = os.path.join(output_dir, 'assignment_error_comparison.png')
        fig.write_image(png_path, scale=2, width=PLOT_CONFIG['figure_width'], height=PLOT_CONFIG['figure_height'])
        print(f"PNG saved to: {png_path}")
    except Exception as e:
        print(f"Note: PNG export requires kaleido. Install with: pip install kaleido")
    
    # Save as PDF
    try:
        pdf_path = os.path.join(output_dir, 'assignment_error_comparison.pdf')
        fig.write_image(pdf_path, width=PLOT_CONFIG['figure_width'], height=PLOT_CONFIG['figure_height'])
        print(f"PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Note: PDF export requires kaleido. Install with: pip install kaleido")
    
    # Show in browser
    fig.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("ASSIGNMENT ERROR COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Area Code: E02005940")
    print(f"Total Population: {TOTAL_POPULATION:,}")
    print("-" * 60)
    print(f"{'Category':<25} {'GraphSAGE':<15} {'GAT (Proposed)':<15} {'Improvement':<15}")
    print("-" * 60)
    
    common_categories = ['Size', 'Religion', 'Ethnicity']
    for cat in common_categories:
        gs_rate = graphsage_rates.get(cat, 0)
        gat_rate = gat_rates.get(cat, 0)
        improvement = gs_rate - gat_rate
        print(f"{cat:<25} {gs_rate:>6.1f}%        {gat_rate:>6.1f}%        {improvement:>+6.1f}%")
    
    # Household Composition (GAT only)
    gat_rate = gat_rates.get('Household Composition', 0)
    print(f"{'Household Composition':<25} {'N/A':<14} {gat_rate:>6.1f}%        {'(new metric)':<15}")
    
    print("=" * 60)
    
    return fig


def create_improvement_plot(output_dir=None):
    """
    Create a bar chart showing the improvement (reduction) in error rates.
    Only for categories present in both methods.
    """
    
    # Calculate error rates
    graphsage_rates = calculate_error_rates(GRAPHSAGE_ERRORS, TOTAL_POPULATION)
    gat_rates = calculate_error_rates(GAT_ERRORS, TOTAL_POPULATION)
    
    # Common categories only
    categories = ['Size', 'Religion', 'Ethnicity']
    
    # Calculate improvements (positive = GAT is better)
    improvements = []
    colors = []
    for cat in categories:
        gs_rate = graphsage_rates.get(cat, 0)
        gat_rate = gat_rates.get(cat, 0)
        imp = gs_rate - gat_rate
        improvements.append(imp)
        colors.append(PLOT_CONFIG['colors']['improvement_positive'] if imp > 0 
                      else PLOT_CONFIG['colors']['improvement_negative'])
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=improvements,
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.85,
        text=[f'{v:+.1f}%' for v in improvements],
        textposition='outside',
        textfont=dict(size=PLOT_CONFIG['fonts']['bar_labels'] + 2, 
                      family=PLOT_CONFIG['font_family'],
                      color='black')
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_width=2, line_color='black')
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Improvement in Assignment Error Rates: GAT vs GraphSAGE',
            font=dict(size=PLOT_CONFIG['fonts']['title'], family=PLOT_CONFIG['font_family']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Error Type', font=dict(size=PLOT_CONFIG['fonts']['axis_title'])),
            tickfont=dict(size=PLOT_CONFIG['fonts']['tick_labels'], family=PLOT_CONFIG['font_family']),
            showline=True,
            linewidth=1.5,
            linecolor=PLOT_CONFIG['colors']['axis']
        ),
        yaxis=dict(
            title=dict(text='Error Rate Improvement (%)', font=dict(size=PLOT_CONFIG['fonts']['axis_title'])),
            tickfont=dict(size=PLOT_CONFIG['fonts']['tick_labels'], family=PLOT_CONFIG['font_family']),
            showgrid=True,
            gridcolor=PLOT_CONFIG['colors']['grid'],
            gridwidth=1,
            showline=True,
            linewidth=1.5,
            linecolor=PLOT_CONFIG['colors']['axis']
        ),
        width=900,
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=80, b=100),
        font=dict(family=PLOT_CONFIG['font_family']),
        annotations=[
            dict(
                text='Positive values indicate improvement (lower error rate with GAT)',
                xref='paper', yref='paper',
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color='gray', style='italic'),
                xanchor='center'
            )
        ]
    )
    
    # Save
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, '..', 'outputs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HTML
    html_path = os.path.join(output_dir, 'assignment_error_improvement.html')
    fig.write_html(html_path)
    print(f"Improvement plot HTML saved to: {html_path}")
    
    # Save as PNG
    try:
        png_path = os.path.join(output_dir, 'assignment_error_improvement.png')
        fig.write_image(png_path, scale=2, width=900, height=550)
        print(f"Improvement plot PNG saved to: {png_path}")
    except Exception as e:
        print(f"Note: PNG export requires kaleido. Install with: pip install kaleido")
    
    # Save as PDF
    try:
        pdf_path = os.path.join(output_dir, 'assignment_error_improvement.pdf')
        fig.write_image(pdf_path, width=900, height=550)
        print(f"Improvement plot PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Note: PDF export requires kaleido. Install with: pip install kaleido")
    
    # Show in browser
    fig.show()
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("ASSIGNMENT ERROR COMPARISON PLOT GENERATOR")
    print("=" * 60)
    
    # Create main comparison plot
    print("\nGenerating comparison plot...")
    create_comparison_plot()
    
    # Create improvement plot
    print("\nGenerating improvement plot...")
    create_improvement_plot()
    
    print("\nDone!")
