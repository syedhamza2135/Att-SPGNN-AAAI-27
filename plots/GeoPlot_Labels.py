"""
Oxford MSOA Choropleth Map with Area Code Labels

Produces a paper-ready geographic visualization of Oxford MSOAs
with population data and area code annotations.
"""

import os, json
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAPER-READY PLOT CONFIGURATION
# ============================================================================
PLOT_CONFIG = {
    'figure_width': 1400,
    'figure_height': 1000,
    'font_family': 'Arial',
    'fonts': {
        'title': 22,
        'labels': 16,
        'colorbar': 14
    },
    'colors': {
        'boundary': '#2C3E50',        # Dark slate for boundary
        'label_text': '#1A1A2E',      # Dark navy for labels
        'label_halo': 'white',        # White halo for readability
        'background': '#F8F9FA'       # Light gray background
    }
}

# Custom color scale - Muted, softer tones for paper readability
CUSTOM_COLORSCALE = [
    [0.0, '#3D5A6C'],      # Muted slate blue
    [0.2, '#4A7C7E'],      # Soft teal
    [0.4, '#6B9080'],      # Sage gray-green
    [0.6, '#A4C3A2'],      # Soft mint
    [0.8, '#D4C5A9'],      # Warm beige
    [1.0, '#C9A87C']       # Muted tan
]

# ── 1.  Paths & data ───────────────────────────────────────────────────
BASE = "../data/"
PERSONS_DIR = "../data/preprocessed-data/individuals"

# Load age data and rename the geography code
age = pd.read_csv(os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv"))
age = age.rename(columns={"geography code": "MSOA21CD"})

# Load shapefiles
msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
red_fp  = os.path.join(BASE, "geodata", "boundary.geojson")

# ── 2.  Read in spatial data ───────────────────────────────────────────
gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
red_bnd = gpd.read_file(red_fp).to_crs(4326)

# ── 3.  Merge population totals ────────────────────────────────────────
gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
gdf_msoa["total"] = gdf_msoa["total"].fillna(0)

# ── 3.5. Manually remove unwanted MSOAs ────────────────────────────────
exclude_codes = [
    "E02005939", "E02005979", "E02005963", "E02005959"
]
gdf_msoa = gdf_msoa[~gdf_msoa["MSOA21CD"].isin(exclude_codes)]

# ── 4.  Clip to red boundary ───────────────────────────────────────────
red_union = red_bnd.unary_union
gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()

# ── 5.  Compute accurate centroids for labels ──────────────────────────
proj_crs = 27700
gdf_proj = gdf_clip.to_crs(proj_crs)
centroids = gdf_proj.geometry.centroid.to_crs(4326)
gdf_clip["lon"] = centroids.x
gdf_clip["lat"] = centroids.y

# ── 6.  Create choropleth plot with custom styling ─────────────────────
fig = px.choropleth(
    gdf_clip,
    geojson=json.loads(gdf_clip.to_json()),
    locations="MSOA21CD",
    featureidkey="properties.MSOA21CD",
    color="total",
    color_continuous_scale=CUSTOM_COLORSCALE,
    projection="mercator",
    hover_data={"MSOA21CD": True, "total": True}
)

# Update choropleth traces styling
fig.update_traces(
    marker_line_color=PLOT_CONFIG['colors']['boundary'],
    marker_line_width=1.5,
    hovertemplate="<b>%{customdata[0]}</b><br>Population: %{customdata[1]:,}<extra></extra>"
)

# ── 7.  Add MSOA21CD labels with enhanced styling ──────────────────────
# Create short labels (last 4 digits only for cleaner look)
gdf_clip["short_code"] = gdf_clip["MSOA21CD"].str[-4:]

fig.add_trace(
    go.Scattergeo(
        lon=gdf_clip["lon"],
        lat=gdf_clip["lat"],
        mode="text",
        text=gdf_clip["MSOA21CD"],  # Full area code
        textfont=dict(
            size=PLOT_CONFIG['fonts']['labels'],
            color=PLOT_CONFIG['colors']['label_text'],
            family=PLOT_CONFIG['font_family'],
            weight='bold'
        ),
        hoverinfo="none",
        showlegend=False
    )
)

# ── 8.  Add boundary outline with enhanced styling ─────────────────────
for poly in red_bnd.geometry.explode(index_parts=False):
    # Main boundary line
    fig.add_trace(
        go.Scattergeo(
            lon=list(poly.exterior.coords.xy[0]),
            lat=list(poly.exterior.coords.xy[1]),
            mode='lines',
            line=dict(
                color=PLOT_CONFIG['colors']['boundary'],
                width=3.5
            ),
            showlegend=False,
            hoverinfo='none'
        )
    )

# ── 9.  Set view area with proper padding ──────────────────────────────
lat_min, lat_max = gdf_clip["lat"].min(), gdf_clip["lat"].max()
lon_min, lon_max = gdf_clip["lon"].min(), gdf_clip["lon"].max()

# Define padding for a balanced view
lat_padding = (lat_max - lat_min) * 0.15
lon_padding = (lon_max - lon_min) * 0.25

# Apply padded view ranges
fig.update_geos(
    visible=False,
    bgcolor=PLOT_CONFIG['colors']['background'],
    lataxis_range=[lat_min - lat_padding, lat_max + lat_padding],
    lonaxis_range=[lon_min - lon_padding, lon_max + lon_padding]
)

# ── 10.  Update layout for paper-ready appearance ──────────────────────
fig.update_layout(
    width=PLOT_CONFIG['figure_width'],
    height=PLOT_CONFIG['figure_height'],
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor=PLOT_CONFIG['colors']['background'],
    plot_bgcolor=PLOT_CONFIG['colors']['background'],
    font=dict(
        family=PLOT_CONFIG['font_family'],
        size=14
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Population",
            font=dict(
                size=18,
                family=PLOT_CONFIG['font_family']
            )
        ),
        tickfont=dict(
            size=14,
            family=PLOT_CONFIG['font_family']
        ),
        len=0.6,
        thickness=20,
        x=1.02,
        y=0.5,
        yanchor='middle',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=PLOT_CONFIG['colors']['boundary'],
        borderwidth=1
    )
)

# ── 11.  Save and show the plot ────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))

# Save as HTML
html_path = os.path.join(output_dir, 'oxford_msoa_map.html')
fig.write_html(html_path)
print(f"HTML saved to: {html_path}")

# Save as PNG (high resolution)
try:
    png_path = os.path.join(output_dir, 'oxford_msoa_map.png')
    fig.write_image(png_path, scale=2, width=PLOT_CONFIG['figure_width'], 
                    height=PLOT_CONFIG['figure_height'])
    print(f"PNG saved to: {png_path}")
except Exception as e:
    print(f"Note: PNG export requires kaleido. Install with: pip install kaleido")

# Save as PDF
try:
    pdf_path = os.path.join(output_dir, 'oxford_msoa_map.pdf')
    fig.write_image(pdf_path, width=PLOT_CONFIG['figure_width'], 
                    height=PLOT_CONFIG['figure_height'])
    print(f"PDF saved to: {pdf_path}")
except Exception as e:
    print(f"Note: PDF export requires kaleido. Install with: pip install kaleido")

# Show in browser
fig.show()
