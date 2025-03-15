import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from pyproj import Transformer
import numpy as np
from shapely.geometry import Polygon
import time
import os

def convert_coordinates(df, from_epsg, to_epsg=4326):
    """Convert coordinates between EPSG systems"""
    transformer = Transformer.from_crs(f"epsg:{from_epsg}", f"epsg:{to_epsg}")
    lat, lon = transformer.transform(df['X'].values, df['Y'].values)
    return lat, lon

def create_grid_and_select(gdf, grid_size_meters, preferred_species=None):
    """Optimized grid creation and selection"""
    start_time = time.time()
    
    utm_gdf = gdf.to_crs(f"epsg:{gdf.crs.to_epsg()}")
    minx, miny, maxx, maxy = utm_gdf.total_bounds
    
    x_coords = np.arange(minx, maxx, grid_size_meters)
    y_coords = np.arange(miny, maxy, grid_size_meters)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    grid_cells = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        coords = [
            (x, y),
            (x + grid_size_meters, y),
            (x + grid_size_meters, y + grid_size_meters),
            (x, y + grid_size_meters),
            (x, y)
        ]
        polygon = Polygon(coords)
        grid_cells.append({'geometry': polygon, 'x': x, 'y': y})
    
    grid_gdf = gpd.GeoDataFrame(grid_cells, geometry='geometry', crs=utm_gdf.crs)
    
    utm_gdf_with_index = utm_gdf.reset_index()
    joined = gpd.sjoin(utm_gdf_with_index, grid_gdf, how='left', predicate='within')
    
    grouped = joined.groupby('index_right')
    
    selected_points = []
    taken_trees = set()
    empty_cells = set(grid_gdf.index)
    
    for idx in grid_gdf.index:
        if idx in grouped.groups:
            cell_points = grouped.get_group(idx)
            for priority in [1, 2]:
                priority_points = cell_points[
                    (cell_points['Species'] == preferred_species) & 
                    (cell_points['Class'] == priority) &
                    (~cell_points['geometry'].isin(taken_trees))
                ]
                if not priority_points.empty:
                    point = priority_points.iloc[0]
                    selected_points.append(point)
                    taken_trees.add(point['geometry'])
                    empty_cells.remove(idx)
                    break
    
    for empty_idx in list(empty_cells):
        cell = grid_gdf.loc[empty_idx]
        x, y = cell['x'], cell['y']
        adjacent = [
            (x + grid_size_meters, y),
            (x - grid_size_meters, y),
            (x, y + grid_size_meters),
            (x, y - grid_size_meters)
        ]
        
        for adj_x, adj_y in adjacent:
            adj_cell = grid_gdf[(grid_gdf['x'] == adj_x) & (grid_gdf['y'] == adj_y)]
            if not adj_cell.empty and adj_cell.index[0] in grouped.groups:
                adj_points = grouped.get_group(adj_cell.index[0])
                for priority in [1, 2]:
                    priority_points = adj_points[
                        (adj_points['Species'] == preferred_species) & 
                        (adj_points['Class'] == priority) &
                        (~adj_points['geometry'].isin(taken_trees))
                    ]
                    if not priority_points.empty:
                        point = priority_points.iloc[0]
                        selected_points.append(point)
                        taken_trees.add(point['geometry'])
                        empty_cells.remove(empty_idx)
                        break
                if empty_idx not in empty_cells:
                    break
    
    for empty_idx in list(empty_cells):
        cell = grid_gdf.loc[empty_idx]
        if empty_idx in grouped.groups:
            cell_points = grouped.get_group(empty_idx)
            class3 = cell_points[
                (cell_points['Species'] == preferred_species) & 
                (cell_points['Class'] == 3) &
                (~cell_points['geometry'].isin(taken_trees))
            ]
            if not class3.empty:
                point = class3.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                empty_cells.remove(empty_idx)
                continue
        
        x, y = cell['x'], cell['y']
        adjacent = [
            (x + grid_size_meters, y),
            (x - grid_size_meters, y),
            (x, y + grid_size_meters),
            (x, y - grid_size_meters)
        ]
        for adj_x, adj_y in adjacent:
            adj_cell = grid_gdf[(grid_gdf['x'] == adj_x) & (grid_gdf['y'] == adj_y)]
            if not adj_cell.empty and adj_cell.index[0] in grouped.groups:
                adj_points = grouped.get_group(adj_cell.index[0])
                class3 = adj_points[
                    (adj_points['Species'] == preferred_species) & 
                    (adj_points['Class'] == 3) &
                    (~adj_points['geometry'].isin(taken_trees))
                ]
                if not class3.empty:
                    point = class3.iloc[0]
                    selected_points.append(point)
                    taken_trees.add(point['geometry'])
                    empty_cells.remove(empty_idx)
                    break
    
    selected_gdf = gpd.GeoDataFrame(selected_points, geometry='geometry', crs=utm_gdf.crs)
    
    execution_time = time.time() - start_time
    return grid_gdf.to_crs("EPSG:4326"), selected_gdf.to_crs("EPSG:4326"), execution_time

def main():
    st.title("Mother Tree Selection App")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(df.head())
            
            required_cols = ['X', 'Y', 'Species', 'Class']
            if not all(col in df.columns for col in required_cols):
                st.error("CSV must contain 'X', 'Y', 'Species', and 'Class' columns")
                return
                
            # Configuration Section
            st.subheader("Analysis Configuration")
            
            # EPSG Selection
            epsg_option = st.selectbox(
                "Select EPSG Coordinate System",
                [32644, 32645],
                format_func=lambda x: f"EPSG:{x} (UTM Zone {44 if x == 32644 else 45}N)"
            )
            
            # Species Selection
            species_list = df['Species'].unique().tolist()
            preferred_species = st.selectbox(
                "Select Preferred Mother Tree Species",
                options=species_list,
                help="Choose the species to prioritize for mother tree selection"
            )
            
            # Grid Size Selection
            grid_size = st.number_input(
                "Grid Size (meters)",
                min_value=10,
                max_value=1000,
                value=20,
                step=10,
                help="Set the size of each grid cell in meters"
            )
            
            # Proceed Button
            if st.button("Proceed with Analysis"):
                with st.spinner("Running analysis..."):
                    lat, lon = convert_coordinates(df, epsg_option)
                    gdf = gpd.GeoDataFrame(
                        df, 
                        geometry=gpd.points_from_xy(lon, lat),
                        crs="EPSG:4326"
                    )
                    
                    grid_gdf, selected_points, execution_time = create_grid_and_select(
                        gdf.to_crs(f"epsg:{epsg_option}"),
                        grid_size_meters=grid_size,
                        preferred_species=preferred_species
                    )
                    
                    # Map View Options
                    st.subheader("Map View Options")
                    show_mother_trees = st.checkbox("Show Mother Trees", value=True)
                    show_other_trees = st.checkbox("Show Other Trees", value=True)
                    show_grid = st.checkbox("Show Grid", value=True)
                    
                    # Create Folium Map
                    m = folium.Map(
                        location=[lat.mean(), lon.mean()],
                        zoom_start=15,
                        tiles="OpenStreetMap"
                    )
                    
                    # Add Grid to Map
                    if show_grid:
                        folium.GeoJson(
                            grid_gdf,
                            style_function=lambda x: {
                                'fillColor': 'transparent',
                                'color': 'blue',
                                'weight': 1
                            }
                        ).add_to(m)
                    
                    # Add Other Trees to Map
                    if show_other_trees:
                        for idx, row in gdf.iterrows():
                            folium.CircleMarker(
                                location=[row.geometry.y, row.geometry.x],
                                radius=3,
                                color='gray',
                                fill=True,
                                fill_color='gray',
                                fill_opacity=0.5,
                                popup=f"Species: {row['Species']}<br>Class: {row['Class']}"
                            ).add_to(m)
                    
                    # Add Mother Trees to Map
                    if show_mother_trees:
                        for idx, row in selected_points.iterrows():
                            folium.CircleMarker(
                                location=[row.geometry.y, row.geometry.x],
                                radius=5,
                                color='red',
                                fill=True,
                                fill_color='red',
                                fill_opacity=0.7,
                                popup=f"Mother Tree<br>Species: {row['Species']}<br>Class: {row['Class']}"
                            ).add_to(m)
                    
                    # Display Map
                    st.subheader(f"Map with {grid_size}x{grid_size}m Grid and Mother Trees")
                    folium_static(m, width=700, height=500)
                    
                    # Statistics
                    st.subheader("Statistics")
                    st.write(f"Execution time: {execution_time:.2f} seconds")
                    st.write(f"Total trees: {len(gdf)}")
                    st.write(f"Selected mother trees: {len(selected_points)}")
                    st.write(f"Grid cells: {len(grid_gdf)}")
                    st.write(f"Empty cells filled: {len(selected_points) - len(grid_gdf) + len([i for i in grid_gdf.index if i not in set(p['index_right'] for p in selected_points)])}")
                    
                    # Download Analyzed Data
                    st.subheader("Download Analyzed Data")
                    if st.button("Download Mother Trees as CSV"):
                        csv = selected_points.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="mother_trees.csv",
                            mime="text/csv"
                        )
                    
                    if st.button("Download Mother Trees as Shapefile"):
                        selected_points.to_file("mother_trees.shp")
                        with open("mother_trees.shp", "rb") as f:
                            st.download_button(
                                label="Download Shapefile",
                                data=f,
                                file_name="mother_trees.zip",
                                mime="application/zip"
                            )
                        os.remove("mother_trees.shp")  # Clean up temporary file
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file with X, Y, Species, and Class columns")

if __name__ == "__main__":
    main()
