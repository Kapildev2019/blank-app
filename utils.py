import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
import numpy as np
from rtree import index
import folium
from streamlit_folium import st_folium

def create_grid(bounds, grid_size):
    """Creates a grid of polygons covering the given bounds."""
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))

    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=pyproj.CRS.from_epsg(4326))  # Assuming initial CRS is 4326
    return grid

def select_mother_trees(gdf, grid_cells, priority_species_1, priority_species_2, species_col, class_col):
    """Selects mother trees based on the specified criteria."""
    mother_trees = []
    selected_tree_ids = set()  # To prevent double selection
    num_grids = len(grid_cells)

    # --- Function to search for mother trees for a given species ---
    def search_for_trees(species):
        nonlocal mother_trees, selected_tree_ids
        for i, grid_cell in enumerate(grid_cells.geometry):
            trees_in_grid = gdf[gdf.within(grid_cell)]
            priority_trees_class1 = trees_in_grid[(trees_in_grid[species_col] == species) & (trees_in_grid[class_col] == 1)]

            if not priority_trees_class1.empty:
                tree = priority_trees_class1.iloc[0]
                tree_id = tree.index
                if tree_id not in selected_tree_ids:
                    mother_trees.append(tree)
                    selected_tree_ids.add(tree_id)
                    if len(mother_trees) == num_grids:
                        return True  # Stop if we have enough mother trees

            elif len(mother_trees) < num_grids:
                priority_trees_class2 = trees_in_grid[(trees_in_grid[species_col] == species) & (trees_in_grid[class_col] == 2)]
                if not priority_trees_class2.empty:
                    tree = priority_trees_class2.iloc[0]
                    tree_id = tree.index
                    if tree_id not in selected_tree_ids:
                        mother_trees.append(tree)
                        selected_tree_ids.add(tree_id)
                        if len(mother_trees) == num_grids:
                            return True  # Stop if we have enough mother trees

            elif len(mother_trees) < num_grids:
                priority_trees_class3 = trees_in_grid[(trees_in_grid[species_col] == species) & (trees_in_grid[class_col] == 3)]
                if not priority_trees_class3.empty:
                    tree = priority_trees_class3.iloc[0]
                    tree_id = tree.index
                    if tree_id not in selected_tree_ids:
                        mother_trees.append(tree)
                        selected_tree_ids.add(tree_id)
                        if len(mother_trees) == num_grids:
                            return True  # Stop if we have enough mother trees
        return False  # Not enough trees found for this species

    # --- Search for trees for priority species 1 ---
    if not search_for_trees(priority_species_1):
        # --- Search for trees for priority species 2 ---
        if priority_species_2:
            if priority_species_2 in gdf[species_col].unique():  # Check if species 2 exists in the data
                search_for_trees(priority_species_2)
            else:
                st.warning(f"Second priority species '{priority_species_2}' not found in the data. Skipping.")

    return gpd.GeoDataFrame(mother_trees) if mother_trees else None

def display_map(gdf, mother_trees):
    """Displays the trees on a map with color-coding and measurement tool."""

    # Get the bounds of the GeoDataFrame
    bounds = gdf.total_bounds

    # Create a Folium map centered on the data
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=10, tiles="OpenStreetMap")

    # Add distance measurement tool
    folium.plugins.measure.Measure(position='topright').add_to(m)

    # Add mother trees in green
    if mother_trees is not None:
        for index, row in mother_trees.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Species: {row['species']}, Class: {row['class']}"
            ).add_to(m)

    # Add other trees in grey
    other_trees = gdf[~gdf.index.isin(mother_trees.index)]
    for index, row in other_trees.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color='grey',
            fill=True,
            fill_color='grey',
            fill_opacity=0.6,
            popup=f"Species: {row['species']}, Class: {row['class']}"
        ).add_to(m)

    # Display the map using streamlit-folium
    st_folium(m, width=725, height=500)