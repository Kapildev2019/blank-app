import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from pyproj import Transformer
import io
import numpy as np
from shapely.geometry import Polygon

def convert_coordinates(df, from_epsg, to_epsg=4326):
    """Convert coordinates between EPSG systems"""
    transformer = Transformer.from_crs(f"epsg:{from_epsg}", f"epsg:{to_epsg}")
    lat, lon = transformer.transform(df['X'].values, df['Y'].values)
    return lat, lon

def get_adjacent_cells(x, y, grid_size, taken_cells, grid_gdf):
    """Get adjacent cell indices not already used"""
    adjacent = [
        (x + grid_size, y),
        (x - grid_size, y),
        (x, y + grid_size),
        (x, y - grid_size)
    ]
    valid_adj = []
    for ax, ay in adjacent:
        cell = grid_gdf[(grid_gdf['x'] == ax) & (grid_gdf['y'] == ay)]
        if not cell.empty and (ax, ay) not in taken_cells:
            valid_adj.append(cell.index[0])
    return valid_adj

def create_grid_and_select(gdf, grid_size_meters, preferred_species=None):
    """Create grid and select mother trees with new rules"""
    utm_gdf = gdf.to_crs(f"epsg:{gdf.crs.to_epsg()}")
    minx, miny, maxx, maxy = utm_gdf.total_bounds
    
    x_coords = np.arange(minx, maxx, grid_size_meters)
    y_coords = np.arange(miny, maxy, grid_size_meters)
    
    grid_cells = []
    try:
        for x in x_coords:
            for y in y_coords:
                # Define polygon coordinates
                coords = [
                    (x, y),
                    (x + grid_size_meters, y),
                    (x + grid_size_meters, y + grid_size_meters),
                    (x, y + grid_size_meters),
                    (x, y)
                ]
                # Create Polygon directly with shapely
                polygon = Polygon(coords)
                
                grid_cells.append({
                    'geometry': polygon,
                    'x': x,
                    'y': y
                })
    except Exception as e:
        raise ValueError(f"Error creating grid cells: {str(e)}")
    
    # Create GeoDataFrame with explicit geometry column
    grid_gdf = gpd.GeoDataFrame(
        grid_cells,
        geometry='geometry',
        crs=utm_gdf.crs
    )
    joined = gpd.sjoin(utm_gdf, grid_gdf, how='left', predicate='within')
    
    selected_points = []
    taken_trees = set()
    taken_cells = set()
    empty_cells = set()
    
    for idx, cell in grid_gdf.iterrows():
        cell_points = joined[joined['index_right'] == idx]
        
        if not cell_points.empty:
            priority1 = cell_points[
                (cell_points['Species'] == preferred_species) & 
                (cell_points['Class'] == 1)
            ]
            if not priority1.empty:
                point = priority1.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                taken_cells.add((cell['x'], cell['y']))
                continue
            
            priority2 = cell_points[
                (cell_points['Species'] == preferred_species) & 
                (cell_points['Class'] == 2)
            ]
            if not priority2.empty:
                point = priority2.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                taken_cells.add((cell['x'], cell['y']))
                continue
        else:
            empty_cells.add(idx)
    
    for empty_idx in empty_cells:
        cell = grid_gdf.loc[empty_idx]
        adj_cells = get_adjacent_cells(cell['x'], cell['y'], grid_size_meters, taken_cells, grid_gdf)
        
        for adj_idx in adj_cells:
            adj_points = joined[joined['index_right'] == adj_idx]
            
            adj_p1 = adj_points[
                (adj_points['Species'] == preferred_species) & 
                (adj_points['Class'] == 1) &
                (~adj_points['geometry'].isin(taken_trees))
            ]
            if not adj_p1.empty:
                point = adj_p1.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                taken_cells.add((grid_gdf.loc[adj_idx, 'x'], grid_gdf.loc[adj_idx, 'y']))
                break
                
            adj_p2 = adj_points[
                (adj_points['Species'] == preferred_species) & 
                (adj_points['Class'] == 2) &
                (~adj_points['geometry'].isin(taken_trees))
            ]
            if not adj_p2.empty:
                point = adj_p2.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                taken_cells.add((grid_gdf.loc[adj_idx, 'x'], grid_gdf.loc[adj_idx, 'y']))
                break
        else:
            for adj_idx in adj_cells:
                adj_points = joined[joined['index_right'] == adj_idx]
                multi_p1 = adj_points[
                    (adj_points['Species'] == preferred_species) & 
                    (adj_points['Class'] == 1) &
                    (~adj_points['geometry'].isin(taken_trees))
                ]
                if not multi_p1.empty:
                    point = multi_p1.iloc[0]
                    selected_points.append(point)
                    taken_trees.add(point['geometry'])
                    break
                    
                multi_p2 = adj_points[
                    (adj_points['Species'] == preferred_species) & 
                    (adj_points['Class'] == 2) &
                    (~adj_points['geometry'].isin(taken_trees))
                ]
                if not multi_p2.empty:
                    point = multi_p2.iloc[0]
                    selected_points.append(point)
                    taken_trees.add(point['geometry'])
                    break
    
    remaining_empty = set(grid_gdf.index) - set([p['index_right'] for p in selected_points])
    for empty_idx in remaining_empty:
        cell = grid_gdf.loc[empty_idx]
        cell_points = joined[joined['index_right'] == empty_idx]
        
        class3 = cell_points[
            (cell_points['Species'] == preferred_species) & 
            (cell_points['Class'] == 3) &
            (~cell_points['geometry'].isin(taken_trees))
        ]
        if not class3.empty:
            point = class3.iloc[0]
            selected_points.append(point)
            taken_trees.add(point['geometry'])
            taken_cells.add((cell['x'], cell['y']))
            continue
        
        adj_cells = get_adjacent_cells(cell['x'], cell['y'], grid_size_meters, set(), grid_gdf)
        for adj_idx in adj_cells:
            adj_points = joined[joined['index_right'] == adj_idx]
            adj_class3 = adj_points[
                (adj_points['Species'] == preferred_species) & 
                (adj_points['Class'] == 3) &
                (~adj_points['geometry'].isin(taken_trees))
            ]
            if not adj_class3.empty:
                point = adj_class3.iloc[0]
                selected_points.append(point)
                taken_trees.add(point['geometry'])
                taken_cells.add((grid_gdf.loc[adj_idx, 'x'], grid_gdf.loc[adj_idx, 'y']))
                break

    selected_gdf = gpd.GeoDataFrame(
        selected_points,
        geometry='geometry',
        crs=utm_gdf.crs
    )
    return grid_gdf.to_crs("EPSG:4326"), selected_gdf.to_crs("EPSG:4326")

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
                
            epsg_option = st.selectbox(
                "Select EPSG Coordinate System",
                [32644, 32645],
                format_func=lambda x: f"EPSG:{x} (UTM Zone {44 if x == 32644 else 45}N)"
            )
            
            species_list = df['Species'].unique().tolist()
            preferred_species = st.selectbox(
                "Select Preferred Mother Tree Species",
                species_list
            )
            
            grid_size = st.number_input(
                "Grid Size (meters)",
                min_value=10,
                max_value=1000,
                value=20,
                step=10
            )
            
            lat, lon = convert_coordinates(df, epsg_option)
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(lon, lat),
                crs="EPSG:4326"
            )
            
            grid_gdf, selected_points = create_grid_and_select(
                gdf.to_crs(f"epsg:{epsg_option}"),
                grid_size_meters=grid_size,
                preferred_species=preferred_species
            )
            
            m = folium.Map(
                location=[lat.mean(), lon.mean()],
                zoom_start=15,
                tiles="OpenStreetMap"
            )
            
            folium.GeoJson(
                grid_gdf,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'blue',
                    'weight': 1
                }
            ).add_to(m)
            
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
            
            st.subheader(f"Map with {grid_size}x{grid_size}m Grid and Mother Trees")
            folium_static(m, width=700, height=500)
            
            st.subheader("Statistics")
            st.write(f"Total trees: {len(gdf)}")
            st.write(f"Selected mother trees: {len(selected_points)}")
            st.write(f"Grid cells: {len(grid_gdf)}")
            st.write(f"Empty cells filled: {len(selected_points) - len(grid_gdf) + len([i for i in grid_gdf.index if i not in set(p['index_right'] for p in selected_points)])}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file with X, Y, Species, and Class columns")

if __name__ == "__main__":
    main()
