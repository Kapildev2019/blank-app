import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
import utils  # Import your helper functions

# --- Configuration ---
st.set_page_config(layout="wide")

# --- Sidebar ---
st.sidebar.header("Configuration")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        df = None

    if df is not None:
        # Column Selection
        st.sidebar.subheader("Column Mapping")
        species_col = st.sidebar.selectbox("Species Column", df.columns)
        x_col = st.sidebar.selectbox("X Coordinate Column", df.columns)
        y_col = st.sidebar.selectbox("Y Coordinate Column", df.columns)
        class_col = st.sidebar.selectbox("Class Column", df.columns)

        # EPSG Selection
        epsg_options = [32644, 32645]
        epsg = st.sidebar.selectbox("EPSG Code", epsg_options)

        # Grid Size Selection
        grid_size = st.sidebar.slider("Grid Size (meters)", 5, 100, 25)

        # Priority Species Selection
        unique_species = df[species_col].unique().tolist()
        priority_species_1 = st.sidebar.selectbox("Priority Species 1", unique_species)

        # Second Priority Species (after processing first)
        priority_species_2 = None  # Initialize

        # --- Main Area ---
        st.title("Mother Tree Selection")

        # Convert to GeoDataFrame
        try:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[x_col], df[y_col]),
                crs=pyproj.CRS.from_epsg(epsg)  # Set CRS
            )
        except Exception as e:
            st.error(f"Error creating GeoDataFrame: {e}")
            gdf = None

        if gdf is not None:
            # Create Grid
            bounds = gdf.total_bounds
            grid_cells = utils.create_grid(bounds, grid_size)  # Implement in utils.py

            # Mother Tree Selection
            mother_trees = utils.select_mother_trees(
                gdf,
                grid_cells,
                priority_species_1,
                species_col,
                class_col,
                priority_species_2  # Pass second species if selected
            )  # Implement in utils.py

            if mother_trees is not None:
                # Display on Map
                st.subheader("Mother Tree Map")
                utils.display_map(mother_trees, grid_cells)  # Implement in utils.py

                # Measurement Tool (Implement in utils.py or use a mapping library feature)
                # st.subheader("Measurement Tool")
                # ...

            else:
                st.warning("No mother trees selected.")
        else:
            st.warning("Please check your data and column selections.")
else:
    st.info("Please upload an Excel file to begin.")