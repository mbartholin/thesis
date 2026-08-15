import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================================
# AUTO-DETECT ALL NC FILES IN SUBFOLDERS
# ============================================================================

# def find_all_nc_files(base_path='.'):
#     """Recursively find all .nc files and create dictionary with clean keys"""
#     base_path = Path('./OBS')
#     nc_files = {}
    
#     # Find all .nc files in current folder and subfolders
#     for nc_file in base_path.rglob('*.nc'):
#         # Create a clean key name from the file path
#         # Remove extension and replace path separators with underscores
#         key = str(nc_file.relative_to(base_path)).replace('/', '').replace('\\', '_').replace('.nc', '').replace('observations_', '')
        
        
#         # Clean up the key name for better readability
#         key = key.replace('ProcessedData_', '')
#         key = key.replace('processed_', '')
        
#         # Store both key and file path
#         nc_files[key] = str(nc_file)
    
#     return nc_files

# # Auto-detect all NC files
# NC_FILES = find_all_nc_files('.')

# # Print found files
# print(f"Found {len(NC_FILES)} NetCDF files:")
# for key, path in list(NC_FILES.items())[:10]:  # Show first 10
#     print(f"  '{key}': '{path}',")
# if len(NC_FILES) > 10:
#     print(f"  ... and {len(NC_FILES)-10} more")


NC_FILES = {
    'UW_XR_1600m_VAD': 'OBS/UW_XR_1600m_VAD.nc',
    'UW_XR_3000m_VAD': 'OBS/UW_XR_3000m_VAD.nc',
    'WEA_XR_1500m_VAD': 'OBS/WEA_XR_1500m_VAD.nc',
    'N-03-07-UW_V2': 'OBS/windcube_N-03-07-UW_V2.nc',
    'Balticsea_Lidar_Lot3': 'OBS/balticsea_lot3.nc',
    'Balticsea_Lidar_Lot4': 'OBS/processed_balticsea_lidar_lot4.nc',
    'Northsea_Lidar_Lot1': 'OBS/processed_northsea_lidar_lot1.nc',
    'Northsea_Lidar_Lot2': 'OBS/processed_northsea_lidar_lot2.nc',
    'Cabauw': 'OBS/Cabauw_2001-2025.nc',
    'Hamburg_met_mast': 'OBS/hamburg_weathermast.nc', 
    # 'Oesterild_north': 'OBS/oesterild_north_202112_202212.nc',
    'Oesterild_LMN': 'OBS/Oesterild_LMN_20190101-20240903.nc',
    'Hovsore': 'OBS/Hovsore_2004-2022.nc',
    'FINO3': 'OBS/FINO3_20090911_20260131_INCL_STD.nc',
    'FINO2': 'OBS/FINO2_20080101_20251130_INCL_STD.nc',
    'FINO1': 'OBS/FINO1_20040101_20260131_INCL_STD.nc',

    }

# ============================================================================
# REST OF YOUR CODE
# ============================================================================

WIND_VARIABLE_NAMES = [
    'wind_speed',
    'wind_speed_filtered', 
    'ws',
    'wind',
    'wspd',
    'wind_speed_10m',
    'wind_speed_100m',
    'sfc_wind',
    'wind_spd',
    'WS10M',
    'WS100M',
    'WSPD',
    'WIND_SPEED',
    'wind_speed_mean',
    'wind_speed_avg', 
    'nl_wspd', 
    'WSPD',
    'F'
]

TIME_DIMENSION_NAMES = [
    'time',
    'Time',
    'TIME',
    't',
    'T',
    'datetime',
    'Datetime',
    'date',
    'Date',
    'timestamp',
    'Timestamp',
    'valid_time',
    'forecast_time',
    'Times',
    'times'
]

OUTPUT_FILE = 'data_availability.png'
FIG_SIZE = (14, 10)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_wind_variable(ds):
    """Find which wind variable exists in the dataset."""
    for var_name in WIND_VARIABLE_NAMES:
        if var_name in ds.variables:
            return var_name
    return None

def find_time_dimension(ds):
    """Find which dimension is the time dimension."""
    # First check coordinates
    for coord_name in TIME_DIMENSION_NAMES:
        if coord_name in ds.coords:
            return coord_name
    
    # Then check dimensions
    for dim_name in TIME_DIMENSION_NAMES:
        if dim_name in ds.dims:
            return dim_name
    
    # Look for any dimension that has datetime values
    for dim_name in ds.dims:
        try:
            if dim_name in ds.coords:
                values = ds[dim_name].values
                # Check if it looks like datetime
                if len(values) > 0:
                    first_val = values[0]
                    if isinstance(first_val, (np.datetime64, pd.Timestamp)):
                        return dim_name
                    # Check if string that can be parsed as datetime
                    if isinstance(first_val, str):
                        pd.to_datetime(first_val)  # Try parsing
                        return dim_name
        except:
            continue
    
    # Look for any coordinate with datetime values
    for coord_name in ds.coords:
        try:
            values = ds[coord_name].values
            if len(values) > 0:
                first_val = values[0]
                if isinstance(first_val, (np.datetime64, pd.Timestamp)):
                    return coord_name
        except:
            continue
    
    return None

def convert_to_time(ds, time_dim):
    """Convert time dimension to proper datetime format."""
    try:
        time_values = ds[time_dim].values
        
        # If it's already datetime
        if isinstance(time_values[0], (np.datetime64, pd.Timestamp)):
            return time_dim
        
        # If it's string
        if isinstance(time_values[0], str):
            ds = ds.assign_coords({time_dim: pd.to_datetime(time_values)})
            return time_dim
        
        # If it's numeric (hours since, etc.)
        if np.issubdtype(time_values.dtype, np.number):
            # Try to interpret as hours since 1900-01-01
            try:
                time_dt = pd.to_datetime('1900-01-01') + pd.to_timedelta(time_values, unit='h')
                ds = ds.assign_coords({time_dim: time_dt})
                return time_dim
            except:
                # Try as days since 1900-01-01
                try:
                    time_dt = pd.to_datetime('1900-01-01') + pd.to_timedelta(time_values, unit='D')
                    ds = ds.assign_coords({time_dim: time_dt})
                    return time_dim
                except:
                    pass
        
        return time_dim
        
    except Exception as e:
        print(f"    ⚠ Could not convert time: {str(e)}")
        return time_dim

# ============================================================================
# AVAILABILITY COMPARISON
# ============================================================================

print("\n=== DATA AVAILABILITY COMPARISON ===\n")

# Create figure
fig, ax = plt.subplots(figsize=FIG_SIZE)

# Process each station
y_pos = 0
stations = []
avail_percentages = []
used_variables = []
time_issues = []
failed_files = []
all_time_ranges = []  # Store time ranges for each station

for station_name, file_path in NC_FILES.items():
    print(f"\nProcessing: {station_name}")
    print(f"  File: {file_path}")
    
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"  ⚠ File not found: {file_path}")
            failed_files.append(station_name)
            y_pos += 1
            continue
        
        # Load data
        ds = xr.open_dataset(file_path)
        print(f"  ✓ Dataset loaded")
        print(f"  ✓ Dimensions: {list(ds.dims.keys())}")
        print(f"  ✓ Coordinates: {list(ds.coords.keys())}")
        
        # Find time dimension
        time_dim = find_time_dimension(ds)
        
        if time_dim is None:
            print(f"  ⚠ No time dimension found!")
            print(f"    Attempted: {TIME_DIMENSION_NAMES}")
            print(f"    Available dims: {list(ds.dims.keys())}")
            print(f"    Available coords: {list(ds.coords.keys())}")
            time_issues.append(station_name)
            y_pos += 1
            continue
        
        print(f"  ✓ Found time dimension: '{time_dim}'")
        
        # Convert time to proper datetime
        try:
            time_dim = convert_to_time(ds, time_dim)
            ds = ds.assign_coords({time_dim: pd.to_datetime(ds[time_dim].values)})
            print(f"  ✓ Time converted to datetime")
        except Exception as e:
            print(f"  ⚠ Time conversion issue: {str(e)}")
        
        # Find wind variable
        wind_var = find_wind_variable(ds)
        
        if wind_var is None:
            print(f"  ⚠ No wind variable found!")
            print(f"    Attempted: {WIND_VARIABLE_NAMES[:5]}...")
            print(f"    Available variables: {list(ds.data_vars.keys())[:10]}")
            y_pos += 1
            continue
        
        print(f"  ✓ Using wind variable: '{wind_var}'")
        used_variables.append((station_name, wind_var))
        
        # Get wind speed data
        ws = ds[wind_var]
        
        # Handle multi-dimensional data
        if len(ws.dims) > 1:
            sel_dict = {}
            
            for dim in ws.dims:
                if dim != time_dim:
                    # Special handling for level dimensions
                    if dim in ['lev_wspd', 'lev_wdir', 'lev_tair', 'lev_rhum', 'height', 'level', 'altitude']:
                        # Get the wind speed data along this dimension
                        dim_data = ws.isel({dim: slice(None)})
                        
                        # Calculate number of valid values for each level
                        valid_counts = []
                        level_values = ds[dim].values if dim in ds.coords else range(dim_data.sizes[dim])
                        
                        print(f"  Analyzing {dim}:")
                        for i in range(dim_data.sizes[dim]):
                            try:
                                # Select this specific level
                                level_slice = dim_data.isel({dim: i})
                                # Count non-NaN values
                                if hasattr(level_slice, 'compute'):
                                    valid_count = level_slice.count().compute()
                                else:
                                    valid_count = level_slice.count().values
                                valid_counts.append(valid_count)
                                
                                # Get level value if available
                                level_val = level_values[i] if i < len(level_values) else i
                                print(f"    Level {level_val} (index {i}): {valid_count:,} valid values")
                            except Exception as e:
                                valid_counts.append(0)
                                print(f"    Level {i}: Error - {str(e)}")
                        
                        # Find level with maximum valid data
                        best_level = np.argmax(valid_counts)
                        best_count = valid_counts[best_level]
                        best_value = level_values[best_level] if best_level < len(level_values) else best_level
                        
                        if best_count > 0:
                            sel_dict[dim] = best_level
                            print(f"  ✓ Selected {dim}=index {best_level} (value: {best_value}) with {best_count:,} valid values")
                        else:
                            sel_dict[dim] = 0
                            print(f"  ⚠ No valid data found in any {dim}, using index 0")
                            
                    elif dim in ['lat', 'latitude', 'y', 'south_north']:
                        sel_dict[dim] = 0
                    elif dim in ['lon', 'longitude', 'x', 'west_east']:
                        sel_dict[dim] = 0
                    else:
                        # For any other dimension, default to 0
                        sel_dict[dim] = 0
                        print(f"  Using default index 0 for dimension: {dim}")
            
            try:
                ws = ws.isel(**sel_dict)
                print(f"  ✓ Final selection: {sel_dict}")
            except Exception as e:
                print(f"  ⚠ Could not select point: {str(e)}")
                # Fallback: take first index of all non-time dims
                try:
                    ws = ws.isel({dim: 0 for dim in ws.dims if dim != time_dim})
                    print(f"  ✓ Fallback: selected first index of all dims")
                except:
                    print(f"  ✗ Could not reduce dimensions")
                    y_pos += 1
                    continue
        
        # Rename time dimension to 'time' for resample
        if time_dim != 'time':
            try:
                ws = ws.rename({time_dim: 'time'})
                print(f"  ✓ Renamed '{time_dim}' to 'time'")
            except:
                pass
        
        # Ensure we have time coordinate
        if 'time' not in ws.coords and 'time' in ds.coords:
            ws = ws.assign_coords(time=ds.time)
        
        # Resample to daily
        try:
            ws_daily = ws.resample(time='1D').mean()
            print(f"  ✓ Resampled to daily")
        except Exception as e:
            print(f"  ⚠ Resample failed: {str(e)}")
            # Try without resample
            ws_daily = ws
            if 'time' in ws.dims:
                print(f"  ✓ Using original time resolution")
            else:
                print(f"  ✗ No time dimension after processing")
                y_pos += 1
                continue
        
        # Create availability mask
        try:
            available = ~np.isnan(ws_daily)
            
            # Handle dask arrays
            if hasattr(available, 'compute'):
                available_values = available.compute()
            else:
                available_values = available.values
            
            # Get time values
            if 'time' in ws_daily.coords:
                times = pd.to_datetime(ws_daily.time.values)
            else:
                print(f"  ⚠ No time coordinate in processed data")
                y_pos += 1
                continue
            
            # Store time range for this station
            all_time_ranges.append({
                'station': station_name,
                'start': times[0],
                'end': times[-1],
                'y_pos': y_pos
            })
            
            # Calculate overall availability
            total_days = len(available_values)
            available_days = np.sum(available_values)
            avail_pct = (available_days / total_days * 100) if total_days > 0 else 0
            avail_percentages.append(avail_pct)
            
            print(f"  ✓ Time range: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
            print(f"  ✓ {available_days:,} of {total_days:,} days ({avail_pct:.1f}%)")
            
            # Find continuous data segments
            in_segment = False
            segment_start = None
            
            for i, has_data in enumerate(available_values):
                if has_data and not in_segment:
                    # Start of a data segment
                    in_segment = True
                    segment_start = times[i]
                elif not has_data and in_segment:
                    # End of a data segment
                    in_segment = False
                    # Plot horizontal line for this segment
                    ax.hlines(y=y_pos, xmin=segment_start, xmax=times[i-1], 
                             color='#2E86AB', linewidth=3, alpha=0.8)
            
            # Handle if data goes to the end
            if in_segment:
                ax.hlines(y=y_pos, xmin=segment_start, xmax=times[-1], 
                         color='#2E86AB', linewidth=3, alpha=0.8)
            
            stations.append(station_name)
            y_pos += 1
            
        except Exception as e:
            print(f"  ✗ Error processing availability: {str(e)}")
            y_pos += 1
            continue
        
    except Exception as e:
        print(f"  ✗ Unexpected error: {str(e)}")
        failed_files.append(station_name)
        y_pos += 1
        continue

# ============================================================================
# SET X-AXIS LIMITS TO EARLIEST AVAILABLE DATA
# ============================================================================

if all_time_ranges:
    # Find the earliest start date and latest end date across all stations
    earliest_start = min([tr['start'] for tr in all_time_ranges])
    latest_end = max([tr['end'] for tr in all_time_ranges])
    
    # Set x-axis limits
    ax.set_xlim(earliest_start, latest_end)
     
    print(f"\n=== PLOT RANGE ===")
    print(f"  Showing data from: {earliest_start.strftime('%Y-%m-%d')}")
    print(f"  to: {latest_end.strftime('%Y-%m-%d')}")
    
    # # Optional: Add a vertical line at earliest start for emphasis
    # ax.axvline(x=earliest_start, color='red', linestyle='--', alpha=0.3, linewidth=1)
else:
    print("\n⚠ No valid time ranges found to set plot limits")

# ============================================================================
# FORMATTING AND SAVING
# ============================================================================

# Formatting
ax.set_xlabel('Year', fontsize=16, fontweight='bold')
ax.set_ylabel('Station', fontsize=16, fontweight='bold')
ax.set_title('Wind Speed Data Availability by Station\n(Blue segments = data available)', 
            fontsize=20, fontweight='bold', pad=20)

# Set y-axis
if stations:
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations, fontsize=14)
    ax.set_ylim(-0.5, len(stations) - 0.5)

# Format x-axis to show years
ax.set_xlim(pd.Timestamp('2000-01-01'), ax.get_xlim()[1])
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.grid(True, axis='x', alpha=0.2, linestyle='--')

# Add availability percentages as text at the end of each line
if stations and avail_percentages:
    for i, (station, avail_pct) in enumerate(zip(stations, avail_percentages)):
        ax.text(ax.get_xlim()[1] + 30, i, f'{avail_pct:.1f}%', 
               va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add notes about stations with issues
if time_issues:
    note_text = f"⚠ Stations with time dimension issues: {', '.join(time_issues)}"
    ax.text(0.02, -0.15, note_text, transform=ax.transAxes, 
           fontsize=8, style='italic', color='red')

# # Add variable name mapping
# if used_variables:
#     var_text = "Wind variables: " + ", ".join([f"{s[:8]}...:{v}" for s, v in used_variables[:3]])
#     if len(used_variables) > 3:
#         var_text += f" and {len(used_variables)-3} more"
#     y_pos = -0.20 if time_issues else -0.15
#     ax.text(0.02, y_pos, var_text, transform=ax.transAxes, 
#            fontsize=8, style='italic', color='gray')

# Adjust layout
plt.tight_layout()

# Save
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ Plot saved: {OUTPUT_FILE}")

# Show summary
print("\n=== SUMMARY ===")
if stations and avail_percentages:
    for station, avail_pct in zip(stations, avail_percentages):
        print(f"  ✓ {station}: {avail_pct:.1f}%")
    
    print("\n=== VARIABLES USED ===")
    for station, var in used_variables:
        print(f"  {station}: {var}")

if time_issues:
    print("\n=== TIME ISSUES ===")
    for station in time_issues:
        print(f"  ⚠ {station}: Time dimension not found")

if failed_files:
    print("\n=== FAILED FILES ===")
    for station in failed_files:
        print(f"  ✗ {station}: Could not process")

plt.show()




# ============================================================================
# ADD TABLE PLOT - NUMBER OF DATA POINTS PER HEIGHT
# ============================================================================

print("\n=== CREATING HEIGHT DATA TABLE ===\n")

# Create a dictionary to store height data for each station
height_data = {}

for station_name, file_path in NC_FILES.items():
    print(f"\nProcessing heights for: {station_name}")
    
    try:
        if not Path(file_path).exists():
            print(f"  ⚠ File not found")
            continue
        
        ds = xr.open_dataset(file_path)
        
        # Find time dimension
        time_dim = find_time_dimension(ds)
        if time_dim is None:
            print(f"  ⚠ No time dimension found")
            ds.close()
            continue
        
        # Find wind variable
        wind_var = find_wind_variable(ds)
        if wind_var is None:
            print(f"  ⚠ No wind variable found")
            ds.close()
            continue
        
        ws = ds[wind_var]
        
        # Check for height/level dimensions
        height_dims = []
        for dim in ws.dims:
            if dim in ['height', 'level', 'altitude', 'lev', 'z', 'lev_wspd', 
                      'lev_wdir', 'lev_tair', 'lev_rhum', 'plev', 'pressure']:
                height_dims.append(dim)
        
        if not height_dims:
            print(f"  ⚠ No height dimension found")
            ds.close()
            continue
        
        # Process each height dimension
        for height_dim in height_dims:
            # Get height values
            if height_dim in ds.coords:
                heights = ds[height_dim].values
            else:
                # If not in coords, use indices as heights
                heights = range(ws.sizes[height_dim])
            
            # Get time dimension values
            if time_dim in ws.dims:
                # Count valid data points per height
                data_counts = []
                valid_heights = []
                
                for i in range(ws.sizes[height_dim]):
                    try:
                        # Select this height level
                        height_slice = ws.isel({height_dim: i})
                        
                        # Count non-NaN values
                        if hasattr(height_slice, 'count'):
                            if hasattr(height_slice, 'compute'):
                                count = height_slice.count().compute()
                            else:
                                count = height_slice.count().values
                            
                            # Get height value (convert to float for consistency)
                            height_val = float(heights[i]) if i < len(heights) else float(i)
                            
                            # Convert count to integer
                            if hasattr(count, 'values'):
                                count = int(count.values)
                            else:
                                count = int(count)
                            
                            # Store if there's any data
                            if count > 0:
                                data_counts.append(count)
                                valid_heights.append(height_val)
                                print(f"    Height {height_val:.1f}m: {count:,} data points")
                    except Exception as e:
                        print(f"    Height {i}: Error - {str(e)}")
                        continue
                
                # Store the data for this station
                if valid_heights:
                    height_data[station_name] = {
                        'heights': valid_heights,
                        'counts': data_counts,
                        'height_dim': height_dim
                    }
                    print(f"  ✓ Found {len(valid_heights)} height levels with data")
                else:
                    print(f"  ⚠ No valid data found at any height")
            else:
                print(f"  ⚠ Time dimension '{time_dim}' not found in wind variable dimensions")
        
        ds.close()
        
    except Exception as e:
        print(f"  ✗ Error processing heights: {str(e)}")
        continue

# ============================================================================
# CREATE TABLE PLOT - HEIGHTS ON Y-AXIS, STATIONS ON X-AXIS
# ============================================================================

if height_data:
    print("\n=== GENERATING HEIGHT TABLE ===\n")
    
    # Find all unique heights across all stations
    all_heights = set()
    for station_info in height_data.values():
        all_heights.update(station_info['heights'])
    all_heights = sorted(list(all_heights))
    
    # Get stations in the same order as the original plot
    ordered_stations = stations if stations else list(height_data.keys())
    
    # Create a mapping from height to row index (heights on y-axis)
    height_to_row = {h: i for i, h in enumerate(all_heights)}
    
    # Prepare data for table (rows = heights, columns = stations)
    table_data = []
    row_labels = []  # Heights
    
    for height in all_heights:
        row_data = []
        for station_name in ordered_stations:
            if station_name in height_data:
                # Find the count for this height
                station_info = height_data[station_name]
                height_list = station_info['heights']
                count_list = station_info['counts']
                
                # Find the index of this height
                if height in height_list:
                    idx = height_list.index(height)
                    count = count_list[idx]
                    row_data.append(count)
                else:
                    row_data.append(0)
            else:
                row_data.append(0)
        
        table_data.append(row_data)
        row_labels.append(f"{height:.0f}m")
    
    # REMOVE HEIGHTS WITH ALL ZERO DATA
    rows_to_keep = []
    for i, row in enumerate(table_data):
        if sum(row) > 0:  # Keep if there's any data in this row
            rows_to_keep.append(i)
    
    # Filter table_data and row_labels
    table_data = [table_data[i] for i in rows_to_keep]
    row_labels = [row_labels[i] for i in rows_to_keep]
    
    # SORT HEIGHTS IN ASCENDING ORDER (lowest at bottom)
    # Create pairs of (height_value, row_data, label)
    height_pairs = []
    for i, (label, row) in enumerate(zip(row_labels, table_data)):
        # Extract numeric height from label (e.g., "100m" -> 100)
        height_val = float(label.replace('m', ''))
        height_pairs.append((height_val, row, label))
    
    # Sort by height (ascending)
    height_pairs.sort(key=lambda x: x[0])
    
    # Unpack sorted data
    table_data = [pair[1] for pair in height_pairs]
    row_labels = [pair[2] for pair in height_pairs]
    
    # Now row_labels are in ascending order (lowest at bottom)
    
    # Determine figure size based on number of stations and heights
    # Make it wider if there are many stations, taller if many heights
    fig_width = max(14, len(ordered_stations) * 0.7)
    fig_height = max(8, len(row_labels) * 0.5)
    
    # Create figure for table
    fig_table, ax_table = plt.subplots(figsize=(fig_width, fig_height))
    
    # Hide axes
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create column labels (station names)
    col_labels = ordered_stations
    
    # Calculate column widths based on station name lengths
    # Make columns wider for longer names
    col_widths = []
    for station in col_labels:
        # Base width, adjusted for name length
        width = max(0.06, min(0.15, 0.06 + len(station) * 0.005))
        col_widths.append(width)
    
    # Ensure total width doesn't exceed reasonable bounds
    total_width = sum(col_widths)
    if total_width > 1.0:
        # Scale down if too wide
        scale_factor = 0.9 / total_width
        col_widths = [w * scale_factor for w in col_widths]
    
    # Create the table
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code the cells based on data availability
    for i, row in enumerate(table_data):
        for j, value in enumerate(row):
            cell = table[(i+1, j)]  # +1 because row 0 is headers
            if value == 0:
                cell.set_facecolor('#ffcccc')  # Light red for no data
            elif value < 1000:
                cell.set_facecolor('#ffffcc')  # Light yellow for sparse data
            elif value < 10000:
                cell.set_facecolor('#ccffcc')  # Light green for moderate data
            else:
                cell.set_facecolor('#99cc99')  # Darker green for abundant data
            
            # Format large numbers with commas
            if value > 0:
                cell.set_text_props(text=f"{value:,}")
    
    # Style header row (stations)
    for j in range(len(ordered_stations)):
        cell = table[(0, j)]
        cell.set_facecolor('#2E86AB')  # Match the blue color from the plot
        cell.set_text_props(weight='bold', color='white', fontsize=8)
        
        # Wrap long station names for better fit
        station_name = col_labels[j]
        # Split by underscores and join with newlines
        if len(station_name) > 12:
            # Try to split at underscores
            parts = station_name.split('_')
            if len(parts) > 1:
                # Put each part on a new line, but group small parts
                wrapped_name = '\n'.join(parts)
                cell.get_text().set_text(wrapped_name)
            else:
                # If no underscores, wrap every 12 characters
                wrapped_name = '\n'.join([station_name[i:i+12] for i in range(0, len(station_name), 12)])
                cell.get_text().set_text(wrapped_name)
        else:
            cell.get_text().set_text(station_name)
        
        # Adjust cell height for wrapped text
        cell.set_height(0.08)  # Slightly taller for wrapped names
    
    # Style row labels (heights) - now in ascending order (lowest at bottom)
    for i in range(len(row_labels)):
        cell = table[(i+1, -1)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Add title
    ax_table.set_title('Number of Data Points by Height at Each Location', 
                      fontsize=16, fontweight='bold', pad=20)
    
    # Add a legend
    legend_elements = [
        '■ No data (0 points)',
        '■ Sparse data (< 1,000 points)',
        '■ Moderate data (1,000 - 10,000 points)',
        '■ Abundant data (> 10,000 points)'
    ]
    
    legend_text = '\n'.join(legend_elements)
    
    # Position legend in top-right corner
    ax_table.text(0.98, 0.98, 'Color Legend:', transform=ax_table.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 weight='bold')
    
    ax_table.text(0.98, 0.88, legend_text, transform=ax_table.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 linespacing=1.5,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add station count summary at bottom
    summary_text = []
    for station_name in ordered_stations:
        if station_name in height_data:
            total_points = sum(height_data[station_name]['counts'])
            # Shorten station name for summary if needed
            short_name = station_name[:20] + '...' if len(station_name) > 20 else station_name
            summary_text.append(f"{short_name}: {total_points:,}")
    
    if summary_text:
        # Show all summary items, split into multiple lines if needed
        summary_lines = []
        current_line = []
        for item in summary_text:
            if len(' | '.join(current_line + [item])) > 150:  # Limit line length
                summary_lines.append(' | '.join(current_line))
                current_line = [item]
            else:
                current_line.append(item)
        if current_line:
            summary_lines.append(' | '.join(current_line))
        
        # Add summary text at bottom
        y_pos = -0.08
        for line in summary_lines:
            ax_table.text(0.5, y_pos, line, 
                         transform=ax_table.transAxes,
                         fontsize=7, ha='center', va='top',
                         style='italic', color='gray')
            y_pos -= 0.04
    
    # Add information about removed heights
    removed_count = len(all_heights) - len(row_labels)
    if removed_count > 0:
        ax_table.text(0.02, -0.05, f"Removed {removed_count} height levels with no data", 
                     transform=ax_table.transAxes,
                     fontsize=8, style='italic', color='gray', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the table
    table_output = 'height_data_table.png'
    plt.savefig(table_output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Height table saved: {table_output}")
    
    # Print summary
    print("\n=== HEIGHT DATA SUMMARY ===")
    print(f"  Total unique heights found: {len(all_heights)}")
    print(f"  Heights with data: {len(row_labels)}")
    print(f"  Removed heights with no data: {len(all_heights) - len(row_labels)}")
    print(f"  Height range: {row_labels[0]} to {row_labels[-1]}")
    
    print("\n  Per station:")
    for station_name in ordered_stations:
        if station_name in height_data:
            station_info = height_data[station_name]
            total_points = sum(station_info['counts'])
            height_range = f"{min(station_info['heights']):.0f}m - {max(station_info['heights']):.0f}m"
            print(f"    {station_name}: {total_points:,} total points across {len(station_info['heights'])} heights ({height_range})")
        else:
            print(f"    {station_name}: No height data available")
    
    plt.show()
    
else:
    print("\n⚠ No height data found for any station")
    
    # Create empty figure with message
    fig_table, ax_table = plt.subplots(figsize=(10, 6))
    ax_table.axis('off')
    ax_table.text(0.5, 0.5, 'No Height Data Available\nfor Any Station', 
                 fontsize=20, ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    
    table_output = 'height_data_table.png'
    plt.savefig(table_output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Empty table saved: {table_output}")
    plt.show()