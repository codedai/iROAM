import re
import os
import logging

import pandas as pd
import numpy as np

from tqdm import tqdm


final_column_order = [
    # Trip / Route / Vehicle
    "trip_id", "fleet", "route",
    # Time
    "datetime", "time_offset_seconds", "relative_position",
    # Travel & Speed
    "travel_distance_m", "moving_speed_m_s",
    # Direction detection
    "direction_step", "multi_direction_trip",
    # Load information
    "load"
]


def parse_distance_to_meters(dist_str):
    """
    Converts a distance string that may end with 'm' or 'km'
    into a numeric (float) value in meters.
    
    Examples:
      '120m'  -> 120.0
      '3.4km' -> 3400.0
      '  240 ' (no suffix) -> 240.0
      'NaN', None, '' -> np.nan
    """
    if pd.isna(dist_str) or not isinstance(dist_str, str) or dist_str.strip() == '':
        return np.nan
    
    dist_str = dist_str.strip().lower()
    
    if dist_str.endswith('km'):
        # Remove 'km' suffix
        val_str = dist_str[:-2]  # everything except the last 2 chars
        return float(val_str) * 1000.0
    elif dist_str.endswith('m'):
        # Remove 'm' suffix
        val_str = dist_str[:-1]  # everything except the last char
        return float(val_str)
    else:
        # No recognized suffix -> assume it's already meters
        return float(dist_str)



def convert_time_offset_to_seconds(df: pd.DataFrame,
                                   col_in='time_offset',
                                   col_out='time_offset_seconds') -> pd.DataFrame:
    """
    Convert a time offset string to an integer number of seconds.
    
    Expected formats:
      - "mm:ss" (e.g., "1:46" -> 106 seconds)
      - "hh:mm:ss" (e.g., "1:01:46" -> 3706 seconds)
    
    If the value is NaN or empty, we set it to NaN in the output.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col_in : str
        Name of the existing time offset column.
    col_out : str
        Name of the new column with integer seconds.
    
    Returns
    -------
    pd.DataFrame
        The original DataFrame with a new column `col_out` containing 
        the integer offset in seconds.
    """
    def parse_offset_to_seconds(offset_str):
        if pd.isna(offset_str) or offset_str == '':
            return np.nan
        
        parts = offset_str.split(':')
        if len(parts) == 2:
            # Format: mm:ss
            mm, ss = parts
            return int(mm)*60 + int(ss)
        elif len(parts) == 3:
            # Format: hh:mm:ss
            hh, mm, ss = parts
            return int(hh)*3600 + int(mm)*60 + int(ss)
        else:
            # Unrecognized format; return NaN or raise an error
            return np.nan
    
    df[col_out] = df[col_in].apply(parse_offset_to_seconds).astype('Int64')  # or just int if you're sure no NaNs
    return df



def upsample_df(df: pd.DataFrame, resolution_seconds: int) -> pd.DataFrame:
    """
    Upsample a DataFrame to a higher temporal resolution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'datetime' (sorted ascending)
          - 'travel_distance_m'
          - 'moving_speed_m_s' (the speed from the *next* row).
        All other columns will be carried over or duplicated from the current row.
    resolution_seconds : int
        Desired time resolution in seconds (e.g., 10 means we add rows every 10s).
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame, sorted by datetime, with additional rows inserted at the
        specified resolution. Each new row has an interpolated (or extrapolated)
        travel_distance_m based on (time delta) * moving_speed_m_s(next).
    """
    # Ensure the DataFrame is sorted by datetime
    # df = df.sort_values('datetime').reset_index(drop=True)
    
    # List to collect the upsampled rows
    if len(df) < 2:
        return None
    rows = []
    
    # Go row by row
    for i in range(len(df) - 1):
        current_row = df.iloc[i].copy()
        next_row = df.iloc[i+1].copy()
        
        # Always append the "current" row to the result
        # rows.append(current_row)
        
        # Extract times
        t_current = current_row['datetime']
        t_next = next_row['datetime']

        t_current_travel_distance = current_row['travel_distance_m']
        t_next_travel_distance = next_row['travel_distance_m']

        middle_travel_distance = (t_current_travel_distance + t_next_travel_distance) / 2

        t_next_moving_speed = next_row['moving_speed_m_s']
        
        # Compute the time range between current and next
        total_delta = (t_next - t_current).total_seconds()
        if total_delta <= 0:
            # No positive gap or times are the same => no interpolation needed
            continue
        
        # We'll insert new rows at multiples of resolution_seconds after t_current,
        # up until (but not including) t_next.
        # e.g. if t_current=08:41:48, resolution=10 => next boundary is 08:41:50, 08:42:00, ...
        
        # 1) Find the first "resolution boundary" after t_current
        #    For example, if we want multiples of 10s, we can do:
        #    round down t_current to nearest 10s, then add resolution until we pass t_current.
        
        # Convert to integer seconds since epoch for easier rounding
        epoch_current = int(t_current.timestamp())
        # The next multiple of 'resolution_seconds' after 'epoch_current'
        # can be found by: ((epoch_current // resolution_seconds) + 1) * resolution_seconds
        # *But* if epoch_current is already on a boundary, we need to make sure to skip to the next boundary.
        
        first_boundary = (epoch_current // resolution_seconds) * resolution_seconds
        if first_boundary < epoch_current:
            # Make sure it's strictly after the current time
            first_boundary += resolution_seconds
        
        # Possibly we've overshot, so ensure it's strictly > epoch_current
        while first_boundary < epoch_current:
            first_boundary += resolution_seconds
        
        # 2) Now create new time points at intervals of resolution_seconds until we reach t_next
        candidate = first_boundary
        epoch_next = int(t_next.timestamp())
        
        while candidate < epoch_next:
            t_candidate = pd.to_datetime(candidate, unit='s')
            
            # Now we calculate the partial delta (seconds) from t_current to t_candidate
            partial_delta = (t_candidate - t_current).total_seconds()
            
            # According to your specification:
            # travel_distance_m(new) = travel_distance_m(current_row)
            #                        + (partial_delta * moving_speed_m_s(next_row))
            # because we are using the *next row's* speed.
            dist_candidate = t_current_travel_distance + (partial_delta * t_next_moving_speed)
            
            # Create a new row with the same columns as the current row
            # and override datetime, travel_distance_m, etc.
            if dist_candidate < middle_travel_distance:
                new_row = current_row.copy()
            else:
                new_row = next_row.copy()

            new_row['moving_speed_m_s'] = t_next_moving_speed
            new_row['datetime'] = t_candidate
            new_row['travel_distance_m'] = dist_candidate
            
            # If you want to also update speed or other columns,
            # you could do additional logic here. For now, we keep them from current_row
            # or next_row as needed.
            
            rows.append(new_row)
            
            candidate += resolution_seconds
    
    # Finally, append the last row from the original DataFrame
    # (since the for loop goes until len(df)-2 -> i+1, we need to ensure the last row is included)
    # if len(df) > 0:
    #     rows.append(df.iloc[-1])
    
    # Convert our list of rows into a DataFrame and sort by datetime
    try:
        upsampled_df = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    except:
        pd.DataFrame(rows).to_csv("error.csv")
        df.to_csv("error_original.csv")
        raise ValueError("Error in upsample_df")
    
    return upsampled_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_columns = [col for col in final_column_order if col in df.columns]
    return df[existing_columns]

def last_step_clean_up(df: pd.DataFrame) -> pd.DataFrame:
    df = reorder_columns(convert_time_offset_to_seconds(df))
    # round to 2 decimal places
    df['travel_distance_m'] = df['travel_distance_m'].round(2)
    df['moving_speed_m_s'] = df['moving_speed_m_s'].round(2)
    return df

def clean_and_transform_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms the DataFrame (parsing datetime, distance, speed, etc.)
    and additionally computes direction of travel. Also identifies if there's a 
    direction switch within the trip and splits into two sub-DataFrames if needed.
    """
    # 1) (Assume you've already done your typical cleaning steps here)
    # For illustration, let's suppose 'df' is already cleaned and sorted by 'datetime'.
    # e.g. df = clean_and_transform(df) from the previous snippet.
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 2) Compute direction by comparing the current travel_distance_m with the *next* row
    # We'll shift the column forward by 1 row so we can do row-by-row comparison.
    travel_dist_next = df['travel_distance_m'].shift(-1)
    travel_dist_curr = df['travel_distance_m']
    
    # direction: +1 if next distance > current distance, else -1
    # For the very last row, direction might be NaN since there's no "next" row
    df['direction_step'] = np.where(travel_dist_next >= travel_dist_curr, +1, -1)
    df['direction_step'] = np.where(travel_dist_next == travel_dist_curr, df['direction_step'].shift(1), df['direction_step'])


    # Take care of the last row, assuming it's the same direction as the previous row
    df['direction_step'].iloc[-1] = df['direction_step'].iloc[-2]
    
    # 3) Detect if there's a sign change in 'direction_step'.
    # The simplest approach: compare direction_step[i] and direction_step[i+1].
    direction_next = df['direction_step'].shift(-1)
    df['direction_change'] = (df['direction_step'] != direction_next)  # True if sign changes
    # when the travel distance is the same, we assume the direction is the same
    df['direction_change'] = np.where(travel_dist_next == travel_dist_curr, False, df['direction_change'])
    df['direction_change'].iloc[-1] = False  # Last row can't have a change
    
    # 4) If there's exactly one sign change, find that index
    change_indices = df.index[df['direction_change'] == True].tolist()
    
    if len(change_indices) == 0:
        # No direction change
        df['multi_direction_trip'] = False
        # Then the entire data is just one direction
        # You can save df as a single file or keep as is
        df_part1 = df
        df_part2 = None
        
    else:
        # There's at least one sign change
        # For simplicity, we handle only the first change
        first_change_idx = change_indices[0]
        df['multi_direction_trip'] = True
        
        # 5) Split into two DataFrames:
        # part 1 = rows from start up to the change
        # part 2 = rows after the change
        df_part1 = df.loc[:first_change_idx].copy()
        df_part2 = df.loc[first_change_idx+1:].copy()
        
        # Optionally, if you suspect multiple sign changes, you could 
        # do additional logic or slicing. For now, we assume exactly two directions.
    
    # 6) Return either the full DataFrame (with the new columns) or the splits
    return df, df_part1, df_part2


def filter_with_time_limit(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """Filter DataFrame to rows within a time range."""
    # Parse start_time and end_time as datetime
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Filter rows within the time range, focusing on 'datetime' column, compare with no date information
    df_filtered = df[(df['datetime'].dt.time >= start_time.time()) & (df['datetime'].dt.time <= end_time.time())]
    return df_filtered

def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Parse out the date from 'route_date', if format is something like "29_2023-01-01" ---
    #     We'll store it in a new column called 'date_only'.
    #     This splits on '_' and grabs the second part (e.g. '2023-01-01').
    df['date_only'] = df['route_date'].str.split('_').str[-1]  # handle if route_date is always pattern <route>_<date>
    
    # --- 2) Convert time_val to a proper time. 
    #     Because time_val is like "9:09:17AM", "9:09:47AM", ...
    #     We'll first parse them as a time, then combine with date_only.
    
    # Combine date and time into a single datetime string, e.g. "2023-01-01 9:09:17AM"
    df['datetime_str'] = df['date_only'] + ' ' + df['time_val']
    # Now parse with format '%Y-%m-%d %I:%M:%S%p' 
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %I:%M:%S%p')
    
    # # --- 3) Clean numeric columns with units, e.g. 'distance' = '136m', 'speed' = '16.9km/h' ---
    # # Distance: remove trailing 'm', then convert to integer
    # if 'distance' in df.columns:
    #     df['distance'] = (df['distance'].apply(parse_distance_to_meters).astype('Int64'))  # or leave as float if you prefer
    
    # # Speed: remove trailing 'km/h', convert to float or int
    # if 'speed' in df.columns:
    #     df['speed'] = (df['speed'].str.replace('km/h', '', regex=False)
    #                               .astype(float)
    #                    # .astype('Int64') if you strictly want integer 
    #                   )
    
    # Some columns, like 'shortest_distance_m' and 'travel_distance_m', might already be numeric, 
    # but let's ensure:
    for col in ['shortest_distance_m', 'travel_distance_m']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- 4) Compute the moving speed from travel_distance_m and datetime ---
    # We do diff on travel_distance_m to get distance increments
    # We do diff on datetime to get time increments in seconds
    if 'travel_distance_m' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
        df['distance_diff_m'] = df['travel_distance_m'].diff()  # difference in meters
        df['time_diff_s'] = df['datetime'].diff().dt.total_seconds()  # difference in seconds
        # Speed in m/s
        df['moving_speed_m_s'] = df['distance_diff_m'] / df['time_diff_s']
        # # If you prefer km/h, multiply by 3.6
        # df['moving_speed_km_h'] = df['moving_speed_m_s'] * 3.6
    
    # --- 5) Clean up columns you don’t need or reorder them (optional) ---
    # For example, we can drop 'datetime_str' and 'date_only', 
    # or keep them if you want for debugging.
    df.drop(columns=['datetime_str', 'date_only', 'time_val', 'route_date', 'index_right', 'store_path'], errors='ignore', inplace=True)
    
    return df

def process_folder(source_folder, upsample_resolution=10):
    """Process all trip files in a source folder."""
    logging.info("Starting processing pipeline...")

    if not os.path.isdir(source_folder):
        logging.error(f"Source folder '{source_folder}' not found or not a directory.")
        return pd.DataFrame()
    for route_folder in tqdm(os.listdir(source_folder), desc="Routes"):
        route_path = os.path.join(source_folder, route_folder)
        if not os.path.isdir(route_path):
            continue

        for date_folder in tqdm(os.listdir(route_path), desc="Dates"):
            date_path = os.path.join(route_path, date_folder)
            df_list_pos = []
            df_list_neg = []
            if not os.path.isdir(date_path):
                continue

            for trip_folder in os.listdir(date_path):
                if trip_folder.endswith('csv'):
                    continue
                for trip_file in os.listdir(os.path.join(date_path, trip_folder)):
                    if trip_file.endswith('_final.csv'):
                        trip_path = os.path.join(os.path.join(date_path, trip_folder), trip_file)
                        df_raw = pd.read_csv(trip_path)
                        df_cleaned = clean_and_transform(df_raw)
                        df_cleaned = filter_with_time_limit(df_cleaned, "08:00:00", "23:59:59")
                        if df_cleaned.empty or df_cleaned.shape[0] < 2:
                            continue
                        _, df_part1, df_part2 = clean_and_transform_direction(df_cleaned)

                        df_part1 = upsample_df(df_part1, upsample_resolution)
                        if df_part1 is not None:
                            df_part1 = last_step_clean_up(df_part1)
                            if df_part1['direction_step'].iloc[-1] == 1:
                                df_list_pos.append(df_part1)
                            else:
                                df_list_neg.append(df_part1)
                        if df_part2 is not None:
                            df_part2 = upsample_df(df_part2, upsample_resolution)
                            if df_part2 is not None:
                                df_part2 = last_step_clean_up(df_part2)
                                if df_part2['direction_step'].iloc[-1] == 1:
                                    df_list_pos.append(df_part2)
                                else:
                                    df_list_neg.append(df_part2)
            if len(df_list_pos) > 0:
                df_pos = pd.concat(df_list_pos)
                df_pos.to_csv(os.path.join(date_path, f"{route_folder}_{date_folder}_pos.csv"), index=False)
            if len(df_list_neg) > 0:
                df_neg = pd.concat(df_list_neg)
                df_neg.to_csv(os.path.join(date_path, f"{route_folder}_{date_folder}_neg.csv"), index=False)
                        # Do the processing on trip_gdf
                        # e.g. filtered_gdf = filter_trip(trip_gdf)
                        # save_filtered_trip(filtered_gdf, output_folder, trip_file)
                        # plot_trip(shape_gdf, buffer_gdf, trip_gdf, filtered_gdf, output_folder, trip_file)

if __name__ == "__main__":
    upsample_resolution = 10  # seconds

    # # Example: reading one file
    # df_raw = pd.read_csv("./filtered/29/2023-01-01/765846299/765846299_final.csv")

    # df_cleaned = clean_and_transform(df_raw)

    # print(df_cleaned.head(10))

    # # df_cleaned = filter_with_time_limit(df_cleaned, "08:00:00", "23:59:59")

    # df_result, df_part1, df_part2 = clean_and_transform_direction(df_cleaned)

    # print(df_part1.head(10))
    # print(df_part1.tail(10).columns)

    # last_step_clean_up(upsample_df(df_part1, upsample_resolution)).to_csv("part1_sample.csv", index=False)
    # # df_cleaned = reorder_columns(df_cleaned)


    # df_part1.to_csv("part1.csv", index=False)

    process_folder("./filtered", upsample_resolution=upsample_resolution)
