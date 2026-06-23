#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

data_path = "archeology/dataset/"

radioCarbonPath = data_path + "radiocarbon_database_regional.csv"
climateMeasurementsPath = data_path + "climateMeasurements.csv"

# Load dataframes
radioCarbonDF = pd.read_csv(radioCarbonPath)
climateDF = pd.read_csv(climateMeasurementsPath)

# Clean radioCarbonDF
radioCarbonDF = radioCarbonDF.dropna(how='all').dropna(axis=1, how='all')

# FIX 1: Clean commas from the date string and convert to numeric before subtraction
radioCarbonDF["date"] = pd.to_numeric(radioCarbonDF["date"].astype(str).str.replace(',', ''), errors='coerce')
radioCarbonDF["year"] = 1950 - radioCarbonDF["date"]

# Clean climateDF
climateDF = climateDF.dropna(how='all').dropna(axis=1, how='all')
climateDF["year"] = 1950 - climateDF["Age_ky.1"].round(0) * 1000


def find_closest_year(row):
    year = row["year"]
    
    # Isolate boundary hits to prevent iloc[0] IndexError on empty dataframes
    previous_years_df = climateDF[climateDF["year"] <= year].sort_values("year", ascending=False)
    next_years_df = climateDF[climateDF["year"] >= year].sort_values("year", ascending=True)
    
    if previous_years_df.empty or next_years_df.empty:
        return np.nan  # Return NaN if the requested year falls outside our climate data range

    previous_year = previous_years_df.iloc[0]
    next_year = next_years_df.iloc[0]

    # FIX 2: Use the exact CSV column name "K (ppm)" instead of "K"
    if previous_year["year"] == year:
        return previous_year["K (ppm)"]
    if next_year["year"] == year:
        return next_year["K (ppm)"]

    last_K = previous_year["K (ppm)"]
    next_K = next_year["K (ppm)"]

    # Linear interpolation formula
    interpolation = (year - previous_year["year"]) / (next_year["year"] - previous_year["year"])
    K = last_K + interpolation * (next_K - last_K)
    return K


# Filter for Malta
maltaDF = radioCarbonDF[radioCarbonDF["Region"] == "Malta"].dropna(subset=["year"])

# FIX 3: Ensure Malta data exists and safely convert boundaries to integers
if maltaDF.empty:
    print("Error: No valid radiocarbon data found for the 'Malta' region.")
else:
    min_year = int(maltaDF["year"].min())
    max_year = int(maltaDF["year"].max())
    
    KValuesForHumans = []
    for i in range(min_year, max_year + 1): # Added +1 so the maximum year is inclusive
        KValue = find_closest_year({"year": i})
        if pd.notna(KValue): # Ignore any years that fell out-of-bounds
            KValuesForHumans.append(KValue)

    if KValuesForHumans:
        mean_k = sum(KValuesForHumans) / len(KValuesForHumans)
        print("Mean K value for humans in the area: ", round(mean_k, 4))
    else:
        print("Could not calculate mean K: All target years were out of the climate timeline's range.")
