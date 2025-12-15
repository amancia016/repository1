import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, time
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt

# Load model suite
model_suite = joblib.load("uber_fare_stacking_ensemble.pkl")
stacking_model = model_suite["stacking_model"]
base_models = model_suite["base_models"]
scaler = model_suite["scaler"]
features = model_suite["features"]
le_distance = model_suite["le_distance"]
le_time = model_suite["le_time"]

st.set_page_config(page_title="Uber Fare Predictor", layout="centered")

st.title("🚕 Uber Fare Predictor")

# Interactive Folium map
st.subheader("Select Pickup and Dropoff Locations")
st.markdown("🗺️ Click **two points** on the map below: First for Pickup, Second for Dropoff")

# Store clicked points in session state
if "points" not in st.session_state:
    st.session_state.points = []

# Initialize map centered on NYC
m = folium.Map(location=[40.75, -73.98], zoom_start=12)

# Add pickup/dropoff markers
for i, point in enumerate(st.session_state.points):
    label = "Pickup" if i == 0 else "Dropoff"
    folium.Marker(
        location=[point["lat"], point["lng"]],
        popup=label,
        icon=folium.Icon(color="green" if i == 0 else "red")
    ).add_to(m)

# Draw line between points
if len(st.session_state.points) == 2:
    folium.PolyLine(
        locations=[
            [st.session_state.points[0]["lat"], st.session_state.points[0]["lng"]],
            [st.session_state.points[1]["lat"], st.session_state.points[1]["lng"]]
        ],
        color="blue", weight=2.5
    ).add_to(m)

# Render the map and listen for clicks
map_data = st_folium(m, width=700, height=450, returned_objects=["last_clicked"], key="map", center=True)

# Capture clicks
if map_data and map_data["last_clicked"] is not None:
    if "points" not in st.session_state:
        st.session_state.points = []

    if len(st.session_state.points) < 2:
        st.session_state.points.append(map_data["last_clicked"])

# Reset map
if st.button("🔄 Reset Map"):
    st.session_state.points = []
    st.rerun()

# Proceed only if two points selected
if len(st.session_state.points) == 2:
    pickup = st.session_state.points[0]
    dropoff = st.session_state.points[1]

    pickup_lat = pickup["lat"]
    pickup_lon = pickup["lng"]
    dropoff_lat = dropoff["lat"]
    dropoff_lon = dropoff["lng"]

    with st.container():
        # Ride type selection
        col1, col2 = st.columns(2)
        with col1:
            ride_type = st.selectbox(
                "🚗 Ride Type", 
                ["UberX", "UberXL"],
                help="UberX seats up to 4 passengers, UberXL seats up to 6 passengers"
            )
        with col2:
            # Passenger count input (adjusted based on ride type)
            max_passengers = 6 if ride_type == "UberXL" else 4
            passenger_count = st.slider(
                "👥 Passenger Count", 
                1, max_passengers, 
                min(1, max_passengers),
                help=f"{ride_type} can accommodate up to {max_passengers} passengers"
            )

        # Trip datetime input
        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("📅 Trip Date", datetime.today().date())
        with col2:
            trip_time = st.time_input("⏰ Trip Time", datetime.now().time())

        trip_datetime = datetime.combine(trip_date, trip_time)

        # Haversine formula
        def haversine(lat1, lon1, lat2, lon2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            return 6371 * 2 * asin(sqrt(a))  # in km

        # Calculate features
        distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        duration_est = (distance_km / 18) * 60  # estimate in minutes

        hour = trip_datetime.hour
        day_of_week = trip_datetime.weekday()
        month = trip_datetime.month
        year = trip_datetime.year
        is_weekend = 1 if day_of_week >= 5 else 0

        # Custom categories
        def categorize_distance(distance):
            if distance <= 1:
                return 'Short (≤1km)'
            elif distance <= 3:
                return 'Medium (1-3km)'
            elif distance <= 10:
                return 'Long (3-10km)'
            else:
                return 'Very Long (>10km)'

        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 24:
                return 'Evening'
            else:
                return 'Night'

        # FIXED: Call functions with actual values and use transform (not fit_transform)
        distance_category = categorize_distance(distance_km)
        time_of_day = categorize_time(hour)
        
        distance_category_encoded = le_distance.transform([distance_category])[0]
        time_of_day_encoded = le_time.transform([time_of_day])[0]

        # Create input data with ALL 14 features in the correct order
        input_data = pd.DataFrame([[
            pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
            passenger_count, hour, day_of_week, month, year,
            distance_km, duration_est,
            distance_category_encoded, time_of_day_encoded, is_weekend
        ]], columns=features)

        # CORRECTED: Stacking ensemble prediction pipeline
        # Step 1: Get predictions from all base models
        base_predictions = []

        for name, model in base_models.items():
            if name in ['ElasticNet']:
                # Use scaled data for ElasticNet
                X_scaled = scaler.transform(input_data)
                pred = model.predict(X_scaled)[0]
            else:
                # Use raw data for tree-based models
                pred = model.predict(input_data)[0]
            
            base_predictions.append(pred)

        # Step 2: Use base model predictions as input to stacking model
        stacking_input = np.array(base_predictions).reshape(1, -1)
        raw_predicted_fare = stacking_model.predict(stacking_input)[0]

        # INFINITE FUTURE PREDICTION SYSTEM
        # Historical CPI data from US Bureau of Labor Statistics
        cpi_data = {
            2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594,
            2013: 232.957, 2014: 236.736, 2015: 237.017, 2016: 240.007,
            2017: 245.120, 2018: 251.107, 2019: 255.657, 2020: 258.811,
            2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689,
            2025: 319.799  # Latest available (March 2025)
        }
        
        # Federal Reserve projections for near-term (June 2025 FOMC)
        fed_inflation_projections = {
            2026: 2.4,  # PCE inflation projection
            2027: 2.1,  # PCE inflation projection
            'long_run': 2.0  # Long-run inflation target
        }
        
        def calculate_future_cpi(target_year):
            """Calculate CPI for any future year using Fed projections + long-term assumptions."""
            baseline_cpi = cpi_data[2012]  # Training data baseline
            
            if target_year <= 2025:
                # Use historical data
                return cpi_data.get(target_year, cpi_data[2025]), cpi_data[target_year] / baseline_cpi
            
            # For future years, project forward from 2025
            current_cpi = cpi_data[2025]
            
            # Use Fed projections for 2026-2027
            if target_year == 2026:
                projected_cpi = current_cpi * (1 + fed_inflation_projections[2026] / 100)
            elif target_year == 2027:
                # Apply 2026 and 2027 inflation rates cumulatively
                cpi_2026 = current_cpi * (1 + fed_inflation_projections[2026] / 100)
                projected_cpi = cpi_2026 * (1 + fed_inflation_projections[2027] / 100)
            else:
                # For 2028 and beyond, use Fed's long-run target (2.0%)
                # First get to 2027
                cpi_2026 = current_cpi * (1 + fed_inflation_projections[2026] / 100)
                cpi_2027 = cpi_2026 * (1 + fed_inflation_projections[2027] / 100)
                
                # Then apply 2% annually for remaining years
                years_beyond_2027 = target_year - 2027
                projected_cpi = cpi_2027 * ((1 + fed_inflation_projections['long_run'] / 100) ** years_beyond_2027)
            
            inflation_multiplier = projected_cpi / baseline_cpi
            return projected_cpi, inflation_multiplier
        
        # Model was trained on 2009-2015 data, use 2012 as baseline (midpoint)
        target_year = trip_datetime.year
        
        # Calculate inflation for any year (past, present, or future)
        projected_cpi, inflation_multiplier = calculate_future_cpi(target_year)
        
        # Apply inflation adjustment
        inflation_adjusted_fare = raw_predicted_fare * inflation_multiplier
        
        # 3. Add modern surcharges (only for years 2019 onwards when they were implemented)
        surcharges = 0
        
        # Check if trip involves Manhattan (rough bounds)
        manhattan_lat_min, manhattan_lat_max = 40.700, 40.800
        manhattan_lon_min, manhattan_lon_max = -74.020, -73.930
        
        pickup_in_manhattan = (manhattan_lat_min <= pickup_lat <= manhattan_lat_max and 
                              manhattan_lon_min <= pickup_lon <= manhattan_lon_max)
        dropoff_in_manhattan = (manhattan_lat_min <= dropoff_lat <= manhattan_lat_max and 
                               manhattan_lon_min <= dropoff_lon <= manhattan_lon_max)
        
        # Apply surcharges based on year (these didn't exist in training period)
        if target_year >= 2019:
            # Base improvement surcharge (started around 2019)
            surcharges += 1.00
            
        if target_year >= 2023:  # Congestion pricing implemented
            if pickup_in_manhattan or dropoff_in_manhattan:
                surcharges += 1.50  # NYC congestion pricing for Uber
        
        # Final adjusted fare
        base_predicted_fare = inflation_adjusted_fare + surcharges
        
        # Apply ride type multiplier
        ride_multipliers = {
            "UberX": 1.0,      # Base price (our model was trained on regular taxi data)
            "UberXL": 1.4      # About 40% more expensive than UberX
        }
        
        ride_multiplier = ride_multipliers[ride_type]
        predicted_fare = base_predicted_fare * ride_multiplier
        
        # Calculate inflation percentage for display
        inflation_percent = ((inflation_multiplier - 1) * 100)
        
        # Display results with future capability indicator
        if target_year > 2027:
            st.success(f"💵 **Estimated {ride_type} Fare ({target_year}) - Future Projection:** ${predicted_fare:.2f}")
        elif target_year > 2025:
            st.success(f"💵 **Estimated {ride_type} Fare ({target_year}) - Fed Projection:** ${predicted_fare:.2f}")
        else:
            st.success(f"💵 **Estimated {ride_type} Fare ({target_year}):** ${predicted_fare:.2f}")
        
        # Show pricing breakdown
        with st.expander("💰 Dynamic Pricing Breakdown"):
            st.write(f"**Original Model Prediction (2012 baseline):** ${raw_predicted_fare:.2f}")
            
            if target_year <= 2025:
                st.write(f"**Historical Inflation Adjustment ({target_year}):** +{inflation_percent:.1f}% = ${inflation_adjusted_fare:.2f}")
            elif target_year <= 2027:
                st.write(f"**Fed Projected Inflation ({target_year}):** +{inflation_percent:.1f}% = ${inflation_adjusted_fare:.2f}")
                st.caption("Based on Federal Reserve June 2025 economic projections")
            else:
                st.write(f"**Future Inflation Projection ({target_year}):** +{inflation_percent:.1f}% = ${inflation_adjusted_fare:.2f}")
                st.caption("Based on Fed's 2% long-run inflation target")
            
            if surcharges > 0:
                st.write(f"**Modern Surcharges:** ${surcharges:.2f}")
                if target_year >= 2019:
                    st.write("  • Improvement Surcharge (2019+): $1.00")
                if target_year >= 2023 and (pickup_in_manhattan or dropoff_in_manhattan):
                    st.write("  • Congestion Pricing (2023+): $1.50")
            
            # Show ride type adjustment
            if ride_multiplier != 1.0:
                before_ride_adjustment = inflation_adjusted_fare + surcharges
                st.write(f"**{ride_type} Premium:** +{((ride_multiplier-1)*100):.0f}% = ${predicted_fare:.2f}")
                st.caption(f"{ride_type} accommodates up to {6 if ride_type == 'UberXL' else 4} passengers")
            
            st.write(f"**Total Estimated {ride_type} Fare:** ${predicted_fare:.2f}")
            
            if target_year > 2027:
                st.info("🔮 **Future Projection Note:** For years beyond 2027, projections assume the Federal Reserve's long-run inflation target of 2.0% annually. Actual inflation may vary based on economic conditions, policy changes, and unforeseen events.")
                
        # Show compound inflation effect for far future years
        if target_year >= 2040:
            years_from_baseline = target_year - 2012
            st.warning(f"⚠️ **Long-term Projection:** This fare estimate is {years_from_baseline} years into the future. Small changes in annual inflation rates compound significantly over time. Actual costs may vary substantially from these projections.")
            
        # Historical comparison chart (extended to future)
        with st.expander("📈 Fare Over Time (Past & Future Projections)"):
            import matplotlib.pyplot as plt
            
            # Extended range: historical + future projections
            years = list(range(2009, min(target_year + 10, 2070)))  # Show 10 years ahead or until 2070
            historical_fares = []
            projection_types = []
            
            for year in years:
                year_cpi, year_multiplier = calculate_future_cpi(year)
                year_fare = raw_predicted_fare * year_multiplier
                
                # Add surcharges for appropriate years
                year_surcharges = 0
                if year >= 2019:
                    year_surcharges += 1.00
                if year >= 2023 and (pickup_in_manhattan or dropoff_in_manhattan):
                    year_surcharges += 1.50
                
                # Apply ride type multiplier
                final_year_fare = (year_fare + year_surcharges) * ride_multiplier
                historical_fares.append(final_year_fare)
                
                # Track data source
                if year <= 2025:
                    projection_types.append('Historical')
                elif year <= 2027:
                    projection_types.append('Fed Projections')
                else:
                    projection_types.append('Long-run Target (2%)')
            
            # Create the chart data
            chart_data = pd.DataFrame({
                'Year': years,
                f'{ride_type} Estimated Fare': historical_fares,
                'Data Source': projection_types
            })
            
            # Color-code the chart
            st.line_chart(chart_data.set_index('Year')[[f'{ride_type} Estimated Fare']], height=300)
            
            # Show data source legend
            st.caption("""
            **Data Sources:**
            • 2009-2025: Historical CPI data (US Bureau of Labor Statistics)
            • 2026-2027: Federal Reserve projections (June 2025 FOMC)
            • 2028+: Fed's long-run inflation target (2.0% annually)
            """)
            
            # Future fare examples
            if target_year > 2025:
                st.write("**Future Fare Examples:**")
                sample_years = [2030, 2040, 2050] if target_year <= 2030 else [target_year + 5, target_year + 15, target_year + 25]
                for sample_year in sample_years:
                    if sample_year <= 2070:  # Reasonable projection limit
                        sample_cpi, sample_multiplier = calculate_future_cpi(sample_year)
                        base_sample_fare = raw_predicted_fare * sample_multiplier + (1.00 + (1.50 if pickup_in_manhattan or dropoff_in_manhattan else 0))
                        sample_fare = base_sample_fare * ride_multiplier
                        sample_inflation = ((sample_multiplier - 1) * 100)
                        st.write(f"• **{sample_year} {ride_type}:** ${sample_fare:.2f} (+{sample_inflation:.1f}% inflation from 2012)")
                
                st.caption("*Future projections based on Federal Reserve's 2% long-run inflation target")

        # Optional: Show individual model predictions
        with st.expander("🔍 Individual Model Predictions (2009-2015 Base)"):
            model_names = list(base_models.keys())
            for i, (name, pred) in enumerate(zip(model_names, base_predictions)):
                st.write(f"**{name}:** ${pred:.2f}")
            st.write(f"---")
            st.write(f"**Raw Stacking Prediction:** ${raw_predicted_fare:.2f}")
            st.write(f"**Final {ride_type} Prediction:** ${predicted_fare:.2f}")
            st.caption("*These are the original model predictions before inflation and ride type adjustments")
            
        # Show trip details
        with st.expander("📊 Trip Details"):
            st.write(f"**Ride Type:** {ride_type} (up to {6 if ride_type == 'UberXL' else 4} passengers)")
            st.write(f"**Selected Passengers:** {passenger_count}")
            st.write(f"**Distance:** {distance_km:.2f} km")
            st.write(f"**Estimated Duration:** {duration_est:.1f} minutes")
            st.write(f"**Distance Category:** {distance_category}")
            st.write(f"**Time of Day:** {time_of_day}")
            st.write(f"**Is Weekend:** {'Yes' if is_weekend else 'No'}")
            st.write(f"**Pickup:** ({pickup_lat:.4f}, {pickup_lon:.4f})")
            st.write(f"**Dropoff:** ({dropoff_lat:.4f}, {dropoff_lon:.4f})")

else:
    st.info("Please click two locations on the map: one for pickup and one for dropoff.")
    
    # Show sample usage
    with st.expander("💡 How to Use"):
        st.markdown("""
        1. **Click on the map** to select your pickup location (green marker)
        2. **Click again** to select your dropoff location (red marker)
        3. **Adjust settings** like passenger count and trip time
        4. **View your predicted fare** and individual model predictions
        5. **Click Reset Map** to start over
        """)
        
    # Show model info
    with st.expander("🤖 About the Model & Future Projections"):
        st.markdown("""
        This app uses a **Stacking Ensemble** of 5 machine learning models trained on 2009-2015 NYC taxi data:
        - **Random Forest** - Tree-based ensemble
        - **XGBoost** - Gradient boosting
        - **LightGBM** - Fast gradient boosting
        - **CatBoost** - Categorical boosting
        - **ElasticNet** - Linear regression with regularization
        
        **Infinite Time Range Capabilities:**
        - **Historical Data (2009-2025)** - Uses actual US Bureau of Labor Statistics CPI data
        - **Near-term Future (2026-2027)** - Uses Federal Reserve economic projections (June 2025 FOMC)
        - **Long-term Future (2028+)** - Uses Fed's 2% long-run inflation target
        - **Modern Surcharges** - Applied when historically appropriate:
          - Improvement Surcharge ($1.00) - Started ~2019
          - Congestion Pricing ($1.50) - Started 2023 for Manhattan trips
        
        **🔮 Future Projection Methodology:**
        The app can predict fares for ANY future year by combining:
        1. **Official Fed projections** for near-term inflation (2026-2027)
        2. **Economic theory** - Fed's 2% long-run inflation target for years beyond
        3. **Compound interest calculations** - Accurately models cumulative inflation effects
        
        **⚠️ Limitations:** Future projections become less reliable over longer time horizons due to economic uncertainty, policy changes, and technological disruption. Use far-future estimates as educational approximations rather than precise forecasts.
        """)