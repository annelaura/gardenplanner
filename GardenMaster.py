import streamlit as st
from planner import datetime, weather_forecast, read_crop_info, calc_start_end, plan_garden_beds, block_type_for_bed, plot_plan
import pandas as pd
import plotly.graph_objects as go

# Set the page configuration 
st.set_page_config(layout="wide")
def main():
    st.title("Garden Planner")

    # Logo or image in sidebar
    st.sidebar.image('assets/logo.png', width=250)
    
    # Define tabs
    tabs = st.tabs(["Home", "Data Import", "Garden Plan", "Weekly Calendar"])

    with tabs[0]:
        show_home_page()
    with tabs[1]:
        show_data_import_page()
    with tabs[2]:
        show_garden_plan_page()
    with tabs[3]:
        show_weekly_calendar_page()

def show_home_page():
    # Display the weather forecast in the third column
    #forecast = weather_forecast()
    st.header("Sorø Weather Forecast - DMI")
    #st.components.v1.html(forecast, height=600)

    # Display the number of sunlight hours today based on avg_HSL in planner.py
    date = pd.Timestamp.today().date()

    from planner import calculate_sunrise_sunset
    lat = 55.41989879338981
    lon = 11.542877610436445
    HSL_df = calculate_sunrise_sunset(date.year, latitude=lat, longitude=lon)
    sunlight = HSL_df.loc[HSL_df.day >=pd.to_datetime(date)].head(3)

    # Display the input value
    st.write("Number of sunlight hours today:", sunlight)
    st.write('today:', date)

plan_df = None
def show_data_import_page():
    st.header("Data Import")
    st.write("File Upload")
    st.text("Upload your crop profile here. The program expects a file that")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
    

    crop_info = None
    bed_plan = None
    garden_figures = None
    year = None
    
    if uploaded_file is not None:
        # Process the uploaded file
        st.write("File name:", uploaded_file.name)
        st.write('filetype:', uploaded_file.type)
        # Read the file content
        crop_info = read_crop_info(uploaded_file)        
        # Display the file content
    if crop_info is not None:
        st.write("File content:", crop_info.head(),"\n")

    st.text("Upload your existing bed plan here. If you do not have any, leave empty.")
    uploaded_file2 = st.file_uploader("Choose another file", type=["csv", "xlsx", "tsv"])
    
    if uploaded_file2 is not None:
        # Process the uploaded file
        st.write("File name:", uploaded_file2.name)
        # Read the file content
        bed_plan = read_crop_info(uploaded_file2)        
        # Display the file content
        st.write("File content:", bed_plan.head(),"\n")
    
    
    # Adding a Streamlit number input field for the year
    year_selected = st.number_input("Enter the year to compute plan for", min_value=2024, max_value=2100, step=1)
    if year_selected is not None:
        # Display the selected year
        st.write("Selected year:", year_selected)

    if crop_info is not None:
        if bed_plan is not None:
            if year_selected is not None:
                st.write("Calculating bed plan\n")
                dfs = []
                for index, row in crop_info.iterrows():
                    dfs.append(calc_start_end(row,year=year_selected)) 
                crop_df = pd.concat(dfs)
                crop_df = pd.merge(crop_df, crop_info[['Afgrøde', 'seeding_type','placement_type','placement_block','FAMILY','multiple','bed_size']], left_on = 'crop', right_on = 'Afgrøde')
                crop_df['succession']=crop_df.groupby('crop').cumcount()+1
                crop_df['crop_no'] = crop_df.crop+'_'+crop_df.succession.astype(str)
                crop_df.loc[crop_df.multiple>=2,'crop_no']=crop_df.crop+'_1.'+crop_df.succession.astype(str)
                crop_df.loc[crop_df.multiple>=2,'succession']=1
                crop_df.reset_index(drop = True, inplace = True)
                crop_df.replace('winter',datetime(2024,12,31), inplace = True)
                crop_df = crop_df.sort_values('garden_start')
                st.write("Crop dataframe shape:", crop_df.shape[0])

                global plan_df
                plan_df = plan_garden_beds(crop_df, bed_plan, num_beds=50, num_greenhouse_beds=10)
                plan_df['block']=[block_type_for_bed(x, num_greenhouse_beds=10) for x in plan_df.bed]
                
                st.write("garden plan crop no:", plan_df.crop_no.unique().shape[0], "\n")

                missing = crop_df.loc[crop_df.crop_no.isin(plan_df.crop_no)==False].sort_values('crop_no')
                st.write("missing in plan crop no:", missing.crop_no.unique().shape[0],"\n")

                st.write("Computing garden plan, go to the tab Garden Plan\n")

def show_garden_plan_page():
    st.header("Garden Plan")
    st.write("Content for Garden Plan goes here.")
    garden_figures = None
    global plan_df 

    if plan_df is not None:
        garden_figures = plot_plan(plan_df)

    if garden_figures is not None:
        #st.plotly_chart(garden_figures)
        # Create two columns
        col1, col2, col3 = st.columns(3)

        # Display the plots in two columns, three rows
        with col1:
            st.plotly_chart(garden_figures[0], use_container_width=True)
            # Create an empty figure
            empty_fig = go.Figure()
            st.plotly_chart(empty_fig, use_container_width=True)

        with col2:
            st.plotly_chart(garden_figures[3], use_container_width=True)
            st.plotly_chart(garden_figures[4], use_container_width=True)
        
        with col3:
            st.plotly_chart(garden_figures[2], use_container_width=True)
            st.plotly_chart(garden_figures[1], use_container_width=True)
            


    # Add more components related to garden plan

def show_weekly_calendar_page():
    st.header("Weekly Calendar")
    st.write("Content for Weekly Calendar goes here.")
    # Add more components related to weekly calendar

if __name__ == '__main__':
    main()