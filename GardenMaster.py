import streamlit as st
from planner import datetime, timedelta,weather_forecast, read_crop_info, calc_start_end, plan_garden_beds, block_type_for_bed, plot_plan, get_calendar_and_seed_order, progress_bar_html
import pandas as pd
import plotly.graph_objects as go
import io
import os
import json

# File path for saving notes
NOTES_FILE = "assets/observations.log"

# Function to save notes to a file with the week number
def save_notes_to_file(note):
    with open(NOTES_FILE, 'a') as f:
        # Get current week number
        current_week = datetime.now().isocalendar()[1]
        # Create log entry with week number
        log_entry = {"week": current_week, "note": note}
        f.write(json.dumps(log_entry) + "\n")

# Function to load notes from a file
def load_notes_from_file():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'r') as f:
            for line in f:
                log_entry = json.loads(line)
                st.session_state.other_tasks.append(log_entry["note"])

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

# Function to handle file uploads or use default files
def handle_file_uploads():
    if "file1" in st.session_state and "file2" in st.session_state:
        # Use uploaded files
        file1 = st.session_state.file1
        file2 = st.session_state.file2
    else:
        # Use default files
        file1, file2 = load_default_files()
    return file1, file2

# Function to load default files from the "assets" directory
def load_default_files():
    default_file1 = os.path.join("assets", "Afgrødeprofiler.xlsx")
    default_file2 = os.path.join("assets", "bed_plan.csv")
    return default_file1, default_file2

plan_df = None
crop_info = None
crop_df = None
def show_data_import_page():
    st.header("Data Import")
    st.write("File Upload")
    st.text("Upload your crop profile here.")
    bed_plan = None
    # Initialize session state for files if not already initialized
    if 'uploaded_file1' not in st.session_state:
        st.session_state.uploaded_file1 = None

    # Handle the first file upload
    uploaded_file1 = st.file_uploader("Choose a file for crop profile", type=["csv", "xlsx", "tsv"], key="file1")
    if uploaded_file1:
        st.session_state.uploaded_file1 = uploaded_file1
    
    # Load files, defaulting to assets if no upload
    default_file1, default_file2 = load_default_files()
    
    global crop_info
    
    if st.session_state.uploaded_file1:
        st.write("File name:", st.session_state.uploaded_file1.name)
        st.write('filetype:', st.session_state.uploaded_file1.type)
        file1 = st.session_state.uploaded_file1
        file1_type = file1.type.split("/")[1]
        crop_info = read_crop_info(file1, file1_type)
    else:
        st.write("No file uploaded. Using default file:", default_file1)
        crop_info = read_crop_info(default_file1, "xlsx")
    
        # Display the content of the files
    st.write("Crop Info:")
    st.write(crop_info.head(), "\n")

    st.text("Upload your current bed plan here.")
    if 'uploaded_file2' not in st.session_state:
        st.session_state.uploaded_file2 = None
    # Handle the second file upload
    uploaded_file2 = st.file_uploader("Choose another file for bed plan", type=["csv", "xlsx", "tsv"], key="file2")
    if uploaded_file2:
        st.session_state.uploaded_file2 = uploaded_file2

    if st.session_state.uploaded_file2:
        st.write("File name:", st.session_state.uploaded_file2.name)
        st.write('filetype:', st.session_state.uploaded_file2.type)
        file2 = st.session_state.uploaded_file2
        file2_type = file2.type.split("/")[1]
        bed_plan = read_crop_info(file2, file2_type)
    else:
        st.write("No file uploaded. Using default file:", default_file2)
        bed_plan = read_crop_info(default_file2, "csv")

    # Display the content of the files
    st.write("Bed Plan:")
    st.write(bed_plan.head(), "\n")
    
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
                global crop_df
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
    df = None
    global plan_df
    global crop_df
    global crop_info 

    if plan_df is not None:
        st.write("got plan\n")
        df, seed_order = get_calendar_and_seed_order(plan_df, crop_df, crop_info)

    if df is not None:
        st.write("got calendar\n")
        st.write("File content:", df.head(), "\n")
        
        # Initialize session state variables
        if 'week_offset' not in st.session_state:
            st.session_state.week_offset = 0

        if 'task_status' not in st.session_state:
            st.session_state.task_status = {}

        if 'other_tasks' not in st.session_state:
            st.session_state.other_tasks = []
        
        if 'notes' not in st.session_state:
            loaded_notes = load_notes_from_file()
            st.session_state.notes = loaded_notes if loaded_notes is not None else []

        def get_current_week_data(week_offset, df):
            today = datetime.now()
            start_date = today - pd.DateOffset(days=today.weekday()) + pd.DateOffset(weeks=week_offset)
            end_date = start_date + pd.DateOffset(days=6)
            df['date'] = pd.to_datetime(df['date'])
            week_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return start_date, end_date, week_data

        # Get data for the current week
        start_date, end_date, week_data = get_current_week_data(st.session_state.week_offset, df)
        if week_data is not None:
            st.write("got week_data\n")
            st.write("File content:", week_data.head(), "\n")

        # Header with week navigation
        st.markdown(f"### Week {start_date.isocalendar()[1]}: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button('⬅️ Previous Week'):
                st.session_state.week_offset -= 1

        with col3:
            if st.button('Next Week ➡️'):
                st.session_state.week_offset += 1

        # Display categories and tasks
        st.header("Tasks for This Week")

        categories = week_data['variable'].unique()
        for category in categories:
            st.subheader(category.replace('_', ' ').title())
            category_tasks = week_data[week_data['variable'] == category].drop_duplicates()
            for _, task in category_tasks.iterrows():
                # Create a unique task ID
                task_id = "_".join(map(str, task.tolist()))
                checkbox_text = ""
                
                if category == 'tp_start':
                    checkbox_text = f"**{task['crop']}** ({task['crop_no']}, {task['date'].strftime('%d/%m/%Y')}):\nSeeding type: {task['seeding_type']}, Seeds pr. cell: {task['seeds_pr_cell']}, Cells pr tray: {task['cells_pr_tray']}, Seeds (+20%): {task['seeds']}({task['seeds_20']}), Trays: {task['trays']}"
                elif category == 'garden_start':
                    checkbox_text = f"**{task['crop']}** ({task['crop_no']}, {task['date'].strftime('%d/%m/%Y')}):\nSeeding type: {task['seeding_type']}, Block: {task['block']}, Bed: {task['bed']}, Bed size: {task['bed_size']}, Bed location: {task['bed_location']}, Seeds: {task['seeds']}, Rows: {task['ROWS']}, Spacing: {task['SPACING']}"
                elif category == 'bedprep_start':
                    checkbox_text = f"**Bed: {task['bed']}** ({task['block']},{task['date'].strftime('%d/%m/%Y')}):\nCrop going in: {task['crop']}"
                elif category == 'harvest':
                    checkbox_text = f"**{task['crop']}** ({task['crop_no']}, {task['date'].strftime('%d/%m/%Y')}):\nHarvest details - Block: {task['block']}, Bed: {task['bed']}, Harvest Period: {task['percentage_harvestperiod']}% (end: {task['harvest_end'].strftime('%d/%m/%Y')})"
                    percentage = task['percentage_harvestperiod']
                # Initialize the checkbox state if task_id not in session_state
                if task_id not in st.session_state.task_status:
                    st.session_state.task_status[task_id] = False
                
                # Create or update the checkbox
                st.session_state.task_status[task_id] = st.checkbox(checkbox_text, value=st.session_state.task_status[task_id])
                if category == 'harvest':
                    st.markdown(progress_bar_html(percentage), unsafe_allow_html=True)
        
        # Other tasks and notes section
        st.subheader("Other Tasks")
        other_task_input = st.text_input("Add a new task for this week:")
        if st.button("Add Task"):
            if other_task_input:
                st.session_state.other_tasks.append(other_task_input)
                st.session_state.task_status[other_task_input] = False

        for task in st.session_state.other_tasks:
            st.session_state.task_status[task] = st.checkbox(
                task, value=st.session_state.task_status[task]
            )

        # Notes section
        # st.subheader("Notes/Comments/Observations")
        # notes = st.text_area("Add any notes or observations about your garden here...")

        # # Persist data (in a real app, save this to a database or a file)
        # if st.button("Save Notes"):
        #     # Save the notes and other tasks
        #     st.success("Notes saved!")
        st.subheader("Notes/Comments/Observations")
        notes = st.text_area("Add any notes or observations about your garden here...", key="notes_area")

        if st.button("Save Notes"):
            if notes:
                # Save the notes
                save_notes_to_file(notes)
                # Clear the text area
                st.session_state.notes.append({"week": datetime.now().isocalendar()[1], "note": notes})
                st.experimental_rerun()

        # Display saved notes
        if st.session_state.notes:
            st.write("Previous Notes:")
            for note_entry in st.session_state.notes:
                st.write(f"Week {note_entry['week']}: {note_entry['note']}")



    # Add more components related to weekly calendar:

    # Hovedkategorier:

    # Direkte såning
    # Start Forspiring
    # Udplantning
    # Høst (ink. Bar som viser høst perioden og hvor meget der er gået af den)
    # Bed prep til næste uge
    # Kultivering:
    # Forhave, blok 1,2,3, Drivhus
    # Andre opgaver

    # NOTER


if __name__ == '__main__':
    main()