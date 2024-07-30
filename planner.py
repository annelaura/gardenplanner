# this file contains the functions for the garden planner

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import ephem
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import pytz
import plotly.express as px
from plotly.subplots import make_subplots

# set constants
lat = 55.41989879338981
lon = 11.542877610436445
city = 'Sorø'

def weather_forecast():
    """
    Set up Selenium options, WebDriver, open a weather page, extract HTML content, and close the browser.

    Returns:
        str: The HTML content of the desired div.
    """
    # Set up Selenium options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure the browser runs in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the WebDriver
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # URL of the DMI Sorø weather page
    url = "https://www.dmi.dk/lokation/show/DK/2612862/Sor%C3%B8/"

    # Open the page with Selenium
    driver.get(url)
    time.sleep(2)  # Wait for the page to load completely

    # Extract the HTML content of the desired div
    overview_div = driver.find_element(By.ID, 'overviewRow').get_attribute('outerHTML')

    # Close the browser
    driver.quit()

    return overview_div

# calculate average daily hours of sunlight from equinox to equinox to find how many hours of daylight each crop needs to maturitu
def calculate_sunrise_sunset(year, latitude, longitude):
    # Set the location
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)

    # Set the timezone
    timezone = 'Europe/Copenhagen'
    timezone_obj = pytz.timezone(timezone)
    observer.date = datetime.now(timezone_obj)

    # Specify the year
    current_date = datetime(year, 1, 1, tzinfo=timezone_obj)
    end_date = datetime(year + 1, 1, 1, tzinfo=timezone_obj)

    # Initialize the DataFrame
    data = []

    while current_date < end_date:
        observer.date = current_date
        sunrise_time = observer.next_rising(ephem.Sun()).datetime()
        sunset_time = observer.next_setting(ephem.Sun()).datetime()
        dusk_time = observer.next_transit(ephem.Sun(), start=sunset_time).datetime()
        hsl_hours = (sunset_time - sunrise_time).total_seconds() / 3600

        # Adjust the times to the specified timezone
        sunrise_time = sunrise_time.replace(tzinfo=pytz.UTC)
        sunset_time = sunset_time.replace(tzinfo=pytz.UTC)
        dusk_time = dusk_time.replace(tzinfo=pytz.UTC)

        sunrise_time = sunrise_time.astimezone(timezone_obj)
        sunset_time = sunset_time.astimezone(timezone_obj)
        dusk_time = dusk_time.astimezone(timezone_obj)

        data.append({
            'day': current_date.date(),
            'sunrise': sunrise_time.time(),
            'sunset': sunset_time.time(),
            'dusk': dusk_time.time(),
            'HSL': hsl_hours
        })

        current_date += timedelta(days=1)

    # Create the DataFrame
    df = pd.DataFrame(data)
    df['day'] = pd.to_datetime(df['day'])

    return df
def DTM_byHSL(HSL_df, day, DTM, direction = 'forward'):
    """
    Calculate the DTM (Days to Maturity) based on the average daily hours of sunlight (HSL) from equinox to equinox.
    
    Args:
        HSL_df (pandas.DataFrame): A DataFrame containing the average daily hours of sunlight from equinox to equinox.
        day (datetime.date): The day for which the calculation is performed.
        DTM (float): The DTM value.
        direction (str, optional): The direction of calculation. Defaults to 'forward'.
    
    Returns:
        int: The DTM based on the average daily hours of sunlight.
    """

    HSL_avg = HSL_df.loc[HSL_df.HSL>=12].HSL.sum()/HSL_df.loc[HSL_df.HSL>=12].shape[0]
    hsl = DTM*HSL_avg
    day = pd.to_datetime(day)
    if direction == 'forward': # calculate DTM forward in time (eg if we want to know when harvest starts from a specified seeding day)
        sub = HSL_df.loc[HSL_df.day>=day]
        sub['cumsum']=sub.HSL.cumsum()
        ready = sub.loc[sub['cumsum']>=hsl].day.tolist()[0]
        dtm_by_HSL = (ready-day).days
    elif direction == 'backward': # calculate DTM backward in time (eg if we want to know the day to seed if we know the day to harvest)
        sub = HSL_df.loc[HSL_df.day<=day].sort_values('day', ascending = False)
        sub['cumsum']=sub.HSL.cumsum()
        start = sub.loc[sub['cumsum']>=hsl].day.tolist()[0]
        dtm_by_HSL = (day-start).days
    return(pd.to_timedelta(dtm_by_HSL, unit='D'))
  
def calc_start_end(row, year): # OBS DTM should be calculated by date and not by avg
    """
    Calculate the start and end dates for different crop successions based on the input parameters in the 'row' DataFrame.
    
    Parameters:
    - row (pandas.Series): A Series containing information about crop planting, growth, and harvest.
    
    Returns:
    - succession_df (pandas.DataFrame): A DataFrame containing the calculated start and end dates for crop successions including planting, growth, and harvest phases.
    """
    print(row['Afgrøde'])
    end_of_year = datetime(2024, 12, 31)
    succession_df = pd.DataFrame()
    bed_prep = []
    
    HSL_df = calculate_sunrise_sunset(year, latitude=lat, longitude = lon)

    if not (row['forspiring'] is pd.NaT): # crops that are grown as transplants
        tp_start = row['forspiring']
        garden_start = row['udplantning']
        tp_days = garden_start - tp_start # days as transplant
        harvest_start = row['forspiring']+DTM_byHSL(HSL_df, row['forspiring'], row['DTM'], direction = 'forward')
        harvest_end = harvest_start + pd.to_timedelta(row['DTM_max']-row['DTM'], unit = 'D')+row['DAYS AFTER MATURITY']
        harvest_window = harvest_end - harvest_start
        new_values = [[tp_start, garden_start, harvest_start, harvest_end]]
        succession_df = pd.DataFrame(new_values, columns = ['tp_start','garden_start','harvest_start','harvest_end'])
        if row['succession']:     
            new_window_start = harvest_end + pd.to_timedelta(1, unit = 'D')
            while new_window_start < end_of_year:
                new_end = new_window_start+harvest_window
                new_tp_start = new_window_start - DTM_byHSL(HSL_df, new_window_start, row['DTM'], direction = 'backward')
                new_garden_start = new_tp_start + tp_days
                # save values and continue
                new_values.append([new_tp_start, new_garden_start, new_window_start, new_end])
                new_window_start = new_end
            succession_df = pd.DataFrame(new_values, columns = ['tp_start','garden_start','harvest_start','harvest_end'])
            # subset to successions in the possible window
            succession_df = succession_df.loc[succession_df.garden_start<=row['last_seeding']]
            if row['winter']: #add last possible start
                print('computing winter succession')
                last_garden_start = row['last_seeding']
                last_tp_start = last_garden_start - tp_days
                # do not include winter crop if it is too close to last garden start
                if last_garden_start <= (succession_df.garden_start.tolist()[-1]+pd.to_timedelta(row['DTM']/2, unit = 'D')):
                    pass
                else:
                    succession_df = pd.concat([succession_df, pd.DataFrame([[last_tp_start,last_garden_start,'winter','winter']], columns = succession_df.columns.tolist())])

    elif (row['forspiring'] is pd.NaT):

        # direct seeding calculation
        if row['seeding_type']=='DS': 
            garden_start = row['DS']
            harvest_start = garden_start + DTM_byHSL(HSL_df, garden_start, row['DTM'], direction = 'forward')
            harvest_end = harvest_start + pd.to_timedelta(row['DTM_max']-row['DTM'], unit = 'D')+row['DAYS AFTER MATURITY']   
            tp_start = 'DS'
            tp_days = 0

        # sætte calculation
        elif row['seeding_type']=='Sætte':
            garden_start = row['udplantning']
            harvest_start = row['udplantning'] + pd.to_timedelta(row['DTM'], unit='D')
            harvest_end = row['udplantning'] + pd.to_timedelta(row['DTM_max'], unit='D')+row['DAYS AFTER MATURITY']
            tp_start = 'Sætte_uden_forspiring'
            tp_days = 0
        harvest_window = harvest_end-harvest_start
        
        new_values = [[tp_start, garden_start, harvest_start, harvest_end]]
        succession_df = pd.DataFrame(new_values, columns = ['tp_start','garden_start','harvest_start','harvest_end'])
        
        if row['succession']: 
            new_window_start = harvest_end + pd.to_timedelta(1, unit = 'D')
            while new_window_start < end_of_year:
                new_end = new_window_start+harvest_window
                new_garden_start = new_window_start - DTM_byHSL(HSL_df, new_window_start, row['DTM'], direction = 'backward')
                new_tp_start = np.nan
                new_values.append([new_tp_start, new_garden_start, new_window_start, new_end])
                new_window_start = new_end
            
            succession_df = pd.DataFrame(new_values, columns = ['tp_start','garden_start','harvest_start','harvest_end'])
            # subset to successions in the possible window
            succession_df = succession_df.loc[succession_df.garden_start<=row['last_seeding']]

            if row['winter']: #add last possible start
                last_garden_start = row['last_seeding']
                last_tp_start = np.nan
                # do not include winter crop if it is too close to last garden start
                if last_garden_start <= (succession_df.garden_start.tolist()[-1]+pd.to_timedelta(row['DTM']/2, unit = 'D')):
                    pass
                else:
                    succession_df = pd.concat([succession_df, pd.DataFrame([[last_tp_start,last_garden_start,'winter','winter']], columns = succession_df.columns.tolist())])
    else:
        print('no seeding information')                                          
    
    # combine all the information
    succession_df['crop']=row['Afgrøde']
    succession_df['bedprep_start']=succession_df.harvest_end.copy()
    succession_df['bedprep_end']=succession_df.harvest_end.copy()
    succession_df.loc[succession_df.harvest_end!='winter', 'bedprep_start'] = succession_df.loc[succession_df.harvest_end!='winter'].harvest_end + pd.to_timedelta(1, unit = 'D')
    succession_df.loc[succession_df.harvest_end!='winter', 'bedprep_end'] = succession_df.loc[succession_df.harvest_end!='winter'].bedprep_start + pd.to_timedelta(1, unit = 'W')
    if row['multiple']>=2:
        succession_df = pd.concat([succession_df]*int(row['multiple']))

    return(succession_df)

def read_crop_info(uploaded_file):
    """
    Reads a crop profile file based on the specified separator.

    Parameters:
        separator (str): The separator used in the file. Defaults to the type of the uploaded file.

    Returns:
        pandas.DataFrame: The crop profile data read from the file.
    """
    crop_info = None
    separator=uploaded_file.type
    if separator == "text/csv":
        crop_info = pd.read_csv(uploaded_file)
    elif separator == "text/tab-separated-values":
        crop_info = pd.read_csv(uploaded_file, sep="\t")
    elif separator == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        crop_info = pd.read_excel(uploaded_file)
    else:
        print("File type not supported")
    # format columns
    if 'DTM_max' in crop_info.columns.tolist():
        crop_info.loc[crop_info.DTM_max.isnull(),'DTM_max']=crop_info.loc[crop_info.DTM_max.isnull()].DTM.tolist()
    if 'DAYS AFTER MATURITY' in crop_info.columns.tolist():
        crop_info['DAYS AFTER MATURITY'] = crop_info['DAYS AFTER MATURITY'].fillna(0)
        crop_info['DAYS AFTER MATURITY'] = pd.to_timedelta(crop_info['DAYS AFTER MATURITY'], unit='D')

    return crop_info

def calculate_crop_info(crop_info):
    """
    Calculate the crop information based on the crop profile data provided.

    Parameters:
    - crop_info (pandas.DataFrame): The crop profile data containing information about crop planting, growth, and harvest.

    Returns:
    - crop_df (pandas.DataFrame): The calculated crop information including details like seeding type, placement type, family, etc.
    """
    dfs = []
    for index, row in crop_info.iterrows():
        print(row['Afgrøde'])
        dfs.append(calc_start_end(row)) # OBS DTM should be calculated by date and not by avg

    crop_df = pd.concat(dfs)
    crop_df = pd.merge(crop_df, crop_info[['Afgrøde', 'seeding_type','placement_type','placement_block','FAMILY','multiple','bed_size']], left_on = 'crop', right_on = 'Afgrøde')
    # change the names
    crop_df['succession']=crop_df.groupby('crop').cumcount()+1
    crop_df['crop_no'] = crop_df.crop+'_'+crop_df.succession.astype(str)
    crop_df.loc[crop_df.multiple>=2,'crop_no']=crop_df.crop+'_1.'+crop_df.succession.astype(str)
    crop_df.loc[crop_df.multiple>=2,'succession']=1
    crop_df.reset_index(drop = True, inplace = True)
    crop_df.replace('winter',datetime(2024,12,31), inplace = True)

    return crop_df

def plan_garden_beds(crop_df, bed_plan, num_beds=50, num_greenhouse_beds = 10):
    """
    Plans the garden beds based on the given crop data and bed plan.
    Args:
        crop_df (pandas.DataFrame): The crop data containing information about the crops.
        bed_plan (pandas.DataFrame): The bed plan containing information about the beds, that is, existing or previous crops.
        num_beds (int, optional): The maximum number of beds to plan. Defaults to 50.
    Returns:
        pandas.DataFrame: The updated bed plan with the planned crops.
    """   
    # add 'priority' to crops, so crops that can only go i one block are prioritized first
    priority = []
    for i, row in crop_df.iterrows():
        if row['placement_block']=='Friland': # 4 options
            priority.append(3)
        elif ', ' in row['placement_block']: # less than 4 options
            priority.append(2)
        else:
            priority.append(1) # one option
    crop_df['priority']=priority
    crop_df.loc[crop_df.crop.str.contains('mix|gulerødder', case= False),'succession']=1
    crop_df.loc[crop_df.crop.str.contains('mix|gulerødder', case= False),'priority']=1

    # Initialize variables to track the current bed, FAMILY, and last end date
    current_bed = 0
    current_FAMILY = ['']
    
    while not crop_df.empty and current_bed <= num_beds-1:
        if current_bed in bed_plan.bed.tolist():
            current_FAMILY = [bed_plan.loc[bed_plan.bed==current_bed].iloc[-1]['FAMILY']]
            last_end_date = bed_plan.loc[bed_plan.bed==current_bed].iloc[-1]['bedprep_end']
            last_end_date = pd.to_datetime(last_end_date)
            bed_succession = 1
        else:
            last_end_date = pd.to_datetime('2024-01-01')  # Adjust the initial date as needed
            bed_succession = 0

        valid_crops = crop_df[(crop_df['garden_start'] >= last_end_date)
                         & (crop_df['placement_type'] == bed_type_for_bed(current_bed, num_greenhouse_beds))
                         & ((crop_df['placement_block'].str.contains(block_type_for_bed(current_bed, num_greenhouse_beds)))|(crop_df['placement_block']==bed_type_for_bed(current_bed, num_greenhouse_beds)))
                         & (crop_df.bed_size<=bed_size_for_bed(current_bed, num_greenhouse_beds))  
                         & (crop_df['FAMILY'].isin(current_FAMILY)==False)].sort_values(['succession','priority','garden_start']).reset_index(drop=True)
        # make if loop to include beds of mixed families, the above valid_crops exclude crops from both families
        if len(current_FAMILY)>=2:
            if current_FAMILY[0]==current_FAMILY[-1]:
                pass
            else:
                additional_valids = crop_df[(crop_df['garden_start'] >= last_end_date)
                         & (crop_df['placement_type'] == bed_type_for_bed(current_bed, num_greenhouse_beds))
                         & (crop_df['bed_size']<=1)
                         & (crop_df['FAMILY']!=current_FAMILY[0]|crop_df['FAMILY']!=current_FAMILY[-1])]         
            valid_crops = pd.concat([valid_crops, additional_valids]).drop_duplicates().sort_values(['succession','priority','garden_start']).reset_index(drop=True)
        bed_succession+=1    
        # it prioritizes crops by
        # 1) no. succession --> crops with only 1 succession is prioritized
        # 2) garden_start
        while not valid_crops.empty:    
            selected_crop = valid_crops.iloc[0]
            end_date = selected_crop['bedprep_end']
            selected_crop['bed']=current_bed
            selected_crop['bed_location']='A'
            selected_crop['bed_succession']=bed_succession
            
            additional_crops = None
            # check if the crop can share bed:
            if selected_crop['bed_size']<=1:
                additional_location = None
                additional_crops = crop_df.loc[(crop_df['garden_start'] >= last_end_date)
                                               & (crop_df.bed_size<=(bed_size_for_bed(current_bed, num_greenhouse_beds)-selected_crop['bed_size']))# space enough to fit in the bed
                                               & (crop_df['placement_type'] == bed_type_for_bed(current_bed, num_greenhouse_beds)) # same bed type
                                               & ((crop_df['placement_block'].str.contains(block_type_for_bed(current_bed, num_greenhouse_beds)))|(crop_df['placement_block']==bed_type_for_bed(current_bed, num_greenhouse_beds))) # same bed type
                                               & (crop_df['bedprep_end'] <= (selected_crop['bedprep_end']+pd.to_timedelta(21, unit = 'D')))
                                               & (crop_df['crop_no']!=selected_crop['crop_no'])].sort_values('bedprep_end').reset_index(drop=True) # same planting period                
                if len(current_FAMILY)>=2:
                    if current_FAMILY[0]!=current_FAMILY[1]:
                        #additional_crops = additional_crops.loc[additional_crops.FAMILY!=current_FAMILY[0]]
                        #location = 'AorB'
                        if selected_crop['FAMILY'] not in current_FAMILY:
                            #selected_location = 'Any'
                            additional_crops = additional_crops.loc[(additional_crops.FAMILY!=current_FAMILY[0])|(additional_crops.FAMILY!=current_FAMILY[-1])]
                            #additional_location = 'AorB'
                        elif selected_crop['FAMILY']==current_FAMILY[0]:
                            selected_location = 'B'
                            additional_location = 'A'
                            additional_crops = additional_crops.loc[additional_crops.FAMILY!=current_FAMILY[0]]
                        elif selected_crop['FAMILY']==current_FAMILY[-1]:
                            selected_location = 'A'
                            additional_location = 'B'
                            additional_crops = additional_crops.loc[additional_crops.FAMILY!=current_FAMILY[-1]]
                    else: 
                        additional_crops = additional_crops.loc[additional_crops.FAMILY.isin(current_FAMILY)==False]

                else:
                    additional_crops = additional_crops.loc[additional_crops.FAMILY.isin(current_FAMILY)==False]
                    selected_location = 'A'
                    additional_location = 'B'
                
                selected_crop['bed_location']=selected_location        
                
                if not additional_crops.empty:
                    print('adding another crop to the same bed')
                    additional_crops['diff']= abs(selected_crop.bedprep_end - additional_crops.bedprep_end)
                    additional_crops = additional_crops.sort_values(['priority','diff']).reset_index(drop=True)
                    additional_crops.drop(['diff'], axis = 1, inplace = True)
                    additional_crop = additional_crops.iloc[0]
                    if additional_crop['FAMILY']==current_FAMILY[0]:
                        selected_location = 'A'
                        additional_location = 'B'
                    elif additional_crop['FAMILY']==current_FAMILY[-1]:
                        additional_location = 'A'
                        selected_location = 'B'
                    selected_crop['bed_location']=selected_location        
                    additional_crop['bed'] = current_bed
                    additional_crop['bed_succession']=bed_succession
                    additional_crop['bed_location']=additional_location
                    if additional_crop['bedprep_end']>=selected_crop['bedprep_end']:
                        end_date = additional_crop['bedprep_end']
                    selected_crop = pd.concat([pd.DataFrame(selected_crop).T,pd.DataFrame(additional_crop).T]).sort_values('bed_location')
                else:
                    selected_crop = pd.DataFrame(selected_crop).T    
            else:
                selected_crop = pd.DataFrame(selected_crop).T
                
            # remove selected crop(s) from crop_df
            crop_df = crop_df[crop_df['crop_no'].isin(selected_crop['crop_no'])==False]
            
            # add additional crop before if possible
            selected_crop.reset_index(drop=True, inplace = True)            
            occupied_by = bed_plan.loc[(bed_plan.bed==current_bed)&(bed_plan.crop_no.isin(selected_crop.crop_no)==False)]
            occupied_by = occupied_by.loc[occupied_by.bed_succession==occupied_by.bed_succession.max()].sort_values('bedprep_end').reset_index(drop=True)
            #if current_bed == 42:
             #   print('\n\nBED 42\n\n{}'.format(occupied_by.head()))
            if occupied_by.empty:
                second_last = [last_end_date]
            else:
                second_last = occupied_by.bedprep_end.tolist()
                
            for i in range(selected_crop.shape[0]):
                potential_before = crop_df.loc[(crop_df.bedprep_end<=selected_crop.garden_start[i])
                                               & (crop_df['placement_type'] == bed_type_for_bed(current_bed, num_greenhouse_beds)) # same bed type
                                               & ((crop_df['placement_block'].str.contains(block_type_for_bed(current_bed, num_greenhouse_beds)))|(crop_df['placement_block']==bed_type_for_bed(current_bed, num_greenhouse_beds))) # same bed type
                                               & (crop_df.FAMILY!=selected_crop.FAMILY[i])]  
                if not potential_before.empty:
                    if selected_crop.shape[0]==1:
                        potential_before = potential_before.loc[(potential_before.FAMILY.isin(current_FAMILY)==False)
                                                           & (potential_before.garden_start>=second_last[-1])]        
                        potential_before['bed_location']=selected_crop.bed_location[0]
                    elif selected_crop.shape[0]>=2:
                        if occupied_by.shape[0]>=2:
                            potential_before_multiple = []    
                            for j in range(occupied_by.shape[0]):
                                potential_before_multiple.append(potential_before.loc[(potential_before.bed_size<=occupied_by.bed_size[j])
                                                                     & (potential_before.FAMILY!=occupied_by.FAMILY[j])
                                                                     & (potential_before.garden_start>=occupied_by.bedprep_end[j])])
                            potential_before = pd.concat(potential_before_multiple)  
                        elif occupied_by.shape[0]==1:
                            potential_before = potential_before.loc[(potential_before.FAMILY!=occupied_by.FAMILY[0])
                                                                     & (potential_before.garden_start>=occupied_by.bedprep_end[0])]
                        elif occupied_by.empty:
                            potential_before = potential_before.loc[(potential_before.garden_start>=last_end_date)]        
                
                if not potential_before.empty:        
                    extra_crop = potential_before.iloc[0]
                    print('\n\nbed:{}\nadding crop before\n'.format(current_bed))
                    print('selected_crop: {} \n extra_crop: {}'.format(selected_crop.iloc[i][['crop_no','garden_start']].tolist(),extra_crop[['crop_no','garden_start','bedprep_end']].tolist()))
                    extra_crop['bed']=current_bed
                    extra_crop['bed_location']=selected_crop['bed_location'][i]
                    extra_crop['bed_succession']=-1
                    extra_crop = pd.DataFrame(extra_crop).T
                    bed_plan = pd.concat([bed_plan, extra_crop])
                    crop_df = crop_df[crop_df['crop_no'].isin(extra_crop['crop_no'])==False]
                    
            # update variables
            bed_plan = pd.concat([bed_plan, selected_crop])
            current_FAMILY = selected_crop['FAMILY'].tolist()
            last_end_date = end_date #+ pd.Timedelta(days=1)
            bed_succession+=1
            # Prepare new valid crops:
            valid_crops = crop_df[(crop_df['garden_start'] >= last_end_date)
                                  & (crop_df['placement_type'] == bed_type_for_bed(current_bed, num_greenhouse_beds))
                                  & ((crop_df['placement_block'].str.contains(block_type_for_bed(current_bed, num_greenhouse_beds)))|(crop_df['placement_block']==bed_type_for_bed(current_bed, num_greenhouse_beds)))
                                  & (crop_df.bed_size<=bed_size_for_bed(current_bed, num_greenhouse_beds))
                                  & (crop_df['FAMILY'].isin(current_FAMILY)==False)].sort_values(['succession','priority','garden_start']).reset_index(drop=True)
        else:
            # If no valid crops for the current bed, move to the next bed
            current_bed += 1
            current_FAMILY = ['']

    return bed_plan

def bed_type_for_bed(bed_number, num_greenhouse_beds):
    # Determine the placement_type for the current bed
    return 'Drivhus' if bed_number <= (num_greenhouse_beds-1) else 'Friland'

def block_type_for_bed(bed_number, num_greenhouse_beds):
    # Determine the placement_type_block for the current bed
    if bed_number <= num_greenhouse_beds-1:
        return('Drivhus')
    elif 10<= bed_number <= 19:
        return('Friland1')
    elif 20<= bed_number <= 29:
        return('Friland2')
    elif 30<= bed_number <= 39:
        return('Friland3')
    elif 40<= bed_number:
        return('Forhave')
    
def bed_size_for_bed(bed_number, num_greenhouse_beds):
    # Determine the placement_type_block for the current bed
    if bed_number <= num_greenhouse_beds-1:
        return(1)
    elif 10<= bed_number <= 19:
        return(1)
    elif 20<= bed_number <= 29:
        return(1)
    elif 30<= bed_number <= 39:
        return(1)
    elif 40<= bed_number:
        return(14/25)

def old_crop_df(crop,bed, location='A', bedsize=1):
    new_row = crop_df.loc[crop_df.crop.str.contains(crop, case = False)].iloc[0:1,:]
    for col in ['garden_start','harvest_start','harvest_end','bedprep_start','bedprep_end']:
        new_row[col]= new_row[col]- pd.DateOffset(years=1)
        if isinstance(new_row['tp_start'].values[0], datetime):
            new_row['tp_start']=new_row['tp_start']- pd.DateOffset(years=1)
    new_row['bed']=bed
    new_row['bed_location']=location
    new_row['bed_succession']=1
    new_row['bed_size']=bedsize
    new_row['FAMILY']=new_row['FAMILY'].tolist()
    return(new_row.reset_index(drop=True))

def plot_plan(plan_df): 
    plot_df = plan_df.sort_values('bed')
    plot_df_B = plot_df.loc[plot_df.bed_size == 1]
    plot_df_B.bed_location = 'B'
    plot_df_A = plot_df.loc[plot_df.bed_size == 1]
    plot_df_A.bed_location = 'A'
    plot_df = pd.concat([plot_df.loc[plot_df.bed_size<=0.9],plot_df_A, plot_df_B])
    # format the dates
    for col in ['garden_start','harvest_start','harvest_end','bedprep_start','bedprep_end']:
       plot_df[col]= pd.to_datetime(plot_df[col])

    plot_dfs = []
    for bed in plot_df.bed.unique().tolist():
        sub = plot_df.loc[plot_df.bed==bed].sort_values('garden_start').reset_index(drop=True)
        placement = []
        count = 0
        last_family_A = ''
        last_family_B = ''
        last_end_A = '' 
        last_end_B = '' 
        for i in range(sub.shape[0]):
            row = sub.iloc[i]
            if i == 0:
                placement.append('A')
                last_end_A = row['bedprep_end']
                last_family_A = row['FAMILY']
            else:
                if row['garden_start']<=(last_end_A-pd.to_timedelta(1, unit='D')):
                    placement.append('B')
                    last_end_B = row['bedprep_end']
                    last_family_B = row['FAMILY']
                else:
                    if row['FAMILY']==last_family_A:
                        placement.append('B')
                        last_end_B = row['bedprep_end']
                        last_family_B = row['FAMILY']
                    elif row['FAMILY']==last_family_B:
                        placement.append('A')
                        last_end_A = row['bedprep_end']
                        last_family_A = row['FAMILY']
                    else:
                        placement.append('A')
                        last_end_A = row['bedprep_end']
                        last_family_A = row['FAMILY']
        sub['bed_location']=placement            
        plot_dfs.append(sub)

    plot_df = pd.concat(plot_dfs)

    # add location for plotting
    plot_df['block_bed_location']=plot_df.bed.astype(int).astype(str)+'_'+ plot_df.bed_location
    plot_df['garden_row']=1
    plot_df['garden_col']=1

    plot_df.loc[plot_df.block=='Forhave','garden_row']=1
    plot_df.loc[plot_df.block=='Forhave','garden_col']=1
    plot_df.loc[plot_df.block=='Drivhus','garden_row']=2
    plot_df.loc[plot_df.block=='Drivhus','garden_col']=2

    plot_df.loc[plot_df.block=='Friland1','garden_row']=1
    plot_df.loc[plot_df.block=='Friland1','garden_col']=2
    plot_df.loc[plot_df.block=='Friland2','garden_row']=1
    plot_df.loc[plot_df.block=='Friland2','garden_col']=3
    plot_df.loc[plot_df.block=='Friland3','garden_row']=2
    plot_df.loc[plot_df.block=='Friland3','garden_col']=3

    plot_df = plot_df.sort_values(['bed','block_bed_location'], ascending = [False,False]) 
    order = plot_df[['block_bed_location']].drop_duplicates().block_bed_location.tolist()
    plot_df['bed_size']=plot_df.bed_size.astype(float)

    garden = plot_df[['garden_start','harvest_start','crop','bed_location','crop_no','bed_size','block','succession','priority','FAMILY','bed']]
    garden.rename(columns = {'garden_start':'start','harvest_start':'end'}, inplace = True)
    garden['info']='garden'
    harvest = plot_df[['harvest_start','harvest_end','crop','bed_location','crop_no','bed_size','block','succession','priority','FAMILY','bed']]
    harvest.rename(columns = {'harvest_start':'start','harvest_end':'end'}, inplace = True)
    harvest['info']='harvest'
    bedprep = plot_df[['bedprep_start','bedprep_end','crop','bed_location','crop_no','bed_size','block','succession','priority','FAMILY','bed']]
    bedprep.rename(columns = {'bedprep_start':'start','bedprep_end':'end'}, inplace = True)
    bedprep['info']='bedprep'
    new_plot_df = pd.concat([garden, harvest, bedprep]).reset_index(drop=True)

    # Define the color scale (palette) to use
    color_scale = px.colors.qualitative.Vivid

    # Map the 'color2' column to colors from the color scale
    #plot_df['color'] = px.colors.qualitative.swatches(color_scale, len(plot_df['FAMILY'])).values()
    color_mapping = {category: color_scale[i % len(color_scale)] for i, category in enumerate(new_plot_df['FAMILY'].unique())}
    new_plot_df['color'] = new_plot_df['FAMILY'].map(color_mapping)

    new_plot_df['annotation_text']=new_plot_df.crop_no.copy()
    new_plot_df.loc[new_plot_df['info']=='harvest','annotation_text']=new_plot_df.loc[new_plot_df['info']=='harvest'].crop_no+'_harvest'
    new_plot_df.loc[new_plot_df['info']=='bedprep','annotation_text']=new_plot_df.loc[new_plot_df['info']=='bedprep'].crop_no+'_bedprep'
    
    # add a date column in text format
    new_plot_df['Start date'] = new_plot_df['start'].dt.strftime('%d %B %Y')
    new_plot_df['End date'] = new_plot_df['end'].dt.strftime('%d %B %Y')

    figures = []
    for block in new_plot_df.block.unique():
        block_data = new_plot_df.loc[new_plot_df.block == block].sort_values(['bed','crop_no','bed_location']).reset_index(drop=True)
        fig = px.timeline(block_data, x_start="start", x_end="end", 
                        y='bed', text='annotation_text',
                        color = 'bed_location', 
                        hover_data=['crop','crop_no','start','end','bed_size','block','succession','priority','FAMILY','color'],
                        #facet_col = 'garden_col', facet_row = 'garden_row',
                        #category_orders={#'block':['Forhave','Drivhus','Friland1','Friland3','Friland2'],
                        #                'garden_row':[1,2],
                            #               'garden_col':[1,2,3],
                            #              'block_bed_location':order,
                                        #'FAMILY': new_plot_df.FAMILY.unique().tolist()}, 
                        template = 'plotly_white', 
                        width = 1000, height = 500, title = block)
        fig.update_layout(barmode='group', showlegend=False)
        #for i, d in enumerate(fig.data):
        #    d.width = new_plot_df[new_plot_df['crop_no']==d.name]['bed_size']
        
        # add grid
        fig.update_xaxes(showgrid=True, ticks= "outside",
                        ticklabelmode= "period", 
                        tickcolor= "darkgrey", gridcolor='darkgrey', dtick='M1',tickformat="%b\n%Y",
                        minor=dict(ticks="inside", showgrid=True, dtick=7*24*60*60*1000,  
                            tick0="2024-07-03", 
                            griddash='dot', 
                            gridcolor='lightgrey',
                            tickcolor = 'lightgrey'))
        fig.update_layout(yaxis={'dtick':1})
        
        # correct colors
        for i, trace in enumerate(fig.data):
            color_df = block_data[['annotation_text','color']].drop_duplicates().set_index('annotation_text')
            trace.marker.color = color_df.loc[trace.text]['color'].tolist()  # Assign colors from the 'color' column of block_data
            text_df = color_df.copy()
            text_df['text']=text_df.index.tolist()
            text_df.loc[text_df.index.str.contains('harvest|bedprep'),'text']=''
            trace.text = text_df.loc[trace.text]['text'].tolist()
            #trace.name =
        #fig.update_traces(textposition='auto', text = 'annotation_text') 
        #fig.for_each_trace(lambda t: t.update(text = []) if t.name in ['garden','bedprep'] else ())
        
        # legend
        legend_df = block_data[['FAMILY','color']].drop_duplicates()
        legend_df['x']=np.nan
        fig.add_traces(list(px.box(legend_df, x='x',y='x', color = 'FAMILY',color_discrete_sequence=legend_df['color'].tolist()).select_traces()))
        for trace in fig['data']: 
            if (not trace['name'] in legend_df.FAMILY.tolist()):
                trace['showlegend'] = False
            else:
                trace['showlegend'] = True
            
        fig.update_layout(showlegend = True, legend_title=None,legend=dict(orientation='h',y=-0.1,xanchor='center',x=0.5))        
        fig.update_xaxes(range=[datetime(2024, 3,1),datetime(2024, 12,31)])

        figures.append(fig)

    # # Create a subplot with a 3x2 grid arrangement for the plots
    # fig = make_subplots(rows=3, cols=2, subplot_titles=['Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5', 'Plot 6'])
   
    # # Add each plot to the corresponding position in the grid
    # fig.add_trace(figures[0]['data'][0], row=1, col=1)
    # fig.add_trace(figures[1]['data'][0], row=1, col=2)
    # fig.add_trace(figures[2]['data'][0], row=2, col=1)
    # fig.add_trace(figures[3]['data'][0], row=2, col=2)
    # fig.add_trace(figures[4]['data'][0], row=3, col=1)
    # #fig.add_trace(figures[5]['data'][0], row=3, col=2)

    return figures
