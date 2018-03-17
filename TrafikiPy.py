


# ----------------------------------------------------------------------
# **Part 1: File Set Up**
# ----------------------------------------------------------------------

#===========DEPENDENCIES=============
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import requests

pd.set_option('display.max_columns', None)


# In[2]:


#============IMPORT==============
csv_file_path = os.path.join('Resources', 'accidents_2014.csv')
traffic_df = pd.read_csv(csv_file_path)


# In[3]:


#============DROP BLANK COLUMNS===========

traffic_df.dropna(
    axis=1,
    how='all',
    inplace=True
)

#============DROP BLANK ROWS WITH BLANK VALUES==========

traffic_df['Junction_Control'].replace(
    np.nan, 'None', inplace=True)

traffic_df.replace(
    '', np.nan, inplace=True)

traffic_df.replace(
    'Unknown', np.nan, inplace=True)

traffic_df.dropna(axis=0, inplace=True)

#===========REPLACING ERRANT/MISPELLED VALUES===============

traffic_df['Light_Conditions'].replace(
    'Darkeness: No street lighting',
    'Darkness: No street lighting', 
    inplace=True
)

traffic_df['Pedestrian_Crossing-Physical_Facilities'].replace(
    'non-junction pedestrian crossing',
    'Non-junction Pedestrian Crossing', 
    inplace=True
)

#===========RENAMING COLUMNS===============

traffic_df.rename(columns=
    {'Accident_Index' : 'Accident Index',
     'Longitude' : 'Longitude', 
     'Latitude' : 'Latitude', 
     'Police_Force' : 'Police Force', 
     'Accident_Severity' : 'Accident Severity', 
     'Number_of_Vehicles' : 'Number of Vehicles', 
     'Number_of_Casualties' : 'Number of Casualties', 
     'Date' : 'Date', 
     'Day_of_Week' : 'Day of Week', 
     'Time' : 'Time', 
     'Local_Authority_(District)' : 'Local Authority District', 
     'Local_Authority_(Highway)' : 'Local Authority Highway', 
     '1st_Road_Class' : '1st Road Class', 
     '1st_Road_Number' : '1st Road Number', 
     'Road_Type' : 'Road Type', 
     'Speed_limit' : 'Speed Limit', 
     'Junction_Control' : 'Junction Control', 
     '2nd_Road_Class' : '2nd Road Class', 
     '2nd_Road_Number' : '2nd Road Number', 
     'Pedestrian_Crossing-Human_Control' : 'Pedestrian Crossing Human Control', 
     'Pedestrian_Crossing-Physical_Facilities' : 'Pedestrian Crossing Physical Facilities', 
     'Light_Conditions' : 'Light Conditions', 
     'Weather_Conditions' : 'Weather Conditions', 
     'Road_Surface_Conditions' : 'Road Surface Conditions', 
     'Special_Conditions_at_Site' : 'Special Conditions at Site', 
     'Carriageway_Hazards' : 'Carriageway Hazards', 
     'Urban_or_Rural_Area' : 'Urban or Rural Area', 
     'Did_Police_Officer_Attend_Scene_of_Accident' : 'Police Attended Scene of Accident', 
     'LSOA_of_Accident_Location' : 'LSOA of Accident Location', 
     'Year' : 'Year', 
    }, inplace=True)

# format Date in Datetime format
traffic_df['Date'] = pd.to_datetime(traffic_df['Date'], format='%d/%m/%y')

# display cleaned file
traffic_df.head()

# ----------------------------------------------------------------------
# **Part 2: Set up overall formatting**
# ----------------------------------------------------------------------
# create color palette with 12 colors (for use with monthly data)
twelveColorPalette = sns.color_palette('hls', 12)
twelve = sns.palplot(twelveColorPalette)

# create color palette with 8 colors (for use with weather conditions)
eightColorPalette = sns.color_palette('hls', 8)
eight = sns.palplot(eightColorPalette)

# create color palette with 5 colors (for use with road/light conditions)
fiveColorPalette = sns.color_palette('hls', 5)
five = sns.palplot(fiveColorPalette)

# create color palette with 3 colors (for data by severity)
threeColorPalette = sns.color_palette('hls', 3)
three = sns.palplot(threeColorPalette)

# display color palettes
three
five
eight
twelve


urban = traffic_df[traffic_df["Urban or Rural Area"] == 1]
rural = traffic_df[traffic_df["Urban or Rural Area"] == 2]

rural_mean_1 = rural.groupby(["Date"]).mean()["Accident Severity"]
rural_mean_2 = rural.groupby(["Date"]).mean()["Number of Casualties"]
rural_count_3 = rural.groupby(["Date"]).count()["Accident Index"]

urban_mean_1 = urban.groupby(["Date"]).mean()["Accident Severity"]
urban_mean_2 = urban.groupby(["Date"]).mean()["Number of Casualties"]
urban_count_3 = urban.groupby(["Date"]).count()["Accident Index"]


plt.rcParams["figure.figsize"] = [16,9]

plt.title("Accident Severity and Average Casualty by City Type", size=20)
plt.ylabel("Average Severity", size=20)
plt.xlabel("Average Casualties", size=20)

sns.set()
plt.scatter(rural_mean_2,
            rural_mean_1,
            color="#DACF68",
            s=rural_count_3,
            edgecolor="black", linewidths= 0.1,
            alpha=0.8, label="Rural")

plt.scatter(urban_mean_2,
            urban_mean_1,
            color="#8757D4",
            s=urban_count_3,
            edgecolor="black", linewidths=0.1, marker="^", 
            alpha=0.8, label="Urban")

plt.legend(title='City Type', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)
plt.savefig('Severity and Casualty by City Type.png')
plt.show()


'''
severity_1 = traffic_df[traffic_df["Accident Severity"] == 1]
severity_2 = traffic_df[traffic_df["Accident Severity"] == 2]
severity_3 = traffic_df[traffic_df["Accident Severity"] == 3]

severity_1_mean_1 = severity_1.groupby(["Local Authority District"]).mean()["Number of Casualties"]
severity_1_mean_2 = severity_1.groupby(["Local Authority District"]).sum()["Number of Casualties"]
severity_1_count_2 = severity_1.groupby(["Local Authority District"]).count()["Accident Index"]

severity_2_mean_1 = severity_2.groupby(["Local Authority District"]).mean()["Number of Casualties"]
severity_2_mean_2 = severity_2.groupby(["Local Authority District"]).sum()["Number of Casualties"]
severity_2_count_2 = severity_2.groupby(["Local Authority District"]).count()["Accident Index"]

severity_3_mean_1 = severity_3.groupby(["Local Authority District"]).mean()["Number of Casualties"]
severity_3_mean_2 = severity_3.groupby(["Local Authority District"]).sum()["Number of Casualties"]
severity_3_count_2 = severity_3.groupby(["Local Authority District"]).count()["Accident Index"]

sns.set()
plt.rcParams["figure.figsize"] = [16,9]

plt.ylim([0,500])
plt.xlim([1,5])
plt.title("Latitude vs. Max Casualties", size=20)
plt.ylabel("Number of Casualties in Authority District", size=20)
plt.xlabel("Average Causalties in Authority District", size=20)


plt.scatter(severity_1_mean_1,
            severity_1_mean_2, 
            color="#5D56D3", 
            edgecolor="black", linewidths= 0.1,
            alpha=0.75, label="Severity 1")

plt.scatter(severity_2_mean_1,
            severity_2_mean_2,  
            color="#7CD96E",
            edgecolor="black", linewidths= 0.1,
            alpha=0.75, label="Severity 2")

plt.scatter(severity_3_mean_1,
            severity_3_mean_2,  
            color="#CC655B", 
            edgecolor="black", linewidths= 0.1,
            alpha=0.75, label="Severity 3")
'''

