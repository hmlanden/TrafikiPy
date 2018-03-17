
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

#============IMPORT==============
csv_file_path = os.path.join('Resources', 'accidents_2014.csv')
traffic_df = pd.read_csv(csv_file_path)

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

plt.savefig('eightColorPalette.png')

# create color palette with 7 colors (for use with day of the week analysis)
sevenColorPalette = sns.color_palette('hls', 7)
seven = sns.palplot(sevenColorPalette)
plt.savefig('sevenColorPalette.png')

# create color palette with 5 colors (for use with road/light conditions)
fiveColorPalette = sns.color_palette('hls', 5)
five = sns.palplot(fiveColorPalette)

plt.savefig('fiveColorPalette.png')

# create color palette with 3 colors (for data by severity)
threeColorPalette = sns.color_palette('hls', 3)
three = sns.palplot(threeColorPalette)

# display color palettes
three
five
eight
twelve


# # Weather and Severity Correlation 

grouper_1 = traffic_df[['Weather Conditions','Accident Severity']]
weather_severity = grouper_1.groupby(by = 'Weather Conditions',as_index=False).sum()
plt = sns.barplot(weather_severity['Accident Severity'],weather_severity['Weather Conditions'])


# # Severity and Weather Correlation

grouper_a = traffic_df[['Weather Conditions','Accident Severity']]
weather_severity_1 = grouper_a.groupby(by = 'Accident Severity',as_index=False).count()
plt1 = sns.barplot(weather_severity_1['Accident Severity'],weather_severity_1['Weather Conditions'])


# # Light Condition and Severity Correlation

grouper_2 = traffic_df[['Light Conditions','Accident Severity']]
light_condition_severity = grouper_2.groupby(by = 'Light Conditions',as_index=False).sum()
plt = sns.barplot(light_condition_severity['Accident Severity'],light_condition_severity['Light Conditions'])


# # Severity Condition and Light Condition

grouper_b = traffic_df[['Light Conditions','Accident Severity']]
light_condition_severity_1 = grouper_b.groupby(by = 'Accident Severity',as_index=False).count()
plt = sns.barplot(light_condition_severity_1['Accident Severity'],light_condition_severity_1['Light Conditions'])


# # Road Type and Severity Correlation

grouper_3 = traffic_df[['Road Type','Accident Severity']]
road_type_severity = grouper_3.groupby(by = 'Road Type',as_index=False).sum()
plt = sns.barplot(road_type_severity['Accident Severity'],road_type_severity['Road Type'])


# # Severity and Road Type Correlation

grouper_c = traffic_df[['Road Type','Accident Severity']]
road_type_severity_1 = grouper_c.groupby(by = 'Accident Severity',as_index=False).count()
plt = sns.barplot(road_type_severity_1['Accident Severity'],road_type_severity_1['Road Type'])


# # Converting Weather Condition to Numbers 

weather_condition_number_list = []

for condition in traffic_df['Weather Conditions']:
    
    if (condition == 'Fine without high winds'):
        weather_condition_number_list.append(7)
        
    if (condition == 'Raining without high winds'):
        weather_condition_number_list.append(6)
        
    if (condition == 'Raining with high winds'):
        weather_condition_number_list.append(5)
        
    if (condition == 'Other'):
        weather_condition_number_list.append(4)
        
    if (condition == 'Fine with high winds'):
        weather_condition_number_list.append(3)
        
    if (condition == 'Fog or mist'):
        weather_condition_number_list.append(2)
    
    if (condition == 'Snowing without high winds'):
        weather_condition_number_list.append(1)
        
    if (condition == 'Snowing with high winds'):
        weather_condition_number_list.append(0)


# # Converting Road Type to Numbers

road_type_number_list = []

traffic_df['Road Type'].value_counts()

for road in traffic_df['Road Type']:
    
    if (road == 'Single carriageway'):
        road_type_number_list.append(4)
    
    if (road == 'Dual carriageway'):
        road_type_number_list.append(3)
        
    if (road == 'Roundabout'):
        road_type_number_list.append(2)
        
    if (road == 'One way street'):
        road_type_number_list.append(1)
        
    if (road == 'Slip road'):
        road_type_number_list.append(0)


# # Light Conditions to Numbers

light_condition_number_list = []

for condition in traffic_df['Light Conditions']:
    
    if (condition == 'Daylight: Street light present'):
        light_condition_number_list.append(4)

    if (condition == 'Darkness: Street lights present and lit'):
        light_condition_number_list.append(3)
        
    if (condition == 'Darkness: No street lighting'):
        light_condition_number_list.append(2)
        
    if (condition == 'Darkness: Street lighting unknown'):
        light_condition_number_list.append(1)
        
    if (condition == 'Darkness: Street lights present but unlit'):
        light_condition_number_list.append(0)
        


# In[33]:


training_data = pd.DataFrame({'Weather':weather_condition_number_list,
                              'Road Type':road_type_number_list,
                              'Light Condition':light_condition_number_list                          
                              })


# # Test Data

# In[34]:


import random    

weather_testing = []
road_testing = []
light_testing = []

for i in range (54147):    
    weather_testing.append(random.randrange(0,8,1))  
    road_testing.append(random.randrange(0,5,1))
    light_testing.append(random.randrange(0,5,1))  
  
     


# In[35]:


test_data = []
for i in range(54147):
    temp=[]
    temp.append(weather_testing[i])
    temp.append(road_testing[i])
    temp.append(light_testing[i])
    test_data.append(temp)      


# # Training Algorithm

# In[36]:


X = training_data[['Weather','Road Type','Light Condition']]
Y = traffic_df['Accident Severity']
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
prediction = clf.predict(test_data)


# In[39]:


csv_data = pd.DataFrame({'Weather Conditions': weather_condition_number_list, 
                         'Road Type': road_type_number_list,
                         'Light Condition': light_condition_number_list,
                         'Severity': traffic_df['Accident Severity']
                        })
csv_data.to_csv('Resources/regression.csv')

plt.savefig('threeColorPalette.png')

# display color palettes
plt.show(three)
plt.show(five)
plt.show(eight)
plt.show(twelve)

# ----------------------------------------------------------------------
# **Part 2: Basic Data Visualizations**
# This section contains basic data visualizations for the overall 
# dataset without deep analytical goals. 
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Part 2.1: Geographical heatmap
# ----------------------------------------------------------------------
# initialize plotly to work offline in jupyter notebook
import plotly
plotly.tools.set_config_file(world_readable=True)
plotly.offline.init_notebook_mode(connected=True)

import plotly.plotly as py
scl = [[0,'#5D56D3'], [0.5,'#7CD96E'], [1,'#CC655B']]

data = [dict(type = 'scattergeo',
             lon = traffic_df['Longitude'],
             lat = traffic_df['Latitude'],
             mode = 'markers',
             marker = dict(
                 size = 1,
                 opacity = 0.75,
                 reversescale = True,
                 autocolorscale = False,
                 symbol = 'circle',
                 colorscale = scl,
                 color = traffic_df['Accident Severity'],
                 cmax = 3,
                 colorbar=dict(
                     title='Accident Severity')))]

layout = dict(title = '<b>2014 Great Britain & Wales Traffic Accidents</b>',
              width=1000,
              height=1000,
              geo = dict(scope = 'globe',
                         projection=dict(type='eckert4'),
                         lonaxis = dict(showgrid = True,
                                        gridwidth = 0.5,
                                        range= [-6, 2.59],
                                        gridcolor='#000000',
                                        dtick = 5),
                         lataxis = dict(showgrid = True,
                                        gridwidth = 0.5,
                                        range = [49.48, 56],
                                        gridcolor ='#000000',
                                        dtick = 5),
            showland = True,
            landcolor = 'FFFFFF',
            subunitcolor = '#E5E5E5',
            countrycolor = '#000000',
        ))

# create figure
fig = dict(data=data, layout=layout)

# display plot
py.image.ishow(fig)

# ----------------------------------------------------------------------
# Part 2.1.1: Geographical heatmap for just London
# ----------------------------------------------------------------------
data = [dict(type = 'scattergeo',
             lon = traffic_df['Longitude'],
             lat = traffic_df['Latitude'],
             mode = 'markers',
             marker = dict(
                 size = 5,
                 opacity = 0.75,
                 reversescale = True,
                 autocolorscale = False,
                 symbol = 'circle',
                 colorscale = scl,
                 color = traffic_df['Accident Severity'],
                 cmax = 3,
                 colorbar=dict(
                     title='Accident Severity'))
        )]

layout = dict(title='<b>2014 London Accidents</b>',
              width=1000,
              height=1000,
              geo = dict(scope = 'globe',
                         projection=dict(type='eckert4'),
                         lonaxis = dict(
                             showgrid = True,
                             gridwidth = 0.5,
                             range= [-1, 1],
                             gridcolor='#000000',
                             dtick = 5),
                         lataxis = dict(
                             showgrid = True,
                             gridwidth = 0.5,
                             range = [50.5, 53],
                             gridcolor ='#000000',
                             dtick = 5),
                         showland = True,
                         landcolor = 'FFFFFF',
                         subunitcolor = '#E5E5E5',
                         countrycolor = '#000000'))

# create figure
fig = dict(data=data, layout=layout)

# display plot
py.image.ishow(fig)

# set up lists/dicts of months
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 
              'December']
monthLength_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
tripleMonthLength_list = [31, 31, 31, 28, 28, 28, 31, 31, 31, 30, 30, 30, 
                          31, 31, 31, 30, 30, 30, 31, 31, 31, 31, 31, 31,
                          30, 30, 30, 31, 31, 31, 30, 30, 30, 31, 31, 31]

# ----------------------------------------------------------------------
# Part 2.2: Count of Traffic Accidents by Month
# ----------------------------------------------------------------------

# set up data
trafficDataByMonth_df = traffic_df[['Date', 'Accident Index']].copy()
trafficDataByMonth_df.index = trafficDataByMonth_df['Date']
trafficDataByMonth_df = pd.DataFrame(trafficDataByMonth_df.resample('M').count()['Accident Index'])
trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Date', 
                                                             'Accident Index']]
trafficDataByMonth_df.rename(columns={'Accident Index':'Accident Count'}, 
                             inplace=True )

# add length column to allow normalization by month lengths
trafficDataByMonth_df['Month Length (Num Days)'] = monthLength_list

# set up x and y values
x_axis = trafficDataByMonth_df['Date']
y_axis = trafficDataByMonth_df['Accident Count']/trafficDataByMonth_df['Month Length (Num Days)']

# create bar plot
sns.set()
plt.rcParams['figure.figsize'] = [15,5]
plt.bar(trafficDataByMonth_df['Date'], 
        trafficDataByMonth_df['Accident Count'], 
        color=twelveColorPalette, width=15, align='center', linewidth=1, 
        edgecolor='black', tick_label=month_list, alpha=0.75)
plt.title("Normalized Accident Count by Month", size=16)
plt.xlabel("Month", size=13)
plt.ylabel("Normalized Accident Count", size=13)
plt.savefig('normalizedAccidentByMonth.png')

# change date column to month names
trafficDataByMonth_df['Date'] = month_list

# display results
plt.show()
trafficDataByMonth_df

# ----------------------------------------------------------------------
# Part 2.3: Grouped Bar chart of severity by month
# ----------------------------------------------------------------------
# create dataframe holding all data
accSevByMonth_df = traffic_df[['Date', 'Accident Index', 'Accident Severity']].copy()
accSevByMonth_df.index = accSevByMonth_df['Date']
accSevByMonth_df['Month'] = accSevByMonth_df.index.month
accSevByMonth_df = pd.DataFrame(accSevByMonth_df.\
                                groupby(['Month', 'Accident Severity']).\
                                count()['Accident Index'])
accSevByMonth_df.reset_index(inplace=True)
accSevByMonth_df[['Month', 
                  'Accident Severity', 
                  'Accident Index']]
accSevByMonth_df.rename(columns={'Accident Index':'Accident Count'}, 
                        inplace=True)

# add month length for normalization
accSevByMonth_df['Month Length'] = tripleMonthLength_list

# normalize
accSevByMonth_df['Accident Count'] = accSevByMonth_df['Accident Count']/accSevByMonth_df['Month Length']

# create bar plot
accidentSeverityByMonth_plt = sns.barplot(x='Month', y='Accident Count', 
                                          data=accSevByMonth_df, 
                                          hue='Accident Severity', 
                                          palette=threeColorPalette, 
                                          edgecolor='black', alpha=0.75, 
                                          linewidth=1)
plt.title("Normalized Accident Severity by Month", size=16)
plt.ylabel("Normalized Accident Count")
plt.savefig('normalizedAccidentSeverityByMonth.png')
plt.show(accidentSeverityByMonth_plt)

# create base dataframe to work with
accSevByMonth2_df = traffic_df[['Date', 'Accident Index', 'Accident Severity']].copy()
accSevByMonth2_df.index = accSevByMonth2_df['Date']
accSevByMonth2_df['Month'] = accSevByMonth2_df.index.month

# create three separate dataframes by severity (including column for normalization)
levelOneByMonth_df = pd.DataFrame(accSevByMonth2_df[accSevByMonth2_df['Accident Severity']==1].\
                                  groupby('Month').count()['Accident Index']).reset_index()
levelOneByMonth_df['Month Length'] = monthLength_list
levelTwoByMonth_df = pd.DataFrame(accSevByMonth2_df[accSevByMonth2_df['Accident Severity']==2].\
                                  groupby('Month').count()['Accident Index']).reset_index()
levelTwoByMonth_df['Month Length'] = monthLength_list
levelThreeByMonth_df = pd.DataFrame(accSevByMonth2_df[accSevByMonth2_df['Accident Severity']==3].\
                                    groupby('Month').count()['Accident Index']).reset_index()
levelThreeByMonth_df['Month Length'] = monthLength_list

# create stacked bar plot
plt.rcParams["figure.figsize"] = [16,8]
plt.bar(levelThreeByMonth_df['Month'], 
        levelThreeByMonth_df['Accident Index']/levelThreeByMonth_df['Month Length'], 
        color='#5D56D3', width=0.75, align='center', linewidth=1, 
        edgecolor='black', tick_label=month_list, alpha=0.75)
plt.bar(levelTwoByMonth_df['Month'], 
        levelTwoByMonth_df['Accident Index']/levelTwoByMonth_df['Month Length'], 
        bottom=levelThreeByMonth_df['Accident Index']/levelThreeByMonth_df['Month Length'],
        color='#7CD96E', width=0.75, align='center', linewidth=1, 
        edgecolor='black', tick_label=month_list, alpha=0.75)
plt.bar(levelOneByMonth_df['Month'], 
        levelOneByMonth_df['Accident Index']/levelOneByMonth_df['Month Length'], 
        bottom=[i+j for i,j in zip(levelThreeByMonth_df['Accident Index']/levelThreeByMonth_df['Month Length'], 
                                   levelTwoByMonth_df['Accident Index']/levelTwoByMonth_df['Month Length'])],
        color='#CC655B', width=0.75, align='center', linewidth=1, 
        edgecolor='black', tick_label=month_list, alpha=0.75)

# format bar plot
plt.title("Normalized Accident Severity by Month", size=16)
plt.xlabel("Month", size=13)
plt.ylabel("Normalized Accident Count", size=13)

# add legend
import matplotlib.patches as mpatches
sev1 = mpatches.Patch(color='#CC655B', label='1', alpha=0.75)
sev2 = mpatches.Patch(color='#7CD96E', label='2', alpha=0.75)
sev3 = mpatches.Patch(color='#5D56D3', label='3', alpha=0.75)
plt.legend(handles=[sev1, sev2, sev3], 
           loc='best', title='Accident Severity')
plt.savefig('stackedSeverityByMonth.png')
plt.show()
