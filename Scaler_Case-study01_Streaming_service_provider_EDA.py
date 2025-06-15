## Business Case-Study 01
## EDA _ Streaming services provider
## JEEVAN JAYANT ITOLIKAR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import matplotlib.pyplot as plt
import math
import seaborn as sns

company_A_data = pd.read_csv(r"D:\IMP\Studies\Machine_Learning\Scaler_Major_Project\Case_Study01_Streaming_Services_Provider_EDA\Streaming_services_provider_data.csv")

# reviewing the datatypes for the columns in the initial dataframe
company_A_data.info()

# Different value-counts for type, country, rating, duration in the dataframe
columns_to_check_value_counts = ['type', 'country','rating', 'duration','listed_in']
#columns_to_check_value_counts = ['listed_in','director']
#columns_to_check_value_counts = ['release_year', 'date_added']

# dataframe and value counts for only tv-shows
columns_to_check_value_counts = ['country','rating', 'duration','release_year']
company_A_data_TV = company_A_data[company_A_data['type'] == 'TV Show']



## Visual analysis of value_counts- top 5 only
len_charts = len(columns_to_check_value_counts)
rows = math.ceil(len_charts / 2)  # Ensuring correct number of rows
fig, axes = plt.subplots(rows, 2, figsize=(12, 8))  
axes = axes.flatten()

def function_to_check_value_counts (dataframe,columns_to_check_value_counts):
    for i, col in enumerate(columns_to_check_value_counts):
        value_counts = dataframe[col].value_counts().head(5)
        #value_counts = dataframe[col].value_counts().head(5).sort_index(ascending=True) 
        bars_chart = value_counts.plot(kind="bar", color="skyblue",ax = axes[i]) ## Plotting barcharts in subplots
        value_counts_unique = value_counts.count() 
        axes[i].set_title(f"Top {value_counts_unique} value counts for {col}") ## setting up title, labels
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=45)

        for bar in bars_chart.patches:
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()/2, f'{bar.get_height()}', ha='center', va = 'bottom')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


function_to_check_value_counts(company_A_data_TV,columns_to_check_value_counts)

## impact of removing all the missing values from dataframe
company_A_data_unique=company_A_data.dropna().reset_index(drop=True)
Number_of_entries_removed = company_A_data["show_id"].count() - company_A_data_unique["show_id"].count()
print(f'Number of entries removed : {Number_of_entries_removed} (out of {company_A_data["show_id"].count()})')

print("Significant amount of entries getting dropped, alternative approaches to be implemented!")

columns_to_fill_wMode = ['country','date_added','rating']
columns_to_fill_NA = ['director','cast']

for col in columns_to_fill_wMode:
    mode_value = company_A_data[col].mode()[0]
    company_A_data[col].fillna(mode_value, inplace=True)

for col in columns_to_fill_NA:
    company_A_data[col].fillna('NA', inplace=True)

## impact of removing all the missing values from dataframe
company_A_data_unique=company_A_data.dropna().reset_index(drop=True)
Number_of_entries_removed = company_A_data["show_id"].count() - company_A_data_unique["show_id"].count()
print(f'Number of entries removed : {Number_of_entries_removed} (out of {company_A_data["show_id"].count()})')


company_A_data_unique_Movies = company_A_data_unique[company_A_data_unique['type'] == 'Movie']
company_A_data_unique_TVShow = company_A_data_unique[company_A_data_unique['type'] == 'TV Show']

# Univariate and Bivariate analysis
company_A_data_unique['release_year'].hist(bins=30, color="skyblue")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.title("Distribution of Movies/TV Shows by Release Year")
plt.show()

company_A_data_unique_Movies['release_year'].hist(bins=30, color="skyblue")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.title("Distribution of Movies by Release Year")
plt.show()

company_A_data_unique_TVShow['release_year'].hist(bins=30, color="skyblue")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.title("Distribution of TV Shows by Release Year")
plt.show()

company_A_data_unique['type'].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["gold", "cyan"])
plt.title("Distribution of Movies vs TV Shows")
plt.ylabel("")
plt.show()

sns.countplot(x="rating", hue="type", data=company_A_data_unique, palette="Set2")
plt.xticks(rotation=45)
plt.title("Content Ratings for Movies vs TV Shows")
plt.show()



## Adding few parameters for analysis
company_A_data_unique['show_id'] = company_A_data_unique['show_id'].apply(lambda x : x[1:]).astype('int')
company_A_data_unique['added_month'] = company_A_data_unique['date_added'].apply(lambda x: str(x).strip(" ").split(",")[0].split(" ")[0])
company_A_data_unique['month']= company_A_data_unique['added_month'].apply(lambda x : datetime.datetime.strptime(x,"%B")).dt.month
company_A_data_unique['day'] = company_A_data_unique['date_added'].apply(lambda x: str(x).strip(" ").split(",")[0].split(" ")[1]).astype('int')
company_A_data_unique['year'] = company_A_data_unique['date_added'].apply(lambda x: str(x).strip(" ").split(",")[1]).astype('int')

company_A_data_unique['Date_Added'] = pd.to_datetime(company_A_data_unique[['month','day','year']])

company_A_data_unique_Movies1 = company_A_data_unique[company_A_data_unique['type'] == 'Movie']
company_A_data_unique_TVShow1 = company_A_data_unique[company_A_data_unique['type'] == 'TV Show']

# Count movies added per month
monthly_counts = company_A_data_unique_Movies1['month'].value_counts().sort_index()

sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Number of Releases")
plt.title("Best Month for Movie Launch")
plt.xticks(ticks=range(0,12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()

# Count tv-shows added per month
monthly_counts_shows = company_A_data_unique_TVShow1['month'].value_counts().sort_index()

sns.barplot(x=monthly_counts_shows.index, y=monthly_counts_shows.values, palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Number of Releases")
plt.title("Best Month for TV Show Launch")
plt.xticks(ticks=range(0,12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()

company_A_data_unique.drop(labels=['added_month','day','year','month','date_added','description'],axis=1,inplace=True)


company_A_data_unique_concat = company_A_data_unique.copy()

company_A_data_unique['listed_in'] = company_A_data_unique['listed_in'].str.strip(" ").str.split(",")
company_A_data_unique = company_A_data_unique.explode(column='listed_in')

company_A_data_unique['listed_in'].value_counts().head(5).plot(kind="bar", color="lightcoral")
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Top Listed Categories")
plt.xticks(rotation=0)
plt.show()

company_A_data_unique['cast'] = company_A_data_unique['cast'].str.strip(" ").str.split(",")
company_A_data_unique = company_A_data_unique.explode(column='cast')

company_A_data_unique['director'] = company_A_data_unique['director'].str.strip(" ").str.split(",")
company_A_data_unique = company_A_data_unique.explode(column='director')

company_A_data_unique['country'] = company_A_data_unique['country'].str.strip(" ").str.split(",")
company_A_data_unique = company_A_data_unique.explode(column='country')

company_A_data_unique.describe(include='all',datetime_is_numeric=True)



## Movies_vs_TV-shows comparison
movie_count = (company_A_data_unique_concat['type'] == "Movie").sum()
TV_Show_count = (company_A_data_unique_concat['type'] == "TV Show").sum()
ss_unique_Movies= company_A_data_unique_concat[company_A_data_unique_concat['type']=='Movie']
ss_unique_TV_Shows= company_A_data_unique_concat[company_A_data_unique_concat['type']=='TV Show']

mean_Movie_duration = round(ss_unique_Movies['duration'].apply(lambda x : x[:-4]).astype('int').mean(),2)
mean_TV_Show_Seasons = round(ss_unique_TV_Shows['duration'].apply(lambda x : x[:1]).astype('int').mean(),1)
mean_release_year_Movies = int(round(ss_unique_Movies['release_year'].mean(),0))
mean_release_year_TV_Shows = int(round(ss_unique_TV_Shows['release_year'].mean(),0))

###Insights
print(f'Total number of Movies listed : {movie_count}')
print(f'Total number of TV Show listed : {TV_Show_count}')
print(f'Average duration of the Movies listed (in mins) : {mean_Movie_duration}, release_year average: {mean_release_year_Movies}')
print(f'Average number of seasons for TV shows listed :{mean_TV_Show_Seasons}, release_year average : {mean_release_year_TV_Shows}')
print(f'Movies listed contain entries from 1942 release year, TV shows from 1990')

ss_unique_Movies['Movie_count'] = ss_unique_Movies.groupby(by=['release_year']).count()['show_id']

image = ss_unique_Movies['Movie_count'].dropna().plot(xlabel = "Release_year",ylabel = "Movie_count",marker="x")
plt.show()

image2 = ss_unique_Movies['Movie_count'].dropna().tail(15).plot(xlabel = "Release_year",ylabel = "Movie_count",marker="o")
x_ticks_image2 = image2.get_xticks()
image2.set_xticks(x_ticks_image2)
image2.set_xticklabels([f"{int(round(x, 0))}" for x in x_ticks_image2])
plt.show()