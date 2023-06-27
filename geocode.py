
#coding: utf-8

"""
Summary
-------
Code to geocode addresses in Toronto. 
"""

import geopandas as gpd
import pandas as pd 
import numpy as np
import os, sys
from pyproj import CRS, Transformer
import fiona


import warnings
warnings.filterwarnings('ignore')

from geopy.extra.rate_limiter import RateLimiter

from geopy.geocoders import Nominatim

def find_loc(file_name,output_file_name):

    #Test the geocoder
    locator = Nominatim(user_agent="myGeocoder")
    location = locator.geocode("233 COLERIDGE AVE, Toronto, Ontario")
    print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))

    
    geocode = RateLimiter(locator.geocode, min_delay_seconds=0.5)
    
    #Get Toronto address
    lat = []
    lon = []
    #Read in the downloaded list of parking ticket addresses (first 100)
    add_csv = pd.read_csv('parking-tickets-2022/'+file_name+'.csv',sep=',')[0:100]
    #The file needs to be in the same folder as the Python script, in a folder called "parking-tickets-2022"
    count = 1
    #Loop through the addresses
    for ad in list(add_csv['location2']): #location2 is the address column in the ticket data
        location = locator.geocode(ad+", Toronto, Ontario")
        print(count)
        print(ad+", Toronto, Ontario")
        try: 
            lat.append(location.latitude)
            lon.append(location.longitude)
            print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
        except (AttributeError,TimeoutError):
            lat.append('NA')
            lon.append('NA')
        count += 1

    add_csv['latitude'] = lat
    add_csv['longitude'] = lon

    #Output the found latitude and longitude to a new file 

    add_csv.to_csv(output_file_name+'.csv',sep=',')
 
  
if __name__ == "__main__":

    file_name = 'jan'
    output_file_name = 'new_jan_2022'

    find_loc(file_name,output_file_name)

