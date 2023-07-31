
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
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt

def calc_rent_increase(rent,years=10,proptax=False):
    rent_tracker = [rent]
    for c in list(range(0,years)): 
        rent2 = rent_tracker[-1] + (rent_tracker[-1]*0.015)
        rent_tracker.append(rent2)
    if proptax == True:
        byear = rent_tracker
    else: 
        byear = [x*12 for x in rent_tracker]
    return sum(byear)

def calc_etf(rent,years=10):
    rent_tracker = [rent]
    for c in list(range(0,years)): 
        rent2 = rent_tracker[-1] + (rent_tracker[-1]*0.0624)
        rent_tracker.append(rent2)
    byear = rent_tracker

    return byear[-1]

def calc_buy_costs(price,maint,ptax,last_sold1,last_sold_price,\
                   inf_years=list(range(2000,2023)),inf_val=[64.61,59.27,57.36,53.37,49.57,\
                        47.05,43.46,40.48,36.22,36.58,\
                        35.28,31.22,29.28,27.8,24.86,\
                        23.58,21.77,20.55,17.66,15.33,\
                        14.58,11.17,2.81],\
                   years=10,percent_down=0.20,mortgage=0.05,yloan=25):
    # Calculate mortgage payment
    
    interest = 0.05 / 12
    loan = price * (1 - percent_down)
    total_number_of_payments = yloan * 12
    number_of_payments_paid = years
    mortgageP = loan * (interest * (1 + interest)**total_number_of_payments) \
                       / ((1 + interest)**total_number_of_payments - 1)
    remaining = loan * ((1 + interest)**total_number_of_payments - (1 + interest)**number_of_payments_paid) \
                        / ((1 + interest)**total_number_of_payments - 1)

    total= mortgageP 


    # Calculate Appreciation (minus)

    ind = inf_years.index(last_sold1)
    inflation = inf_val[ind]
    before_price = last_sold_price #+ last_sold_price*(inflation/100)
    app = (price-before_price) #/ (2023-2013)
    #app = app * years
    

    # Maint Fees

    maint_calc = calc_rent_increase(float(maint),years=10)

    costs_maint = mortgageP + maint_calc 
    

    # Property Tax

    ptax_calc = calc_rent_increase(float(ptax),years=10,proptax=True)

    costs_ptax = costs_maint + ptax_calc

    cost_adj = -costs_ptax + price + app - remaining

    agent_fees = price * 0.05

    cost_adj = cost_adj+(-agent_fees)
    down_pay = price*percent_down

    return round(cost_adj,1),round(app,1),round(down_pay,1),round(mortgageP + float(maint),1)

def get_potential_app(bdata):
    bdata = bdata[bdata['latitude'].notna()]
    bdata_train = bdata[bdata['Last Sold '].notna()]

    bdata_train = bdata_train[['Cost','Maintenance','Bedrooms','Sq Foot','Last Sold ',\
                               'Price Last Sold','latitude','longitude']]


    reg = RandomForestRegressor(
        n_estimators=100, max_features='sqrt', random_state=1)
    X_pd = bdata_train[['Cost','Maintenance','Bedrooms','Sq Foot','Last Sold ',\
                               'latitude','longitude']]
    X_train = np.array(bdata_train[['Cost','Maintenance','Bedrooms','Sq Foot','Last Sold ',\
                               'latitude','longitude']])
    y = np.array(bdata_train['Price Last Sold']).reshape(-1, 1)
    bdata['Est_2003'] = [2013]*len(bdata)
    X_test = np.array(bdata[['Cost','Maintenance','Bedrooms','Sq Foot','Est_2003',\
                               'latitude','longitude']])

    X_test_check = np.array(bdata_train[['Cost','Maintenance','Bedrooms','Sq Foot','Last Sold ',\
                               'latitude','longitude']])

    freg = reg.fit(X_train, y)
    Zi = freg.predict(X_test)
    Zc = freg.predict(X_test_check)
    bdata['If_Sold_Last_2003_Price'] = [round(x) for x in list(Zi)]

    y_true = y 
    y_pred = Zc 

    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_true, y_pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_true, y_pred, squared=False))
    print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_true, y_pred))
    print('Explained Variance Score:', metrics.explained_variance_score(y_true, y_pred))
    print('Max Error:', metrics.max_error(y_true, y_pred))
    print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_true, y_pred))
    print('Median Absolute Error:', metrics.median_absolute_error(y_true, y_pred))
    print('R^2:', metrics.r2_score(y_true, y_pred))
    print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_true, y_pred))
    print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_true, y_pred))

    #from sklearn.inspection import PartialDependenceDisplay

    #PartialDependenceDisplay.from_estimator(freg,X_pd,['Cost','Bedrooms',\
                               #'latitude','longitude'],kind='both')
    #plt.show()
    print(bdata[bdata['Last Sold '] == 2013][['Price Last Sold','If_Sold_Last_2003_Price']])
    return bdata

    
if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    inflation_years = list(range(2000,2023))
    inflation_values = [64.61,59.27,57.36,53.37,49.57,\
                        47.05,43.46,40.48,36.22,36.58,\
                        35.28,31.22,29.28,27.8,24.86,\
                        23.58,21.77,20.55,17.66,15.33,\
                        14.58,11.17,2.81]
     
    rcsv = pd.read_csv('geocode_rent.csv',sep=',')
    rcsv['total_rent'] = rcsv['Rent'].apply(calc_rent_increase)
    rcsv['total_rent'] = 0-rcsv['total_rent']
    print(rcsv)

    
    bcsv = pd.read_csv('geocode_buy.csv',sep=',')
    bcsv = bcsv[bcsv['Last Sold '].notna()]

    bcsv['total_buy'],bcsv['appreciation'],bcsv['down_pay'],bcsv['mpay'] = np.vectorize(calc_buy_costs)(bcsv['Cost'],bcsv['Maintenance'],bcsv['Property Tax'],\
                                                     bcsv['Last Sold '],bcsv['Price Last Sold'])
  
    bcsv1 = pd.read_csv('geocode_buy.csv',sep=',')
    bcsv1 = bcsv1[bcsv1['latitude'].notna()]
    bcsv2 = get_potential_app(bcsv1)

    print(bcsv2)
    
    bcsv2['total_buy2'],bcsv2['appreciation'],bcsv2['down_pay'],bcsv2['mpay'] = np.vectorize(calc_buy_costs)(bcsv2['Cost'],bcsv2['Maintenance'],bcsv2['Property Tax'],\
                                                     bcsv2['Est_2003'],bcsv2['If_Sold_Last_2003_Price'])
  
    print(bcsv2)

    #Map the condos

    fig = px.scatter_mapbox(bcsv2, lat="latitude", lon="longitude",
                        color="appreciation", zoom=10,
                        color_continuous_scale=px.colors.diverging.Spectral_r,
                        mapbox_style='open-street-map')
    fig.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers'))

    plot(fig, auto_open=True)

    #Averages
    
    average_appt_cost = np.nanmean(rcsv['total_rent'])
    print(average_appt_cost)

    average_condo_cost = np.nanmean(bcsv2['total_buy2'])
    print(average_condo_cost)

    bcsv2['dp_etf'] = bcsv2['down_pay'].apply(calc_etf) #-bcsv2['down_pay']
    bcsv2['diff'] = bcsv2['total_buy2']-bcsv2['dp_etf']
    pd.options.display.max_columns = None
    print(bcsv2[bcsv2['diff'] < 0][['Address','Cost','Bedrooms','total_buy2','appreciation','down_pay','dp_etf','mpay']])

    gic_better = bcsv2[bcsv2['diff'] < 0][['Address','Cost','Bedrooms','total_buy2','appreciation','down_pay','dp_etf','mpay']]
    gic_better_p = len(gic_better) / len(bcsv2) *100
    print('Percent of condo investments with 20percent down that are worse than ETF %s'%(gic_better_p))

    #Top 10% Performers

    print(np.percentile(bcsv2['total_buy2'],95))
    p90 = np.percentile(bcsv2['total_buy2'],95)
    print(bcsv2[bcsv2['total_buy2'] >= p90])

    #Bottom 10

    print(np.percentile(bcsv2['total_buy2'],10))
    p10 = np.percentile(bcsv2['total_buy2'],10)
    print(bcsv2[bcsv2['total_buy2'] <= p10])
    
    
