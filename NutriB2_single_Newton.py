#!/usr/bin/env python
# coding: utf-8

# # **Automatization Process for LHI**

# **This code is designed to create systematic process for the automatization of Landscape Heterogeneity Index calculation**<br><br>
# **Baturalp Arisoy, TUM Msc Geodesy and Geoinformation, 2022**<br>**NutriB2**

# This original script has been modified by Isaac Newton Kissiedu, MSc. Cartography Student at TUM. Contact (isaacnewtonfx@gmail.com)

import os,arcpy,time,sys
import pandas as pd
from arcpy.ia import *
from arcpy.sa import *


start_time = time.time()



########## PROGRAM SETTINGS ##########

## Parent Directory. 
# Change this parameter to match with your directory path. All paths in this code will reference it!
# Path must end with \\
PARENT_DIR = "C:\\HIWI_TASK\\Part_2\\Python\\"

## Get Plot By Plot Number 
# Loads the specified image from the DOP20s folder and processes it
# NB: Only select your plot no<br>If the plot number is between 1-9, always enter it with 0<br><br>**E.g.**<br> **AEG01 - Correct**<br> **AEG1 - Incorrect**
PLOT = 'AEG03'



## Point of Interest
# we will create a buffer around this point
LON = 9.53237875
LAT = 48.4088815

## Projection
EPSG_CODE_GEOGRAPHIC = 4326  #WGS 1984, FOR THE WORLD
EPSG_CODE_PROJECTED  = 32632 #UTM ZONE 32N, MUST BE SET BASED ON THE IMAGE BEING PROCESSED

## Buffer radius (in meters)
BUFFER_RADIUS = 250 # 500m diameter was required in task

#---------- END PROGRAM SETTINGS ----------#



########## PREPROCESSING ##########
print("Running Preprocessing")


## Setting up environement
# environments = arcpy.ListEnvironments("*workspace")
# print(environments)

arcpy.env.overwriteOutput = True
#arcpy.env.addOutputsToMap = True
arcpy.env.workspace = PARENT_DIR + 'Workspace'
arcpy.env.scratchWorkspace  = PARENT_DIR + 'Workspace'
print('Default Workspace of Python Code text file: ' + os.getcwd())
print('Current Workspace for geoprocesing: ' + arcpy.env.workspace)


## Set spatial references
crs_geog = arcpy.SpatialReference(EPSG_CODE_GEOGRAPHIC)
crs_proj = arcpy.SpatialReference(EPSG_CODE_PROJECTED)


## Check required extensions
arcpy.CheckOutExtension("ImageAnalyst")


## Fix plot name (not neccessary right now)
# if PLOT[3] == '0':
#     PLOT = PLOT.replace('0', '')
# print(PLOT)
    

## Create Point Shapefile
# arcpy.CreateFeatureclass_management returns a Result object; take the first item, which is the full path
fc_poi = arcpy.CreateFeatureclass_management(PARENT_DIR + "Vector", 'poi.shp', 'POINT', None, None, None, crs_geog)[0]
arcpy.AddField_management(fc_poi, 'x', 'DOUBLE')
arcpy.AddField_management(fc_poi, 'y', 'DOUBLE')
with arcpy.da.InsertCursor(fc_poi, ['SHAPE@', 'x', 'y']) as cur:
    p = arcpy.Point(LON, LAT)
    cur.insertRow([p, LON, LAT])


## Project the Point of Interest
fc_poi_proj = PARENT_DIR + "Vector\\poi_proj.shp"
arcpy.Project_management(fc_poi, fc_poi_proj, crs_proj)


## Create a buffer
fc_poi_proj_buf = PARENT_DIR + "VECTOR\\poi_proj_buf.shp"
arcpy.analysis.Buffer(fc_poi_proj, fc_poi_proj_buf, "{} Meters".format(BUFFER_RADIUS))

## Clip training samples with buffer. 
# Otherwise, TrainSupportVectorMachineClassifier will throw a very strange error "Workspace does not exist"
# No training sample must go outside the image used for the training
training_samples = PARENT_DIR + "Vector\\training_samples_{}.shp".format(PLOT)
# training_samples_clipped = PARENT_DIR + "Vector\\training_samples_{}_clipped.shp".format(PLOT)
# arcpy.analysis.Clip(training_samples, fc_poi_proj_buf, training_samples_clipped)


# ## Clip main image with buffer
# in_ras  = PARENT_DIR + "Raster\\{}.tif".format(PLOT)
# out_ras_clipped = PARENT_DIR + "Raster\\{}_clipped.tif".format(PLOT)
# outExtractByMask = ExtractByMask(in_ras, fc_poi_proj_buf)
# outExtractByMask.save(out_ras_clipped)


# ## Do image segmentation on the clipped image to simplify the image by grouping similar pixels together
# in_ras  = PARENT_DIR + "Raster\\{}_clipped.tif".format(PLOT)
# segemented_ras = PARENT_DIR + "Raster\\{}_seg.tif".format(PLOT)
# if not os.path.exists(segemented_ras):
#     print("Running image segmentation...")
    # seg_raster = SegmentMeanShift(in_ras, "15", "15", "20", "#")
#     seg_raster.save(segemented_ras)
#     print("Done running image segmentation...")


## Do image segmentation on the main image to simplify the image by grouping similar pixels together
in_ras  = PARENT_DIR + "Raster\\{}.tif".format(PLOT)
segemented_ras = PARENT_DIR + "Raster\\{}_seg.tif".format(PLOT)
if not os.path.exists(segemented_ras):
    print("Running image segmentation...")
    seg_raster = SegmentMeanShift(in_ras, "15", "15", "20", "#")
    seg_raster.save(segemented_ras)
    print("Done running image segmentation...")

# print("Done running Preprocessing")
# #---------- END PREPROCESSING ----------#



########## PROCESSING ##########
print("Running Processing")

## Parameters for TrainSupportVectorMachineClassifier
input_rasterLayer       = segemented_ras
input_train_features    = training_samples
output_definition       = PARENT_DIR + "Output\\class_def_{}.ecd".format(PLOT)
input_additional_raster = "#"
max_samples_per_class   = "#" #Default is 500
used_attributes         = "COLOR;MEAN;STD"


## Support Vector Machine (SVM) Classification

## Training SVM
# It takes about 20min on my Core i7 4th gen processor and 16GB RAM laptop!
# Run this code block only if class definition file does not exist. 
if not os.path.exists(output_definition):
    print("Running TrainSupportVectorMachineClassifier...")

    TrainSupportVectorMachineClassifier(input_rasterLayer, 
                                        input_train_features,
                                        output_definition,
                                        input_additional_raster, 
                                        max_samples_per_class,
                                        used_attributes)
                                        
    print("Done running TrainSupportVectorMachineClassifier")

## Classification and export geoTIFF
out_tif_file = PARENT_DIR + "Output\\SVM\\classified_raster_{}.tif".format(PLOT)

print("Running ClassifyRaster...")
classified_raster = ClassifyRaster(input_rasterLayer, output_definition, "#")
classified_raster = classified_raster.save(out_tif_file)
print("Done running ClassifyRaster")


# ## Clip classified image with buffer
out_ras_clipped = PARENT_DIR + "Output\\SVM\\classified_raster_{}_clipped.tif".format(PLOT)
outExtractByMask = ExtractByMask(out_tif_file, fc_poi_proj_buf)
outExtractByMask.save(out_ras_clipped)


## Calculate class summary table
out_table = PARENT_DIR + "Output\\Summary\\summary_classes_{}.csv".format(PLOT)
SummarizeCategoricalRaster(out_tif_file, out_table)

# calculate classes proportions
data = pd.read_csv(out_table)

# organize the data into vertical columns
data = data.T 

# create a sequential index and make the class names become a variable
data.reset_index(inplace=True) 

# set new col names
data.columns = ["classes", "count_pixels"] 

# total pixels in the image
all_pixels = data['count_pixels'].sum() 

# calculate a new column for the proportion of pixels per class
data['class_proportion'] = round( (data['count_pixels'] / all_pixels) * 100, 2)

# save the final summary table
data.to_csv(PARENT_DIR + "Output\\Summary\\summary_classes_final_{}.csv".format(PLOT))

end_time = time.time()

# get the execution time
elapsed_time = end_time - start_time

if elapsed_time < 60:
    print('Execution time:', elapsed_time, 'seconds')
else:
    print('Execution time:', round(elapsed_time/60, 2), 'minutes')

print("Done running Processing")
print("Program has ended successfully")
#---------- END PROCESSING ----------#