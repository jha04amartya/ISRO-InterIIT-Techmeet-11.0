import numpy as np
import pandas as pd
import csv
import PIL

from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature

import numba
from numba import njit,prange
from math import radians, cos, sin, asin, sqrt

# we are scraping through each image and create blocks of 1 X 1 degree by degree solution in order to create a map
# write the range of lats and longs where you want the map
long_per_pixel = 7.24501125281312e-05
lat_per_pixel = 0.0001677774918709176
minimum_latitude = 50
maximum_latitude = 179 # (180-1)
minimum_longitude = 200
maximun_longitude = 359 # (360 - 1)

lat_long_set = [[i,j] for i in np.arange(minimum_latitude, maximum_latitude, 0.5) for j in np.arange(minimum_longitude, maximun_longitude, 0.5)]

yl = maximum_latitude - minimum_latitude
xl = maximun_longitude - minimum_longitude
yl = yl / lat_per_pixel
xl = xl/ long_per_pixel
atlas = [[0]*yl for _ in range(xl)]

images = []
with open("coordinates_tmc2.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        image = {
            "name": row[0],
            "lat1": float(row[1]),
            "lon1": float(row[2]),
            "lat2": float(row[3]),
            "lon2": float(row[4]),
            "lat3": float(row[5]),
            "lon3": float(row[6]),
            "lat4": float(row[7]),
            "lon4": float(row[8])
        }
images.append(image)

map_image = np.array([])
'''
for img in images:
    overlapx = 
'''
for x,y in lat_long_set:
    # Now we are going to find the image which has the 1 X 1 degree box as defined in the problem 
    # check the range of latitudes and longitude and find the problem 
    min_lat = x -90
    max_lat = x+0.5 
    min_long = y -90
    max_long = y+0.5 
    found_patch = False
    images_found = []
    for image in images:
        # check if the image contains the area of i and i+1 latitudes and j and j+1 longitudes 
        contains = False
        point1 = Feature(geometry=Point((min_lat, min_long)))
        point2 = Feature(geometry=Point((min_lat, max_long)))
        point3 = Feature(geometry=Point((max_lat, min_long)))
        point4 = Feature(geometry=Point((max_lat, max_long)))
        polygon = Polygon(
        [
            [
                (image["lat1"], image["lon1"]),
                (image["lat2"], image["lon2"]),
                (image["lat3"], image["lon3"]),
                (image["lat4"], image["lon4"]),
            ]
        ]   
        )
        bool1 = boolean_point_in_polygon(point1, polygon)
        bool2 = boolean_point_in_polygon(point2, polygon)
        bool3 = boolean_point_in_polygon(point3, polygon)
        bool4 = boolean_point_in_polygon(point4, polygon)
        if bool1 and bool2 and bool3 and bool4:
            contains = True
            found_patch = True 
        # Now I am going to use the csv file to find the exact pixel and the box in which the latitudes and longitudes are contained
    
        if contains:
            # we are assuming that tmc images are downloaded and present in a tmc folder in a directory
            dir_tmc_images = "./tmc/"
            usable_image_name = image["name"].replace("g_grd", "d_img")
            # ch2_tmc_ncn_20191015T1021251544_g_grd_d18
            image = "pass"
            corresponding_csv = pd.read_csv(dir_tmc_images + usable_image_name + "/" + "geometry/"+ "caliberated/" + usable_image_name.split("_")[3].split("T") + image["name"] + ".csv")
            # find the nearest pixels and pixels within which the coordinates lie and use that to scrape the image 
            ul = [min_lat,min_long]
            ur = [min_lat,max_long]
            ll = [max_lat,min_long]
            lr = [max_lat,max_long]
            corners = np.array([ul,ur,lr,ll])
            have_dist = np.zeros((len(detailed_coords),4))
            lonlat = np.array(detailed_coords)[:,:2]
            for i in prange(len(have_dist)):
                have_dist[i] = haversine(corners[:,1],corners[:,0],lonlat[i][0],lonlat[i][1])
            #ul,ur,lr,ll
            ixs = np.argmin(have_dist,axis = 0)
            
            #pixel coordinates of the corners in the image
            #pixel and scan = (scan,pixel)
            pixel_coords = np.zeros((4,2),dtype = np.int32)
            for i in range(4):
                pixel_coords[i] = np.array(detailed_coords)[ixs[i],2:4]

            #The bounding box for the corner pixel coordinates
            maxs,mins = pixel_coords.max(axis = 0),pixel_coords.min(axis = 0)
            bounding_box = im_tmc[int(mins[1]):int(maxs[1]),int(mins[0]):int(maxs[0])]
            box_pix_coords = pixel_coords.copy()
            box_pix_coords[:,0] -= mins[0]
            box_pix_coords[:,1] -= mins[1]
            box_pix_coords
            
            #image processing strip from bounding box
            bb = bounding_box
            bb_maxs = box_pix_coords.max(axis = 0)
            bb_mins = box_pix_coords.min(axis = 0)
            mask = np.zeros((bb.shape[0], bb.shape[1]))
            cv2.fillConvexPoly(mask, box_pix_coords, 1)
            mask = mask.astype(bool)
            out = np.zeros_like(bb)
            out[mask] = bb[mask]
            patch = bb[mask]
            # create a subimage with the outer limits of the points
            subimg = out[bb_mins[1]:bb_maxs[1],bb_mins[0]:bb_maxs[0]]
            a = max(abs(box_pix_coords[3][0]-box_pix_coords[2][0]),abs(box_pix_coords[1][0]-box_pix_coords[0][0]))
            subimg__ = np.zeros((subimg.shape[0],a))
            for i in range(len(subimg)):
                row = subimg[i]
                for j in range(len(subimg[0])):
                    if row[j] != 0 :
                        if len(row[j:])>a:
                            subimg__[i] = row[j:j+a]
                        else:
                            subimg__[i][:len(row[j:])] = row[j:]
            
            #subimg__ is the output that you want
            atlas = np.conat(atlas, subimg__)
            break
