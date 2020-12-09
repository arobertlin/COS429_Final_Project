from __future__ import print_function
import cv2 as cv
import argparse
import os 

# directory = 'KTH_Data/running/frames/person01_running_01/seg3'
# # firstimage = cv.imread('KTH_Data/running/frames/person01_running_d1/seg3/frame214.jpg', cv.IMREAD_GRAYSCALE)
# firstimage = None
# secondimage = cv.imread('KTH_Data/running/frames/person01_running_d1/seg3/frame215.jpg', cv.IMREAD_GRAYSCALE)

# diff = cv.absdiff(firstimage, secondimage)

# threshold_value = 40
# set_to_value = 255
# ret, result = cv.threshold(diff, threshold_value, set_to_value, cv.THRESH_BINARY)

# print(type(result))


# [create]
# [capture]
src_directory = 'KTH_Data'
dest_directory = "KTH_Data_3D_motion_cuboid_w_squatting"
for parent_parent_directory in os.listdir(src_directory):
    parent_parent_dest_path = os.path.join(dest_directory, parent_parent_directory) 
    parent_parent_src_path = os.path.join(src_directory, parent_parent_directory)
    if '.ipynb_checkpoints' in parent_parent_dest_path:
        continue
    os.mkdir(parent_parent_dest_path)
    for parentdirectory in os.listdir(parent_parent_src_path):
        if parentdirectory != 'frames2':
            continue
        parent_dest_path = os.path.join(parent_parent_dest_path, parentdirectory) 
        parent_src_path = os.path.join(parent_parent_src_path, parentdirectory)
        if '.ipynb_checkpoints' in parent_dest_path:
            continue
        os.mkdir(parent_dest_path)
        for subdirectory in os.listdir(parent_src_path):
            destpath = os.path.join(parent_dest_path, subdirectory) 
            srcpath = os.path.join(parent_src_path, subdirectory)
            if '.ipynb_checkpoints' in destpath:
                continue
            os.mkdir(destpath)
            for subsubdirectory in os.listdir(srcpath): 
                # Path 
                sub_destpath = os.path.join(destpath, subsubdirectory) 
                sub_srcpath = os.path.join(srcpath, subsubdirectory)
                if '.ipynb_checkpoints' in sub_destpath:
                    continue
                os.mkdir(sub_destpath)     
                firstimage = None
                first = True
                print(subsubdirectory)
                for filename in sorted(os.listdir(sub_srcpath)):
                    print('Working')
                    secondimage = cv.imread(os.path.join(sub_srcpath,filename), cv.IMREAD_GRAYSCALE)
                    
                    if first:
                        first = False
                        firstimage = secondimage
                        continue
                        
                    print(filename)
                    try:
                        diff = cv.absdiff(firstimage, secondimage)

                        threshold_value = 40
                        set_to_value = 255
                        ret, result = cv.threshold(diff, threshold_value, set_to_value, cv.THRESH_BINARY)
                    except:
                        pass
            
                    output = os.path.join(sub_destpath, filename)
                    if result is not None:
                        cv.imwrite('{}'.format(output), result)
                    else:
                        print(output)
                    firstimage = secondimage
                    
