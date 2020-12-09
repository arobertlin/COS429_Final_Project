from __future__ import print_function
import cv2 as cv
import argparse
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# backSub = cv.createBackgroundSubtractorMOG2()
# ## [create]
# ## [capture]
# src_directory = 'KTH_Data'
# dest_directory = "KTH_Data_Preprocessed_w_squatting"
# for parent_parent_directory in os.listdir(src_directory):
#     parent_parent_dest_path = os.path.join(dest_directory, parent_parent_directory) 
#     parent_parent_src_path = os.path.join(src_directory, parent_parent_directory)
#     if '.ipynb_checkpoints' in parent_parent_dest_path:
#         continue
#     os.mkdir(parent_parent_dest_path)
#     for parentdirectory in os.listdir(parent_parent_src_path):
#         if parentdirectory != 'frames2':
#             continue
#         parent_dest_path = os.path.join(parent_parent_dest_path, parentdirectory) 
#         parent_src_path = os.path.join(parent_parent_src_path, parentdirectory)
#         if '.ipynb_checkpoints' in parent_dest_path:
#             continue
#         os.mkdir(parent_dest_path)
#         for subdirectory in os.listdir(parent_src_path):
#             destpath = os.path.join(parent_dest_path, subdirectory) 
#             srcpath = os.path.join(parent_src_path, subdirectory)
#             if '.ipynb_checkpoints' in destpath:
#                 continue
#             os.mkdir(destpath)
#             for subsubdirectory in os.listdir(srcpath):    
#                 # Path 
#                 sub_destpath = os.path.join(destpath, subsubdirectory) 
#                 sub_srcpath = os.path.join(srcpath, subsubdirectory)
#                 if '.ipynb_checkpoints' in sub_destpath:
#                     continue
#                 os.mkdir(sub_destpath)     
#                 for filename in os.listdir(sub_srcpath):
#                     print(filename)
#                     image = cv.imread(os.path.join(sub_srcpath,filename))
#                     fgMask = backSub.apply(image)
#                     output = os.path.join(sub_destpath, filename)
#                     if fgMask is not None:
#                         cv.imwrite('{}'.format(output), fgMask)
#                     else:
#                         print(output)


model_name = "InceptionV3_squatting"
classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'squatting', 'walking']


# 7 X 7 right now
confusion_matrix = np.array([[101, 2, 2, 0, 0, 0, 1], 
                           [6, 72, 4, 4, 2, 18, 3],
                           [2, 66, 138, 0, 0, 0, 0],
                           [12, 0, 0, 94, 21, 0, 14],
                           [5, 0, 0, 37, 120, 0, 1],
                           [0, 3, 0, 0, 0, 126, 0],
                           [18, 1, 0, 9, 1, 0, 125]])


con_mat_norm = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(con_mat_norm.shape)

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)

figure = plt.figure(figsize = (6, 6))
sns.heatmap(con_mat_df, annot=True, cmap = plt.cm.Blues)
plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(model_name + "_confusion")