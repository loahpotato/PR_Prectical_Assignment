import numpy as np
from numpy import reshape
from scipy.spatial import distance
import tools
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

THRESHOLD = 200

# calculate concentration_rate feature
def cal_concentration(image):

    black_point_list = []
    sum_x = 0
    sum_y = 0
    dis_list = []

    for num_x, rows in enumerate(image):
        for num_y, column in enumerate(rows):
            if column > THRESHOLD:
                black_point_list.append([num_x, num_y])
    
    for item in black_point_list:
        sum_x += item[0]
        sum_y += item[1]

    val_x = sum_x / len(black_point_list)
    val_y = sum_y / len(black_point_list)

    for item in black_point_list:
        dis_list.append( distance.euclidean( [val_x, val_y], item ) )
    avg = sum(dis_list) / len(dis_list)

    return avg
