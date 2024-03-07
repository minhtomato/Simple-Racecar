import pandas as pd
import numpy as np
import tensorflow as tf
import os

#eigentlich nicht gut, habe aber keinen besseren Weg gefunden
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#learnend constants from normalizer
LEARNED_MEANS = np.array([ 2.9114618e+01,  9.0919641e+02,  2.2618099e+01,  2.0540454e+03,
         2.2350567e+01,  3.1214478e+03,  3.0489363e+01,  4.1120122e+03,
         3.6451164e+01,  4.6826846e+03,  1.5953860e+01,  4.9184155e+03,
        -1.8753021e+01,  4.8326768e+03, -3.4846458e+01,  4.4459541e+03,
        -1.4296617e+01,  3.8252571e+03, -4.7434478e+00,  3.1695378e+03,
         9.8633614e+01])

LEARNED_VARIANCES = np.array([1.5271010e+06, 1.6488716e+06, 1.0609779e+06, 1.6856356e+06,
        1.1636901e+06, 2.3239340e+06, 1.9838620e+06, 3.3765532e+06,
        3.5502898e+06, 4.3483820e+06, 5.3716085e+06, 5.7465245e+06,
        6.7475600e+06, 7.9761805e+06, 7.1915225e+06, 1.0374997e+07,
        7.0851285e+06, 1.1882636e+07, 7.0572995e+06, 1.2012724e+07,
        6.9620044e+02])

#Returns the unit vector of the vector.
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

#Returns the angle in radians between vectors 'v1' and 'v2'
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#rotates a vector
def affineTransform(vector, angle, position):
    rotation_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)],
                                [np.sin(angle),      np.cos(angle)]])

    return np.matmul(np.linalg.pinv(rotation_matrix), vector - position)

#changes points to match the inteface of the movement component
def pointsInterface(points, direction, position):
    vectors = []

    #x and y are mirrored in unreal engine
    if direction['y'] < 0:
        right_rotation = False
    else:
        right_rotation = True

    theta = angle_between(np.array([0,1]), np.array([direction['y'], direction['x']]))
    
    if right_rotation:
       theta *= -1

    for point in points:
        x = point['x']
        y = point['y']

        mirrored_point = np.array([y,x])
        transformed_point = affineTransform(mirrored_point, theta, np.array([position['y'],position['x']]))
        vectors.append(transformed_point[0])
        vectors.append(transformed_point[1])

    return vectors

def get_model(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dense(9)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def fitDataToModel(points, velocity:float, direction, position):
    #only take first 20 points
    points = points[:10]
    points = pointsInterface(points, direction, position)

    #fill up to 5 points if there are less
    while len(points) < 20:
        points.append(0)

    
    #concat lists to fit input of NN
    points.append(velocity)
    return np.array([points])

#geht wahrscheinlich besser, aber es war schon spät
def classToThrottleSteering(label):
    if label == 0:
        return -1, -1
    if label == 1:
        return -1, 0
    if label == 2:
        return -1, 1
    if label == 3:
        return 0, -1
    if label == 4:
        return 0, 0
    if label == 5:
        return 0, 1
    if label == 6:
        return 1, -1
    if label == 7:
        return 1, 0
    if label == 8:
        return 1, 1
    
def getThrottleSteering(model, features):
    label = np.argmax(model(features))
    throttle, steering = classToThrottleSteering(label)
    print('features:', features)
    print('steering:', steering, 'throttle:', throttle)
    return throttle, steering
    

def initModel():
    normalizer = tf.keras.layers.Normalization(mean=LEARNED_MEANS, variance=LEARNED_VARIANCES)
    model = get_model(normalizer)
    model.load_weights('./NNdata/weights')
    return model

