import keras.backend as K
from tensorflow import cast, less_equal
import numpy as np

def _calc_cartesian_pos(pos):
    cart_pos = K.stack([pos[:, 0]*K.sin(pos[:, 1]), pos[:, 0]*K.cos(pos[:, 1])], axis=-1) 
    return cart_pos

def euclidian_dist(pos1, pos2):
    pos_1_cart = _calc_cartesian_pos(pos1)
    pos_2_cart = _calc_cartesian_pos(pos2)

    dist = K.sqrt(K.sum(K.square(pos_1_cart - pos_2_cart), axis=-1))
    return dist

def collisionCheck(pos1, pos2):
    return (euclidian_dist(pos1, pos2) < 120) 

def is_collision(
        object_positions, 
        ):
    """
    A function that takes the current position, its predicted end position and the positions of the nearest four objects as input. 
    From this the function should calculate if the predicted position would result in a collision - probably calculated both at the endpoint and midway.
    The function should return 1 if there is a colission and 0 if not. This is easier to use in the loss function than booleans.

    The object numbers are also included as optional parameters if we wish to do a calculation that takes the individual object's radius into acount.  
    
    Returns: A vector containing the sum of colissions.
    """
    positions_length = K.eval(object_positions.shape[0])
    
    current_object_end_position = object_positions[:, 10:12]
    dist_1 = euclidian_dist(current_object_end_position, object_positions[:, 2:4])
    dist_2 = euclidian_dist(current_object_end_position, object_positions[:, 4:6])
    dist_3 = euclidian_dist(current_object_end_position, object_positions[:, 6:8])
    dist_4 = euclidian_dist(current_object_end_position, object_positions[:, 8:10])

    
    check_1 = cast(less_equal(x=dist_1, y=120), dtype=np.float32)
    check_2 = cast(less_equal(x=dist_2, y=120), dtype=np.float32)
    check_3 = cast(less_equal(x=dist_3, y=120), dtype=np.float32)
    check_4 = cast(less_equal(x=dist_4, y=120), dtype=np.float32)

    collision_tensor = K.stack((check_1, check_2, check_3, check_4), axis=1)
    return K.reshape(K.sum(collision_tensor, axis=-1), (positions_length, 1))