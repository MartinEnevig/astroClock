import numpy as np
from typing import Optional, Literal, Union


def nearest_objects(
        current_object_pos: np.ndarray,
        object_0_pos: np.ndarray,
        object_1_pos: np.ndarray,
        object_2_pos: np.ndarray,
        object_3_pos: np.ndarray,
        object_4_pos: np.ndarray,
        object_5_pos: np.ndarray,
        object_6_pos: np.ndarray,
        object_7_pos: np.ndarray,
        object_8_pos: np.ndarray,
) -> np.ndarray:
    """
    A function to sort the bodies by how close they are to the current body. is going to be used to create the training dataset, 
    and to feed data to the model in production. Maybe just an argsort on theta, or maybe that is not enough? We could perhaps get a situation 
    where objects are close on the theta parameter but distant enough on r?
    """
    posArray = np.empty(9, dtype=np.float32)
    posArray[0] = object_0_pos[1] 
    posArray[1] = object_1_pos[1] 
    posArray[2] = object_2_pos[1] 
    posArray[3] = object_3_pos[1] 
    posArray[4] = object_4_pos[1] 
    posArray[5] = object_5_pos[1] 
    posArray[6] = object_6_pos[1] 
    posArray[7] = object_7_pos[1] 
    posArray[8] = object_8_pos[1] 

    for i in range(9):
        posArray[i] -= current_object_pos[1]
        if posArray[i] < -np.pi : posArray[i] += 2*np.pi
        elif posArray[i] > np.pi : posArray[i] -= 2*np.pi
    

    return np.argsort(np.abs(posArray))

def is_collision(
        current_object_start_position: np.ndarray,
        current_object_end_position: np.ndarray,
        object_1_pos: np.ndarray,
        object_2_pos: np.ndarray,
        object_3_pos: np.ndarray,
        object_4_pos: np.ndarray,
        current_object_no: Optional[int],
        object_1_no: Optional[int],
        object_2_no: Optional[int],
        object_3_no: Optional[int],
        object_4_no: Optional[int], 
        ) -> Union[Literal[0], Literal[1]]:
    """
    A function that takes the current position, its predicted end position and the positions of the nearest four objects as input. 
    From this the function should calculate if the predicted position would result in a collision - probably calculated both at the endpoint and midway.
    The function should return 1 if there is a colission and 0 if not. This is easier to use in the loss function than booleans.

    The object numbers are also included as optional parameters if we wish to do a calculation that takes the individual object's radius into acount.  
    """
    if collisionCheck(current_object_end_position, object_1_pos):
        return 1
    if collisionCheck(current_object_end_position, object_2_pos):
        return 1  
    if collisionCheck(current_object_end_position, object_3_pos):
        return 1 
    if collisionCheck(current_object_end_position, object_4_pos):
        return 1

    return 0

def collisionCheck(pos1: np.ndarray, pos2: np.ndarray) -> bool:
    # Simple one-size collision check
    return (calcDistSquared(pos1, pos2) < 14400) # 14400 = 120mm ^2


def calcCartesianPos(pos: np.ndarray)-> np.ndarray:
    posXY = np.empty(2, dtype=np.float32)
    posXY[0], posXY[1] = np.sin(pos[1])*pos[0] , np.cos(pos[1])*pos[0]
    return posXY

def calcDistSquared(pos1: np.ndarray, pos2: np.ndarray):
    posXY1 = calcCartesianPos(pos1)
    posXY2 = calcCartesianPos(pos2)
    distSquared = (posXY1[0]-posXY2[0])**2 + (posXY1[1]-posXY2[1])**2
    return distSquared