import numpy as np
from typing import Optional, Literal, Union


def nearest_objects(
        current_object_pos: np.ndarray,
        object_1_pos: np.ndarray,
        object_2_pos: np.ndarray,
        object_3_pos: np.ndarray,
        object_4_pos: np.ndarray,
        object_5_pos: np.ndarray,
        object_6_pos: np.ndarray,
        object_7_pos: np.ndarray,
        object_8_pos: np.ndarray,
        object_9_pos: np.ndarray,
) -> np.ndarray:
    """
    A function to sort the bodies by how close they are to the current body. is going to be used to create the training dataset, 
    and to feed data to the model in production. Maybe just an argsort on theta, or maybe that is not enough? We could perhaps get a situation 
    where objects are close on the theta parameter but distant enough on r?
    """
    raise NotImplementedError

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
    raise NotImplementedError 