import numpy as np

with open('astroClock\data\polar_positions.npy', 'rb') as file:
    position_data = np.load(file)

def rotate_objects(positions: np.ndarray, current_object_index: int) -> np.ndarray:
    """
    Rotates objects so that a given current_object is at theta zero. 
    Args:
    Positions. An np.ndarray of positions.
    current_object_index. The index value of the object chosen as current_object,
    for the given rotation.
    """
    current_object_theta = positions[current_object_index, 2]
    rotated_positions = positions[:, 2] - current_object_theta
    positions[:, 2] = np.where(rotated_positions >= 0, rotated_positions, rotated_positions + 2*np.pi) 
    
    return positions

def get_starting_object(positions: np.ndarray) -> int:
    """
    Args:
    Positions. An np.ndarray with the positions where we want difference between
    a positions theta coordinate and the next position's theta to be calculated.
    
    Difference is calculated by by creating a new array, where the order of the 
    positions is shifted by one using np.roll. The original array's theta column is
    then subtracted from the new array's theta column, giving the difference. 
    """
    positions_theta = rotate_objects(positions=positions, current_object_index=9)[:, 2]
    rolled_positions_theta = np.roll(positions_theta, 1)
    rolled_positions_theta[0] = np.pi*2
    differences = rolled_positions_theta - positions_theta

    return np.argmax(differences)

def order_objects(positions: np.ndarray) -> np.ndarray:
    """
    Orders objects by theta and determines the object to be moved first in
    the current round.
    Arguments: Positions. An np.ndarray of shape (10, 2) holding the positions of
    each object in the current round.
    Returns: An np.ndarray of object numbers in order of movement in the current round.

    First an index is added to keep track of which object has which position.
    Then the positions are sorted by theta - column 2 in the indexed positions.
    Finally we measure the distance between the thetas of each object and the object 
    closest in front of it. The object with the largest distance measured in theta
    to the object on front of it is the starting object.

    Measurements are done with positions rotated, so the object being measured has
    theta zero. 
    """
    
    index = np.arange(0, 10).reshape(-1, 1).reshape(-1, 1)
    indexed_positions = np.hstack((index, positions))
    sorted_positions = indexed_positions[indexed_positions[:, 2].argsort()[::-1]]

    starting_object = get_starting_object(sorted_positions)
    
    order_of_excecution = np.roll(sorted_positions, -starting_object, axis=0)[:, 0]

    return order_of_excecution
