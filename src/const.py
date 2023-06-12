BOARD_RADIUS = 2830 / 2 #[mm]
HORIZON_WIDTH = 150 #[mm]
HORIZON_RADIUS = BOARD_RADIUS - HORIZON_WIDTH
MIN_PLANET_CLEARANCE = 20 #[mm]
MIN_POS_RADIUS = 10 #[mm]
MAX_STEPSIZE = 150 #[mm]

sun = {
    'Name': 'Sun',
    'Abv': 'Su',
    'MarkerRadius': 135/2,
}

jupiter = {
    'Name': 'Jupiter',
    'Abv': 'Ju',
    'MarkerRadius': 110/2,
}

neptune = {
    'Name': 'Neptune',
    'Abv': 'Ne',
    'MarkerRadius': 110/2,
}

saturn = {
    'Name': 'Saturn',
    'Abv': 'Sa',
    'MarkerRadius': 110/2,
}

uranus = {
    'Name': 'Uranus',
    'Abv': 'Ur',
    'MarkerRadius': 110/2,
}


moon = {
    'Name': 'Moon',
    'Abv': 'Mo',
    'MarkerRadius': 135/2,
}

mercury = {
    'Name': 'Mercury',
    'Abv': 'Me',
    'MarkerRadius': 80/2,
}

venus = {
    'Name': 'Venus',
    'Abv': 'Ve',
    'MarkerRadius': 80/2,
}

mars = {
    'Name': 'Mars',
    'Abv': 'Ma',
    'MarkerRadius': 80/2,
}

stellanova = {
    'Name': 'StellaNova',
    'Abv': 'St',
    'MarkerRadius': 80/2,
}

galacticcenter = {
    'Name': 'GalacticCenter',
    'Abv': 'C',
    'MarkerRadius': 80/2,
}

planets = (sun, jupiter, neptune, saturn, uranus, moon, mercury, venus, mars, galacticcenter) # stellanova,

NUM_PLANETS = len(planets)
