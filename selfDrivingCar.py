states = {'Clear Road', 'Veichle Ahead', 'Pedestrian Crossing', 'School Zone', 'Obstacle on Road', 'Traffic Light Red', 'Traffic Light Green', 'Lane Changing Opportunity', 'Rush Hour Traffic', 'Night Driving'}
actions = {'Maintain Speed', 'Accelerate', 'Decelerate', 'Stop', 'Changing Lane Left', 'Changing Lane Right', 'Steer Around Obstacles'}

# Rows represent current state, columns represent actions, and the inner list represents the next states probabilities
transition_matrix = [
    [0, 0, 0, 0, 0, 0, 0], # Clear Road
    [0, 0, 0, 0, 0, 0, 0], # Veichle Ahead
    [0, 0, 0, 0, 0, 0, 0], # Pedestrian Crossing
    [0, 0, 0, 0, 0, 0, 0], # School Zone
    [0, 0, 0, 0, 0, 0, 0], # Obstacle on Road
    [0, 0, 0, 0, 0, 0, 0], # Traffic Light Red
    [0, 0, 0, 0, 0, 0, 0], # Traffic Light Green
    [0, 0, 0, 0, 0, 0, 0], # Lane Changing Opportunitty
    [0, 0, 0, 0, 0, 0, 0], # Rush Hour Traffic
    [0, 0, 0, 0, 0, 0, 0], # Night Driving 
    [0, 0, 0, 0, 0, 0, 0], # Destination Reached
    [0, 0, 0, 0, 0, 0, 0] # Collision
]

# rows represent current state, columns represent actions
reward_matrix = [
    [0, 0, 0, 0, 0, 0, 0], # Clear Road
    [0, 0, 0, 0, 0, 0, 0], # Veichle Ahead
    [0, 0, 0, 0, 0, 0, 0], # Pedestrian Crossing
    [0, 0, 0, 0, 0, 0, 0], # School Zone
    [0, 0, 0, 0, 0, 0, 0], # Obstacle on Road
    [0, 0, 0, 0, 0, 0, 0], # Traffic Light Red
    [0, 0, 0, 0, 0, 0, 0], # Traffic Light Green
    [0, 0, 0, 0, 0, 0, 0], # Lane Changing Opportunitty
    [0, 0, 0, 0, 0, 0, 0], # Rush Hour Traffic
    [0, 0, 0, 0, 0, 0, 0], # Night Driving 
    [0, 0, 0, 0, 0, 0, 0], # Destination Reached
    [0, 0, 0, 0, 0, 0, 0] # Collision
]