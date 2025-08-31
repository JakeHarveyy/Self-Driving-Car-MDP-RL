states = {'Clear Road', 'Veichle Ahead', 'Pedestrian Crossing', 'In School Zone', 'Obstacle Ahead', 'Traffic Light Red', 'Traffic Light Green', 'Destination Reached', 'Acident'}
actions = {'Maintain Speed', 'Accelerate', 'Decelerate', 'Stop', 'Change Lane', 'Steer Around'}


#R(s,a)
# rows represent current state, columns represent actions
# Columns :'Maintain Speed', 'Accelerate', 'Decelerate', 'Stop', 'Change Lane', 'Steer Around']
reward_matrix = [              # Rows:
    [10, 5, -5, -10, -5, -10], # Clear Road
    [-10, -50, 5, 1, 10, -50], # Veichle Ahead
    [-100, -100, 5, 10, -50, -100], # Pedestrian Crossing
    [-10, -50, 10, 1, -5, -10], # In School Zone
    [-100, -100, 5, 5, 10, 10], # Obstacle Ahead
    [-50, -50, 5, 10, -10, -10], # Traffic Light Red
    [10, 5, -5, -10, 1, -10], # Traffic Light Green
    [0, 0, 0, 0, 0, 0], # Destination Reached (terminal)
    [0, 0, 0, 0, 0, 0]  # Accident (terminal)
]

#R(s'|s a)
# Rows represent current state, columns represent actions, and the inner list represents the next states probabilities
transition_matrix = [
    # S1: Clear Road
    [
        [0.70, 0.15, 0.02, 0.0, 0.03, 0.05, 0.0, 0.05, 0.0],  # Maintain Speed
        [0.60, 0.25, 0.02, 0.0, 0.03, 0.05, 0.0, 0.05, 0.0],  # Accelerate
        [0.80, 0.10, 0.01, 0.0, 0.02, 0.05, 0.0, 0.02, 0.0],  # Decelerate
        [0.85, 0.05, 0.01, 0.0, 0.02, 0.05, 0.0, 0.02, 0.0],  # Stop
        [0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10],   # Change Lane (unnecessary, high risk)
        [0.80, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15]    # Steer Around (unnecessary, high risk)
    ],
    # S2: Vehicle Ahead
    [
        [0.0, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20],    # Maintain Speed (risky)
        [0.0, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40],    # Accelerate (very risky)
        [0.1, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],   # Decelerate (safe)
        [0.1, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],   # Stop (safe)
        [0.8, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05],   # Change Lane (ideal action)
        [0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60]     # Steer Around (very risky)
    ],
    # S3: Pedestrian Crossing
    [
        [0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50],    # Maintain Speed
        [0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90],    # Accelerate
        [0.4, 0.0, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Decelerate
        [0.6, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Stop (ideal action)
        [0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60],    # Change Lane
        [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80]     # Steer Around
    ],
    # S4: In School Zone
    [
        [0.0, 0.0, 0.20, 0.70, 0.0, 0.0, 0.0, 0.0, 0.10],    # Maintain Speed (risky)
        [0.0, 0.0, 0.30, 0.30, 0.0, 0.0, 0.0, 0.0, 0.40],    # Accelerate (very risky)
        [0.3, 0.0, 0.10, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0],    # Decelerate (ideal action)
        [0.1, 0.0, 0.05, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0],    # Stop
        [0.0, 0.0, 0.20, 0.60, 0.0, 0.0, 0.0, 0.0, 0.20],    # Change Lane
        [0.0, 0.0, 0.20, 0.60, 0.0, 0.0, 0.0, 0.0, 0.20]     # Steer Around
    ],
    # S5: Obstacle Ahead
    [
        [0.0, 0.0, 0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.70],    # Maintain Speed
        [0.0, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.90],    # Accelerate
        [0.1, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.05, 0.0],   # Decelerate
        [0.1, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.05, 0.0],   # Stop
        [0.8, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.05, 0.05],   # Change Lane (ideal action)
        [0.8, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.05, 0.05]    # Steer Around (ideal action)
    ],
    # S6: Traffic Light Red
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.50],    # Maintain Speed
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.80],    # Accelerate
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.15, 0.0, 0.0],   # Decelerate
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.70, 0.30, 0.0, 0.0],    # Stop (ideal action)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.60],    # Change Lane
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.60]     # Steer Around
    ],
    # S7: Traffic Light Green
    [
        [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],     # Maintain Speed (ideal action)
        [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],     # Accelerate
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.1],     # Decelerate (might get rear-ended)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.0, 0.1],     # Stop (risky)
        [0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],     # Change Lane
        [0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0]      # Steer Around
    ],
    # S8: Destination Reached (Terminal)
    [
        [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]
    ],
    # S9: Accident (Terminal)
    [
        [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]
    ]
]
