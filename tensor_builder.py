import numpy as np
import torch
import re

'''''
Processes sample event and tracking data provided by 
Metrica (https://github.com/metrica-sports/sample-data)
into 13 channel game state representation proposed in SoccerMap 
(https://arxiv.org/pdf/2010.10202)

'''''

def find_closest_tracking_snapshot(event_time, tracking_data):
    # Calculate time differences
    tracking_data['time_diff'] = np.abs(tracking_data['Time [s]'] - event_time)
    
    # Get minimum time difference (closest snapshot)
    closest_snapshot = tracking_data.loc[tracking_data['time_diff'].idxmin()]
    
    return closest_snapshot

def build_sparse_matrix(team, carrier_id=None):
    
    '''''
    Builds inital channels of game state representation:
        - Six sparse matrices with the location, and the two components of the velocity
            vector for the players in both the attacking team and the defending team,
            respectively.
    
    Inputs:
        - Team (attacking or defending)
        - Carrier ID (if processing attacking team)
    
    Outputs:
        - Sparse (104x86) matrix of team positions
        - 2 Sparse (104x86) matrices of respective velocities
        - (conditonal) Individual Matrices for ball carrier

    '''''

    team_pos = []
    team_vel = []

    field_size=(104, 68)

    # Placeholders for ball carrier
    carrier_pos = None
    carrier_vel = None

    # Sort positions
    position_keys = sorted([key for key in team.index if "_x" in key or "_y" in key])

    # Tracking 11 active players
    player_count = 0

    for i in range(0, len(position_keys), 2):
        x_key = position_keys[i]
        y_key = position_keys[i + 1]
        
        # Ensure non-zero
        if not (np.isnan(team[x_key]) or np.isnan(team[y_key])):
            pos = (team[x_key], team[y_key])
            team_pos.append(pos)

            # Find corresponding velocity if exists.
            vx_key = x_key.replace('_x', '_vx')
            vy_key = y_key.replace('_y', '_vy')

            if vx_key in team and vy_key in team:
                vel = (team[vx_key] * field_size[0], team[vy_key] * field_size[1])
                team_vel.append(vel) 

                # Identify ball carrier values
                if carrier_id and f"_{carrier_id}_" in x_key:
                    carrier_pos = pos
                    carrier_vel = vel
            
            player_count += 1

        # Max of 11 players on field
        if player_count == 11:
            break

    # Initialize sparse matrices
    sparse_pos = np.zeros(field_size)
    sparse_velx = np.zeros(field_size)
    sparse_vely = np.zeros(field_size)

    for pos, vel in zip(team_pos, team_vel):
        x, y = int(pos[0]), int(pos[1]) 
        x = min(max(x, 0), 104 - 1)
        y = min(max(y, 0), 68 - 1)
        sparse_pos[x, y] = 1           # Set position indicator
        sparse_velx[x, y] = vel[0]     # Set x velocity
        sparse_vely[x, y] = vel[1]     # Set y velocity

    if carrier_id:
        return sparse_pos, sparse_velx, sparse_vely, carrier_pos, carrier_vel, team_pos
    else:
        return sparse_pos, sparse_velx, sparse_vely

def calculate_distance_matrices(ball_x, ball_y):

    field_size=(104, 68)
    # Todo: make goal position dynamic per team
    goal_pos=(0, 34)
    distance_to_ball = np.zeros(field_size)
    distance_to_goal = np.zeros(field_size)

    ball_x *= field_size[0]
    ball_y *= field_size[1]
    ball_x = min(max(ball_x, 0), 104 - 1)
    ball_y = min(max(ball_y, 0), 68 - 1)
    goal_x, goal_y = goal_pos
    
    # Calculate distances for each cell on the field
    for i in range(field_size[0]):
        for j in range(field_size[1]):
            distance_to_ball[i, j] = np.sqrt((i - ball_x)**2 + (j - ball_y)**2)
            distance_to_goal[i, j] = np.sqrt((i - goal_x)**2 + (j - goal_y)**2)
    
    return distance_to_ball, distance_to_goal

def calculate_angle_matrices(ball_x, ball_y):
    # Matrices for sin, cosine of the angle to the ball, and angle in radians to the goal

    # Todo: dynamic goal pos
    goal_pos=(0,34)

    field_size=(104, 68)

    angle_to_goal = np.zeros(field_size)
    sin_to_ball = np.zeros(field_size)
    cos_to_ball = np.zeros(field_size)

    ball_x *= field_size[0]
    ball_y *= field_size[1]
    ball_x = min(max(ball_x, 0), 104 - 1)
    ball_y = min(max(ball_y, 0), 68 - 1)
    goal_x, goal_y = goal_pos
    
    
    for i in range(field_size[0]):
        for j in range(field_size[1]):
            angle_to_goal[i, j] = np.arctan2(goal_y - j, goal_x - i)
            angle_to_ball = np.arctan2(ball_y - j, ball_x - i)
            
            # Calculate sin and cosine of the angle to the ball
            sin_to_ball[i, j] = np.sin(angle_to_ball)
            cos_to_ball[i, j] = np.cos(angle_to_ball)
    
    return angle_to_goal, sin_to_ball, cos_to_ball

def calculate_velocity_angle_sine_cosine(ball_carrier_pos, ball_carrier_vel, teammates_pos):
    # Matrices for sin and cosine of angles between velocity vector and teammate vectors
    field_size = (104, 68)
    sparse_sin = np.zeros(field_size)
    sparse_cos = np.zeros(field_size)
    
    x_carrier, y_carrier = ball_carrier_pos
    vx_carrier, vy_carrier = ball_carrier_vel
    
    vx_carrier = vx_carrier * field_size[0]
    vy_carrier = vy_carrier * field_size[1]

    # Magnitude of the carrier velocity 
    carrier_velocity_magnitude = np.sqrt(vx_carrier**2 + vy_carrier**2)
    
    # Calculate sine and cosine for teammate vector
    for (x_teammate, y_teammate) in teammates_pos:
        dx = x_teammate - x_carrier
        dy = y_teammate - y_carrier
        teammate_distance = np.sqrt(dx**2 + dy**2)
        
        if carrier_velocity_magnitude > 0 and teammate_distance > 0:
            # Dot product between carier vector and teammate vector
            dot_product = vx_carrier * dx + vy_carrier * dy
            # Angle cosine and sin using normalized vectors
            cosine_angle = dot_product / (carrier_velocity_magnitude * teammate_distance)
            angle = np.arccos(np.clip(cosine_angle, -1, 1))
            
            # Calculate sine and cosine of the angle
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)
            
            # Populate matrices at teammate's position
            tx, ty = int(x_teammate), int(y_teammate)
            tx = min(max(tx, 0), 104 - 1)
            ty = min(max(ty, 0), 68 - 1)
            sparse_sin[tx, ty] = sin_angle
            sparse_cos[tx, ty] = cos_angle

    return sparse_sin, sparse_cos

# Find ball carrier from event data
def helper(event):
    holder = event['From']
    numbers = re.findall(r'\d+', holder)
    return int(''.join(numbers)) if numbers else None


def build_representation(event, pos_team, opp_team):

    ballx = max(0, min(1, event['Start X']))
    bally = max(0, min(1, event['Start Y']))
    endx = max(0, min(1, event['End X'])) * 104
    endy = max(0, min(1, event['End Y'])) * 68
    endx = min(max(endx, 0), 104 - 1)
    endy = min(max(endy, 0), 68 - 1)

    carrier = helper(event)
    
    pos_team_pos, pos_team_velx, pos_team_vely, carrier_pos, carrier_vel, team_pos = build_sparse_matrix(pos_team, carrier)
    opp_team_pos, opp_team_velx, opp_team_vely = build_sparse_matrix(opp_team)
    dis_ball, dis_goal = calculate_distance_matrices(ballx, bally)
    angle_goal, sin_ball, cos_ball = calculate_angle_matrices(ballx, bally)
    sparse_sin, sparse_cos = calculate_velocity_angle_sine_cosine(carrier_pos, carrier_vel, team_pos)

    game_state_tensor = torch.stack([
        torch.tensor(pos_team_pos, dtype=torch.float32),
        torch.tensor(pos_team_velx, dtype=torch.float32),
        torch.tensor(pos_team_vely, dtype=torch.float32),
        torch.tensor(opp_team_pos, dtype=torch.float32),
        torch.tensor(opp_team_velx, dtype=torch.float32),
        torch.tensor(opp_team_vely, dtype=torch.float32),
        torch.tensor(sparse_sin, dtype=torch.float32),
        torch.tensor(sparse_cos, dtype=torch.float32),
        torch.tensor(dis_ball, dtype=torch.float32),
        torch.tensor(dis_goal, dtype=torch.float32),
        torch.tensor(angle_goal, dtype=torch.float32),
        torch.tensor(sin_ball, dtype=torch.float32),
        torch.tensor(cos_ball, dtype=torch.float32)

    ], dim=0)  # Stacks along the channel dimension

    # Add a batch dimension, for shape (1, 13, 104, 68)
    game_state_tensor = game_state_tensor.unsqueeze(0)

    # Target
    label = [endx, endy]

    # Outcome
    outcome = event['Success']

    return game_state_tensor, label, outcome