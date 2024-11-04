import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_output(model, sample, ball_position, cmap='coolwarm', title='Default'):
    """
    Plots the SoccerMapDCN output as a heatmap with team positions on a soccer field.

    Parameters:
        model: The trained model for generating the output.
        sample: A sample tensor containing input data (should have shape (13, 104, 68)).
        ball_position: Coordinates of the ball on the field (ballx, bally).
        cmap: Colormap for the heatmap. Defaults to 'coolwarm'.
    """
    model.eval()

    with torch.no_grad():
        output = model(sample.unsqueeze(0))
        output_squeezed = np.squeeze(output).detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 68)

    heatmap = ax.imshow(output_squeezed.T, origin='lower', cmap=cmap, alpha=0.6, extent=(0, 104, 0, 68))
    
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Activation Intensity')

    attacking_team = sample[0].detach().numpy()  # Channel 1 (attacking team positions)
    defending_team = sample[3].detach().numpy()  # Channel 4 (defending team positions)
    
    attacking_x, attacking_y = np.where(attacking_team != 0)
    defending_x, defending_y = np.where(defending_team != 0)
    
    ballx, bally = ball_position
    ax.plot(ballx, bally, color='black', label='Ball', marker='o')

    ax.scatter(attacking_x, attacking_y, color='red', label='Attacking Team', marker='X')

    ax.scatter(defending_x, defending_y, color='blue', label='Defending Team', marker='o')

    ax.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.show()