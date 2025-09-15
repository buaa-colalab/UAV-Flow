import numpy as np

def calculate_new_pose(location, rotation, backward_distance, up_distance):
    """
    Calculate a new pose based on an object's location and rotation.
    
    :param location: [x, y, z] position of the object
    :param rotation: [pitch, yaw, roll] rotation of the object in degrees
    :param backward_distance: Distance to move backward
    :param up_distance: Distance to move upward
    :return: New location [x, y, z] and rotation [pitch, yaw, roll]
    """
    # Convert rotation angles to radians
    pitch, yaw, roll = np.radians(rotation)
    
    # Compute forward vector (based on yaw and pitch)
    forward_vector = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch)
    ])
    
    # Compute up vector (default is Z axis in Unreal)
    up_vector = np.array([0, 0, 1])
    
    # Calculate the new location
    new_location = np.array(location) - backward_distance * forward_vector + up_distance * up_vector
    
    # The new rotation is the same as the original
    new_rotation = rotation
    
    return new_location.tolist(), new_rotation

# Example usage
original_location = [100, 200, 300]
original_rotation = [0, 90, 0]  # Pitch, Yaw, Roll in degrees
backward_distance = 50
up_distance = 30

new_location, new_rotation = calculate_new_pose(original_location, original_rotation, backward_distance, up_distance)
print("New Location:", new_location)
print("New Rotation:", new_rotation)