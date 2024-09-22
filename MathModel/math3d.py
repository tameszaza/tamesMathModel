import trimesh
import numpy
print(trimesh.boolean._engines.keys())
import numpy as np
import random




def create_intersection_sphere(point, radius=0.02):
    """
    Creates a small sphere for visualization at the intersection point.
    """
    return trimesh.primitives.Sphere(radius=radius, center=point)

def find_intersections(start_point, end_point, objects):
    """
    Finds the unique faces intersected by a ray between start_point and end_point across the given objects.
    
    Parameters:
    start_point (numpy array): The starting point of the ray.
    end_point (numpy array): The end point of the ray.
    objects (list of trimesh objects): List of objects to check for intersections.

    Returns:
    int: The number of unique faces intersected by the ray.
    list: The 3D positions of the intersection points.
    list: The scene objects with added visualizations of intersections.
    """
    # Ray direction and length
    ray_direction = end_point - start_point
    ray_length = np.linalg.norm(ray_direction)  # Compute the length of the ray
    ray_direction_normalized = ray_direction / ray_length  # Normalize the ray direction

    
    intersection_positions = []  # To store 3D positions of intersection points
    

    # Loop over all objects and find intersections
    for obj in objects:
        # Get the intersection locations and face indices for each object
        locations, _, _ = obj.ray.intersects_location([start_point], [ray_direction_normalized], max_ray_length=ray_length)

        
        for loc in locations:
            if np.linalg.norm(loc - start_point) <= ray_length:  # Ensure it's within the ray's finite length
                intersection_positions.append(loc)
                

    return intersection_positions






def add_ray_visualization(start_point, end_point, scene_objects):
    """
    Adds a visual representation of the ray as a thin cylinder between start_point and end_point.
    """
    line_length = np.linalg.norm(end_point - start_point)
    line_radius = 0.01  # Set a small radius to make the line thin
    line_cylinder = trimesh.primitives.Cylinder(radius=line_radius, height=line_length)

    # Compute the direction vector and rotation matrix to orient the cylinder
    ray_direction = end_point - start_point
    ray_direction_normalized = ray_direction / line_length  # Normalize direction
    axis = np.array([0, 0, 1])  # Default axis for cylinder (z-axis)
    rotation_matrix = trimesh.geometry.align_vectors(axis, ray_direction_normalized)

    # Apply rotation and translation to the line cylinder
    line_cylinder.apply_transform(rotation_matrix)
    midpoint = (start_point + end_point) / 2
    line_cylinder.apply_translation(midpoint)

    # Add the cylinder to the scene
    scene_objects.append(line_cylinder)


def add_wireframe_visualization(object, scene_objects, color=[0, 0, 0, 255]):
    """
    Adds a wireframe version of the mesh to the scene using the mesh's edges.
    Allows setting a custom color (default black).
    
    Parameters:
    object (trimesh.Trimesh): The object to create a wireframe from.
    scene_objects (list): The list of scene objects to which the wireframe will be added.
    color (list): RGBA color of the wireframe (default is black [0, 0, 0, 255]).
    """
    # Get the edges of the mesh
    edges = object.edges_unique
    # Convert the edges into lines for visualization
    path = trimesh.load_path(object.vertices[edges])
    
    # Apply the color to the path lines (not per vertex but for the entire path)
    path.colors = np.array([color] * len(path.entities))  # Apply the same color to all segments
    
    # Add the wireframe path to the scene objects
    scene_objects.append(path)


def random_camera_rotation():
    """
    Generates a random 3x3 rotation matrix for a camera.
    """
    # Random Euler angles for rotation
    angles = np.radians(np.random.uniform(0, 360, 3))  # Random rotation in each axis
    rotation_matrix = trimesh.transformations.euler_matrix(*angles)[:3, :3]  # Create a 3x3 rotation matrix
    return rotation_matrix

def random_camera_position(range_min=-10, range_max=10):
    """
    Generates a random position for a camera within the given range.
    """
    return np.random.uniform(range_min, range_max, 3)




def create_camera_frustum(camera_position, camera_rotation, fov_horizontal, fov_vertical, near_clip, far_clip):
    """
    Creates a solid 3D camera frustum object given camera parameters.
    
    Parameters:
    camera_position (numpy array): 3D position of the camera.
    camera_rotation (numpy array): 3x3 rotation matrix of the camera.
    fov_horizontal (float): Horizontal field of view (in degrees).
    fov_vertical (float): Vertical field of view (in degrees).
    near_clip (float): Near clipping plane distance.
    far_clip (float): Far clipping plane distance.

    Returns:
    trimesh.Trimesh: 3D frustum object with solid volume.
    tuple: Two far clip vertices with the largest distance in the x, z plane.
    """
    # Convert FoV from degrees to radians
    fov_horizontal_rad = np.deg2rad(fov_horizontal)
    fov_vertical_rad = np.deg2rad(fov_vertical)
    
    # Calculate half extents for near and far planes
    near_height = 2 * np.tan(fov_vertical_rad / 2) * near_clip
    near_width = 2 * np.tan(fov_horizontal_rad / 2) * near_clip
    
    far_height = 2 * np.tan(fov_vertical_rad / 2) * far_clip
    far_width = 2 * np.tan(fov_horizontal_rad / 2) * far_clip

    # Define the 8 corners of the frustum (4 on the near plane, 4 on the far plane)
    near_plane = np.array([
        [-near_width / 2, -near_height / 2, -near_clip],  # Bottom-left
        [ near_width / 2, -near_height / 2, -near_clip],  # Bottom-right
        [ near_width / 2,  near_height / 2, -near_clip],  # Top-right
        [-near_width / 2,  near_height / 2, -near_clip],  # Top-left
    ])

    far_plane = np.array([
        [-far_width / 2, -far_height / 2, -far_clip],  # Bottom-left
        [ far_width / 2, -far_height / 2, -far_clip],  # Bottom-right
        [ far_width / 2,  far_height / 2, -far_clip],  # Top-right
        [-far_width / 2,  far_height / 2, -far_clip],  # Top-left
    ])

    # Combine the points into a single array
    frustum_vertices = np.vstack((near_plane, far_plane))

    # Ensure camera rotation is a 3x3 matrix
    if camera_rotation.shape != (3, 3):
        raise ValueError("camera_rotation must be a 3x3 matrix")

    # Apply the camera's rotation and position to the frustum vertices
    rotated_vertices = (camera_rotation @ frustum_vertices.T).T
    translated_vertices = rotated_vertices + camera_position

    # Define the faces of the frustum using vertex indices
    frustum_faces = np.array([
        # Near plane
        [0, 1, 2],
        [0, 2, 3],
        
        # Far plane
        [4, 5, 6],
        [4, 6, 7],
        
        # Sides to close the frustum and make it a solid shape
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ])

    # Create a Trimesh object to represent the frustum
    frustum_mesh = trimesh.Trimesh(vertices=translated_vertices, faces=frustum_faces)
    
    # Ensure proper normals for the solid object
    frustum_mesh.fix_normals()

    

    return frustum_mesh


def load_stl_model(filepath):
    """
    Loads an STL model from the given file path.
    
    Parameters:
    filepath (str): Path to the STL file.
    
    Returns:
    trimesh.Trimesh: The loaded mesh object.
    """
    try:
        # Load the STL file
        mesh = trimesh.load(filepath)
        print(f"STL model loaded from {filepath}")
        return mesh
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return None
    
def validate_3d_object(mesh):
    """
    Validates and checks the integrity and properties of a 3D object (Trimesh).
    
    Parameters:
    mesh (trimesh.Trimesh or None): The 3D object to validate. If the mesh is None, it indicates an invalid or empty mesh.
    
    Returns:
    None
    """
    # If the mesh is None (e.g., due to an empty intersection), handle gracefully
    if mesh is None:
        print("The mesh is empty (None). No validation can be performed.")
        return
    
    # Check if the mesh is watertight
    if mesh.is_watertight:
        print("The mesh is watertight.")
    else:
        print("The mesh is NOT watertight.")
    
    # Check if the mesh encloses a valid volume
    if mesh.is_volume:
        print("The mesh encloses a valid volume.")
    else:
        print("The mesh does NOT enclose a valid volume.")
    
    # Check for degenerate faces (faces with zero area)
    degenerate_faces = mesh.area == 0
    if degenerate_faces.any():
        print(f"There are {degenerate_faces.sum()} degenerate faces in the mesh.")
    else:
        print("No degenerate faces were found.")
    
    # Check for non-manifold edges (edges shared by more than two faces)
    edges_sorted = mesh.edges_sorted
    edges_face_count = trimesh.grouping.group_rows(edges_sorted, require_count=True)
    
    non_manifold_edges = edges_sorted[edges_face_count > 2]
    
    if len(non_manifold_edges) > 0:
        print(f"The mesh has {len(non_manifold_edges)} non-manifold edges.")
    else:
        print("No non-manifold edges were found.")
    
    # Check for any other issues with face normals
    issues = mesh.fix_normals()
    if issues:
        print(f"Fixed {issues} issues with face normals.")

    # Size of the mesh (bounding box dimensions)
    bounding_box = mesh.bounds
    size = bounding_box[1] - bounding_box[0]  # Bounding box size (max - min)
    print(f"Bounding box size (width, height, depth): {size}")
    
    # Position of the mesh (centroid)
    centroid = mesh.centroid
    print(f"Mesh centroid position: {centroid}")
    
    # Volume of the mesh (if applicable)
    if mesh.is_volume:
        volume = mesh.volume
        print(f"Mesh volume: {volume}")
    else:
        print("Mesh volume could not be computed (not a closed volume).")
    
    print("Validation complete.")
    

def random_camera_position_near_vertices(room_mesh, max_attempts=100, offset_range=0.5):
    """
    Attempts to find a valid random camera position near the mesh vertices by applying slight offsets.
    
    Parameters:
    room_mesh (trimesh.Trimesh): The mesh of the room.
    max_attempts (int): The maximum number of attempts to find a valid position inside the room.
    offset_range (float): The range of random offsets applied to each vertex position.
    
    Returns:
    numpy array: A random position inside or near the room's vertices.
    """
    vertices = room_mesh.vertices  # Get all the vertices from the room mesh
    vertex_count = len(vertices)
    
    for attempt in range(max_attempts):
        # Randomly pick a vertex
        random_vertex = vertices[np.random.randint(0, vertex_count)]
        
        # Apply a small random offset to the vertex position to get a more varied position
        random_offset = np.random.uniform(-offset_range, offset_range, 3)
        candidate_position = random_vertex + random_offset
        
        # Log the generated candidate position
        print(f"Attempt {attempt + 1}: Candidate position: {candidate_position}")

        # Return the candidate position (without checking contains)
        return candidate_position
    
    # If no valid position is found, return the center of the bounding box
    print("Failed to find valid position near mesh vertices after max_attempts. Returning center position.")
    return (room_mesh.bounds[0] + room_mesh.bounds[1]) / 2


def visualize_bounding_box_in_room(mesh):
    """
    Create a wireframe bounding box and ensure it's positioned correctly in the room.
    """
    box = trimesh.path.creation.box_outline(bounds=mesh.bounds)
    return box



def visualize_bounding_box(mesh):
    """
    Create a wireframe bounding box for visualization.
    """
    box = trimesh.path.creation.box_outline(bounds=mesh.bounds)
    return box

def visualize_bounding_box_with_wireframe(mesh):
    """
    Create a wireframe bounding box for visualization.
    """
    box = trimesh.path.creation.box_outline(bounds=mesh.bounds)
    return box


def debug_frustum_mesh(mesh, mesh_name="Frustum"):
    """
    Prints detailed information about the given mesh for debugging.
    """
    print(f"Debugging {mesh_name}:")
    print(f" - Number of vertices: {len(mesh.vertices)}")
    print(f" - Number of faces: {len(mesh.faces)}")
    
    if not mesh.is_watertight:
        print(f" - {mesh_name} is NOT watertight.")
    else:
        print(f" - {mesh_name} is watertight.")
    
    if mesh.is_volume:
        print(f" - {mesh_name} encloses a valid volume: {mesh.volume}")
    else:
        print(f" - {mesh_name} does NOT enclose a valid volume.")
    
    # Try fixing normals
    normal_issues = mesh.fix_normals()
    print(f" - Number of normal issues fixed: {normal_issues}")

    # More detailed validation output
    print(f" - Bounding box size: {mesh.bounds[1] - mesh.bounds[0]}")
    print(f" - Centroid: {mesh.centroid}")
    
    print("------------------------------------------------------------")

def repair_and_validate(mesh, name="Frustum"):
    """
    Attempt to repair the mesh to make it watertight and valid for union.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"{name} is not a valid mesh.")
        return None

    if not mesh.is_watertight:
        print(f"{name} is not watertight. Attempting to repair...")

        # Attempt various repairs
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.merge_vertices()
        mesh = trimesh.repair.fill_holes(mesh)
        
        
    
    # Validate and provide detailed debug information
    debug_frustum_mesh(mesh, name)
    
    return mesh

def validate_mesh_before_union(mesh_list):
    """
    Validate and repair each mesh in the list before performing a union.
    """
    validated_meshes = []
    
    for idx, mesh in enumerate(mesh_list):
        print(f"Validating Mesh {idx + 1} before union...")
        repaired_mesh = repair_and_validate(mesh, f"Mesh {idx + 1}")
        
        if repaired_mesh and repaired_mesh.is_watertight:
            validated_meshes.append(repaired_mesh)
        else:
            print(f"Mesh {idx + 1} is not valid for union and will be skipped.")
    
    return validated_meshes
def repair_post_union(mesh, name="Final Frustum Union"):
    """
    Attempt to repair the final frustum union to make it watertight and valid.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"{name} is not a valid mesh.")
        return None

    if not mesh.is_watertight:
        print(f"{name} is not watertight. Attempting to repair post-union...")

        # Attempt various post-union repairs
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh = trimesh.repair.fill_holes(mesh)

        # Re-check watertightness after repair
        if mesh.is_watertight:
            print(f"{name} has been repaired and is now watertight post-union.")
        else:
            print(f"{name} could not be fully repaired and is still not watertight post-union.")
    
    # Provide debug info after post-union repair
    debug_frustum_mesh(mesh, name)
    
    return mesh



def generate_points_between(start_point, end_point, number_of_points):
    """
    Generates a list of points evenly spaced between the start and end points.
    
    Parameters:
    start_point (numpy array): The starting point (1D array like [x1, y1, z1]).
    end_point (numpy array): The ending point (1D array like [x2, y2, z2]).
    number_of_points (int): The number of intermediate points between the start and end point.
    
    Returns:
    numpy array: A list of evenly spaced points between start and end points (including both).
    """
    return np.linspace(start_point, end_point, number_of_points)


def find_nearest_and_second_nearest_points(target_point, list_of_points):
    """
    Finds the nearest and second nearest points from the list of candidate points to the target point.
    
    Parameters:
    target_point (list or numpy array): The target point (1D array or list like [x, y, z]).
    list_of_points (list of lists or numpy array): A list of candidate points (2D array or list of lists).
    
    Returns:
    tuple: (nearest_point, second_nearest_point) both as numpy arrays.
    """
    # Ensure both target_point and list_of_points are numpy arrays
    target_point = np.array(target_point)
    list_of_points = np.array(list_of_points)

    # Calculate the Euclidean distance between the target point and each point in the list
    distances = np.linalg.norm(list_of_points - target_point, axis=1)
    
    # Get the indices of the sorted distances (smallest to largest)
    sorted_indices = np.argsort(distances)
    
    # Get the nearest and second nearest points
    nearest_point = list_of_points[sorted_indices[0]]
    second_nearest_point = list_of_points[sorted_indices[1]]
    
    return nearest_point, second_nearest_point

def find_nearest_point(target_point, list_of_points):
    """
    Finds the nearest point from the list of candidate points to the target point.
    
    Parameters:
    target_point (list or numpy array): The target point (1D array or list like [x, y, z]).
    list_of_points (list of lists or numpy array): A list of candidate points (2D array or list of lists).
    
    Returns:
    numpy array: The nearest point to the target from the list of candidate points.
    """
    # Convert to numpy arrays for proper arithmetic operations
    target_point = np.array(target_point)
    list_of_points = np.array(list_of_points)
    
    # Calculate the Euclidean distance between the target point and each point in the list
    distances = np.linalg.norm(list_of_points - target_point, axis=1)
    
    # Find the index of the point with the minimum distance
    nearest_index = np.argmin(distances)
    
    # Return the nearest point
    return list_of_points[nearest_index]



def generate_points_by_angle_around_camera(camera_position, camera_rotation, radius, number_of_points=100, max_angle=np.deg2rad(370)):
    """
    Generates a list of points in the xz-plane around the camera based on its rotation, 
    covering a 270-degree arc from left to right (not covering the back).

    Parameters:
    camera_position (numpy array): The 3D position of the camera (e.g., [x, y, z]).
    camera_rotation (numpy array): The camera rotation matrix (3x3), for orientation in space.
    number_of_points (int): The number of points to generate along the arc.
    max_angle (float): The maximum angle to cover (default is 270 degrees, converted to radians).

    Returns:
    numpy array: A list of points distributed around the camera based on angle.
    """
    points = []

    # Define the angle range from -135 degrees (left) to +135 degrees (right)
    angles = np.linspace(-max_angle / 2, max_angle / 2, number_of_points)
    
    # Extract the camera's rotation in the xz-plane by projecting the forward direction onto the xz-plane
    forward_vector = camera_rotation @ np.array([1, 0, 0])  # Forward vector
    forward_vector_xz = np.array([forward_vector[0], 0, forward_vector[2]])  # Project to xz-plane
    forward_vector_xz /= np.linalg.norm(forward_vector_xz)  # Normalize the vector

    # Generate points on the xz-plane for each angle
    for angle in angles:
        # Create a rotation matrix for the current angle around the y-axis (up-axis)
        rotation_matrix = np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],  # No change in the y-axis
            [np.sin(angle), 0, np.cos(angle)]
        ])
        
        # Rotate the forward vector using the rotation matrix to get the direction for the current angle
        rotated_vector = rotation_matrix @ forward_vector_xz
        
        # Extend the rotated vector by the desired radius
        extended_point = camera_position + rotated_vector * radius
        
        # Append the point to the list
        points.append(extended_point)

    return np.array(points)


def generate_3d_mesh_from_square(square_points):
    """
    Generates a 3D mesh from the given square points in the xz-plane
    and extends it in both +y and -y directions by 5 units.
    Ensures that face normals are oriented outward.
    """
    y_offset = 5  # Extension in the y-direction (both +y and -y)
    
    # Define the top and bottom vertices of the rectangular prism
    top_vertices = square_points + np.array([0, y_offset, 0])   # Move 5 units in the +y direction
    bottom_vertices = square_points - np.array([0, y_offset, 0])  # Move 5 units in the -y direction
    
    # Combine the top and bottom vertices
    vertices = np.vstack((top_vertices, bottom_vertices))  # First 4 are top, last 4 are bottom
    
    # Define the faces of the rectangular prism using the vertex indices
    faces = np.array([
        # Top face (vertices 0, 1, 2, 3)
        [0, 1, 2],
        [0, 2, 3],
        
        # Bottom face (vertices 4, 5, 6, 7)
        [4, 5, 6],
        [4, 6, 7],
        
        # Side faces connecting top and bottom
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7]
    ])
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Check and fix normals if needed
    if not mesh.is_winding_consistent:
        mesh.fix_normals()  # This will ensure the face normals are consistent and pointing outward
    
    # Return the corrected mesh
    return mesh

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in space.

    Parameters:
    point1 (numpy array): Coordinates of the first point.
    point2 (numpy array): Coordinates of the second point.

    Returns:
    float: The distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def find_second_nearest_point(target_point, list_of_points):
    """
    Find the second nearest point from the list of candidate points to the target point.

    Parameters:
    target_point (numpy array): The target point.
    list_of_points (list or numpy array): A list of candidate points (convert to numpy array if it's a list).

    Returns:
    numpy array: The second nearest point to the target from the list of candidate points.
    """
    # Ensure list_of_points is a numpy array
    list_of_points = np.array(list_of_points)

    # Calculate the Euclidean distance between the target point and each point in the list
    distances = np.linalg.norm(list_of_points - target_point, axis=1)
    
    # Find the indices of the sorted distances (smallest to largest)
    sorted_indices = np.argsort(distances)
    
    # Return the second closest point (at index 1 in the sorted list)
    return list_of_points[sorted_indices[1]]


def process_camera_frustum(camera_position, camera_rotation, fov_horizontal, fov_vertical, near_clip, far_clip, stl_model, i :int, resolution = 2000, extension_distance = 5, threshold = 1.5):
    """
    Processes the camera frustum, performs intersection with the STL model, and subtracts the extra meshes.
    Returns the final frustum mesh ready to be appended to the list.
    """
    
    # Create the camera frustum and find farthest pair
    frustum = create_camera_frustum(camera_position, camera_rotation, fov_horizontal, fov_vertical, near_clip, far_clip)
    
    
    that_point = []
    between_points_list = generate_points_by_angle_around_camera(camera_position, camera_rotation, number_of_points=resolution, radius=far_clip*2) #extend the distance
 
    passss = []
    
    # Iterate over the points generated by angle-based distribution
    # Iterate over the points generated by angle-based distribution
    for index, between_point in enumerate(between_points_list):
        inter = find_intersections(camera_position, between_point, [stl_model])
        #print("inter = ", inter)
        #print("passs = ", len(passss))
 
    # Left side: `inter` has fewer than 2 points, and `passss` had at least 2 points
        if len(inter) < 2 and len(passss) >= 2:
            nearest_point, second_nearest_point = find_nearest_and_second_nearest_points(camera_position, np.array(passss))
            #print("right", calculate_distance(nearest_point, second_nearest_point))
            if calculate_distance(nearest_point, second_nearest_point) < threshold:
                that_point.append([nearest_point, 1 ])  # Left side, index < resolution / 2
                #print("Right side trigger", that_point)
 
    # Right side: `inter` has at least 2 points, and `passss` had fewer than 2 points
        elif len(inter) >= 2 and len(passss) < 2:
            
            nearest_point, second_nearest_point = find_nearest_and_second_nearest_points(camera_position, np.array(inter))
            #print("left", calculate_distance(nearest_point, second_nearest_point), len(passss) < 2 and index != 0)
            if len(passss) < 2 and index != 0 and calculate_distance(nearest_point, second_nearest_point) < threshold:
                that_point.append([nearest_point,0])  # Right side, index >= resolution / 2
                #print("Left side trigger", that_point)
 
    # Update passss with the current `inter` for the next iteration
        passss = inter
 
    # Generate subtraction meshes
    subtract_mesh = []
    
    for point, point_type in that_point:
        direction_vector = camera_position - point
        direction_norm = direction_vector / np.linalg.norm(direction_vector)
        extended_direction = direction_norm * 40
        extended_end_point = point - extended_direction
 
        # Compute perpendicular vector in the xz-plane
        perpendicular_direction = np.array([-direction_norm[2], 0, direction_norm[0]]) * 20
 
        # Define square points based on whether point_type is 0 or 1
        if point_type == 1:  # Left side (from the camera's perspective)
            third_point = point + perpendicular_direction
            fourth_point = extended_end_point + perpendicular_direction
            square_vertices = np.array([point, extended_end_point, fourth_point, third_point])
        else:  # Right side (from the camera's perspective)
            third_point = point - perpendicular_direction
            fourth_point = extended_end_point - perpendicular_direction
            square_vertices = np.array([point, extended_end_point, fourth_point, third_point])
 
        # Generate 3D mesh for the square
        
        square_mesh = generate_3d_mesh_from_square(square_vertices)
        #validate_3d_object(square_mesh)
        subtract_mesh.append(square_mesh)
        #print("subtract mesh", subtract_mesh)
    
    # Perform boolean operations (intersection and difference)
    frustum_intersection = trimesh.boolean.intersection([frustum, stl_model], engine='manifold')
    if not (len(frustum_intersection.vertices) == 0 or len(frustum_intersection.faces) == 0):
        
        for each_subtract_mesh in subtract_mesh:
            
            frustum_intersection = trimesh.boolean.difference([frustum_intersection] + [each_subtract_mesh], engine='manifold')
    
    
    # Validate and return the final frustum mesh
    if frustum_intersection is not None:
        print(f"Frustum {i + 1} intersection completed. Validating and repairing...")
        repaired_frustum = frustum_intersection
        
        if repaired_frustum.is_watertight:
            print("hello world")
            return repaired_frustum, frustum
        else:
            print(f"Frustum {i + 1} is still not watertight after repair and will be skipped.")
            return None, None
    else:
        print(f"Frustum {i + 1} intersection resulted in an empty mesh.")
        return None, None
    
def create_camera_scene(stl_filepath: str,
                        position_set: list[np.ndarray],
                        rotation_set: list[np.ndarray],
                        show_wireframe: bool = True,
                        resolution: int = 2000,
                        fov_horizontal: float = 80.4,
                        fov_vertical: float = 58.1,
                        near_clip: float = 0.3,
                        far_clip: float = 7.74,
                        want_coverage_volume_ratio: bool = False,
                        threshold = 1.5) -> trimesh.Scene:
    """
    Creates a scene with multiple camera frustums inside a room loaded from an STL file.
 
    Parameters:
    - stl_filepath (str): Path to the STL file containing the room model.
    - position_set (list[np.ndarray]): List of camera positions as 3D vectors.
    - rotation_set (list[np.ndarray]): List of camera rotations as 3x3 matrices.
    - show_wireframe (bool): If True, displays the wireframe of the room. Default is True.
    - resolution (int): Number of points to calculate between frustum vertices. Default is 100.
    - fov_horizontal (float): Horizontal field of view in degrees. Default is 80.4.
    - fov_vertical (float): Vertical field of view in degrees. Default is 58.1.
    - near_clip (float): Near clipping plane distance. Default is 0.3.
    - far_clip (float): Far clipping plane distance. Default is 7.74.
 
    Returns:
    - trimesh.Scene: The scene with frustums, room model, and optional wireframe.
    """
    
    # Load the STL file (room model)
    stl_model = trimesh.load(stl_filepath)
 
    # Initialize scene objects
    scene_objects = []
    frustum_set = []
    raw_frustum_set = []
 
    # Process each camera
    for i in range(len(position_set)):
        if not stl_model.contains([position_set[i]])[0]:
            print("camera position not in the room\nskipping camera ", i)
            continue
            
        camera_position = position_set[i]
        camera_rotation = rotation_set[i]
 
        # Process each camera frustum
        final_frustum, raw_frustum= process_camera_frustum(camera_position, camera_rotation, fov_horizontal, fov_vertical, near_clip, far_clip, stl_model, i, resolution, threshold=threshold)
        if final_frustum is not None:
            frustum_set.append(final_frustum)
        scene_objects
        if raw_frustum is not None:
            raw_frustum_set.append(raw_frustum)
    # Perform union of valid frustums
    if frustum_set:
        final_frustum = trimesh.boolean.union(frustum_set, engine='manifold')
 
        if final_frustum is not None:
            debug_frustum_mesh(final_frustum, "Final Frustum Union")
            scene_objects.append(final_frustum)
        else:
            print("Final frustum union resulted in an empty mesh.")
    else:
        print("No valid frustums for union.")
    if raw_frustum_set:
        final_raw = trimesh.boolean.union(raw_frustum_set, engine='manifold')
        final_raw = trimesh.boolean.difference([final_raw] + [stl_model], engine='manifold')
        validate_3d_object(final_raw)
 
    # Add the wireframe for the room model, if needed
    if show_wireframe:
        add_wireframe_visualization(stl_model, scene_objects=scene_objects)
    
    # Create the scene
    scene = trimesh.Scene(scene_objects)
    if(want_coverage_volume_ratio):
        if frustum_set:
            ratio = final_frustum.volume / stl_model.volume
        else :
            ratio = 0
        outside_volume = 0
        if(final_raw.is_volume):
            outside_volume = final_raw.volume
        return scene, ratio, outside_volume

    return scene



def rotation_matrix_x(theta): # this function perfrom rotation in this or der rotate x -> rotate y -> rotate z
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
def rotate(theta_x, theta_y, theta_z):
    # Create the individual rotation matrices
    rotation_x = rotation_matrix_x(theta_x)
    rotation_y = rotation_matrix_y(theta_y)
    rotation_z = rotation_matrix_z(theta_z)

# Combine the rotations: first rotate around X, then Y, then Z
    camera_rotation = rotation_z @ rotation_y @ rotation_x
    return camera_rotation
