import os
import numpy as np
import nibabel as nib
import open3d as o3d


def reconstruct_tumor(input_file, output_dir):
    """
    Perform Poisson Surface Reconstruction on a
    MedSAM2 predicted 3D tumor segmentation.

    Parameters
    ----------
    input_file : str
        Path to .nii or .nii.gz segmentation file.

    output_dir : str
        Directory where generated files will be saved.

    Returns
    -------
    dict
        Reconstruction information and output file paths.
    """

    print("=" * 60)
    print("POISSON RECONSTRUCTION")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD NIFTI SEGMENTATION
    # --------------------------------------------------------

    print("\nLoading segmentation...")

    img = nib.load(input_file)

    mask = img.get_fdata()

    # Convert segmentation into binary mask
    mask = mask > 0

    affine = img.affine
    voxel_spacing = img.header.get_zooms()

    tumor_voxels = int(np.sum(mask))

    print("Mask shape:", mask.shape)
    print("Voxel spacing:", voxel_spacing)
    print("Tumor voxels:", tumor_voxels)

    if tumor_voxels == 0:
        raise ValueError("No tumor voxels found in the uploaded segmentation.")

    # --------------------------------------------------------
    # 2. EXTRACT TUMOR SURFACE
    # --------------------------------------------------------

    print("\nExtracting tumor surface points...")

    surface_points_voxel = []
    surface_normals_voxel = []

    directions = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    shape = mask.shape

    for dx, dy, dz in directions:

        shifted = np.zeros_like(mask, dtype=bool)

        x_from = max(0, -dx)
        x_to = min(shape[0], shape[0] - dx)

        y_from = max(0, -dy)
        y_to = min(shape[1], shape[1] - dy)

        z_from = max(0, -dz)
        z_to = min(shape[2], shape[2] - dz)

        shifted[
            x_from:x_to,
            y_from:y_to,
            z_from:z_to
        ] = mask[
            x_from + dx:x_to + dx,
            y_from + dy:y_to + dy,
            z_from + dz:z_to + dz
        ]

        # Tumor voxel whose neighboring voxel is background
        exposed = mask & (~shifted)

        indices = np.argwhere(exposed)

        if len(indices) == 0:
            continue

        # Voxel center
        centers = indices.astype(np.float64) + 0.5

        # Move to exposed face center
        direction = np.array(
            [dx, dy, dz],
            dtype=np.float64
        )

        face_centers = centers + 0.5 * direction

        normals = np.tile(
            direction,
            (len(indices), 1)
        )

        surface_points_voxel.append(face_centers)
        surface_normals_voxel.append(normals)

    if not surface_points_voxel:
        raise ValueError("Could not extract tumor surface.")

    surface_points_voxel = np.vstack(surface_points_voxel)
    surface_normals_voxel = np.vstack(surface_normals_voxel)

    print(
        "Surface points:",
        len(surface_points_voxel)
    )

    # --------------------------------------------------------
    # 3. CONVERT TO PHYSICAL COORDINATES
    # --------------------------------------------------------

    print("\nConverting to physical coordinates...")

    surface_points = nib.affines.apply_affine(
        affine,
        surface_points_voxel
    )

    linear_transform = affine[:3, :3]

    surface_normals = (
        surface_normals_voxel @ linear_transform.T
    )

    norms = np.linalg.norm(
        surface_normals,
        axis=1
    )

    valid = norms > 0

    surface_points = surface_points[valid]
    surface_normals = surface_normals[valid]

    surface_normals = (
        surface_normals /
        norms[valid, None]
    )

    print(
        "Physical points:",
        len(surface_points)
    )

    # --------------------------------------------------------
    # 4. CREATE OPEN3D POINT CLOUD
    # --------------------------------------------------------

    print("\nCreating Open3D point cloud...")

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        surface_points
    )

    pcd.normals = o3d.utility.Vector3dVector(
        surface_normals
    )

    print("Point cloud created.")

    # --------------------------------------------------------
    # 5. SAVE POINT CLOUD
    # --------------------------------------------------------

    base_name = os.path.basename(input_file)

    if base_name.endswith(".nii.gz"):
        patient_id = base_name[:-7]
    elif base_name.endswith(".nii"):
        patient_id = base_name[:-4]
    else:
        patient_id = os.path.splitext(base_name)[0]

    point_cloud_file = os.path.join(
        output_dir,
        patient_id + "_surface_points.ply"
    )

    print("\nSaving point cloud...")

    o3d.io.write_point_cloud(
        point_cloud_file,
        pcd
    )

    print(
        "Saved:",
        point_cloud_file
    )

    # --------------------------------------------------------
    # 6. POISSON RECONSTRUCTION
    # --------------------------------------------------------

    print("\nRunning Poisson reconstruction...")

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            pcd,
            depth=8
        )
    )

    print(
        "Poisson reconstruction completed."
    )

    vertices_before = len(mesh.vertices)
    triangles_before = len(mesh.triangles)

    print(
        "Vertices:",
        vertices_before
    )

    print(
        "Triangles:",
        triangles_before
    )

    # --------------------------------------------------------
    # 7. REMOVE LOW-DENSITY ARTIFACTS
    # --------------------------------------------------------

    print(
        "\nRemoving low-density regions..."
    )

    densities = np.asarray(densities)

    density_threshold = np.quantile(
        densities,
        0.02
    )

    vertices_to_remove = (
        densities < density_threshold
    )

    mesh.remove_vertices_by_mask(
        vertices_to_remove
    )

    vertices_after = len(mesh.vertices)
    triangles_after = len(mesh.triangles)

    print(
        "Remaining vertices:",
        vertices_after
    )

    print(
        "Remaining triangles:",
        triangles_after
    )

    # --------------------------------------------------------
    # 8. SAVE FINAL MESH
    # --------------------------------------------------------

    mesh_file = os.path.join(
        output_dir,
        patient_id + "_poisson_mesh.ply"
    )

    print("\nSaving Poisson mesh...")

    o3d.io.write_triangle_mesh(
        mesh_file,
        mesh
    )

    print(
        "Saved:",
        mesh_file
    )

    # --------------------------------------------------------
    # 9. CALCULATE BASIC INFORMATION
    # --------------------------------------------------------

    voxel_volume = (
        float(voxel_spacing[0])
        * float(voxel_spacing[1])
        * float(voxel_spacing[2])
    )

    tumor_volume = tumor_voxels * voxel_volume

    print("\n" + "=" * 60)
    print("POISSON RECONSTRUCTION COMPLETE")
    print("=" * 60)

    return {
        "patient_id": patient_id,
        "tumor_voxels": tumor_voxels,
        "tumor_volume_mm3": tumor_volume,
        "surface_points": len(surface_points),
        "vertices": vertices_after,
        "triangles": triangles_after,
        "point_cloud_file": point_cloud_file,
        "mesh_file": mesh_file,
    }
