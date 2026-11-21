import os
import uuid
import json

import numpy as np
import open3d as o3d

from flask import Flask, render_template, request, redirect, url_for, flash

from poisson_engine import reconstruct_tumor


app = Flask(__name__)

# Used by Flask for flash messages
app.secret_key = "helios-poisson-secret-key"


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

def allowed_file(filename):

    filename = filename.lower()

    return (
        filename.endswith(".nii") or
        filename.endswith(".nii.gz")
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD + POISSON RECONSTRUCTION
# ============================================================

@app.route("/reconstruct", methods=["POST"])
def reconstruct():

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        flash("No file selected.")

        return redirect(
            url_for("index")
        )


    file = request.files["file"]


    if file.filename == "":

        flash("Please select a NIfTI file.")

        return redirect(
            url_for("index")
        )


    if not allowed_file(file.filename):

        flash(
            "Invalid file. Please upload a .nii or .nii.gz file."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    original_filename = file.filename

    unique_id = uuid.uuid4().hex[:8]

    safe_filename = (
        unique_id
        + "_"
        + original_filename
    )

    input_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )


    # Save uploaded NIfTI file
    file.save(input_path)


    print("\n" + "=" * 60)
    print("NEW HELIOS RECONSTRUCTION REQUEST")
    print("=" * 60)

    print("Uploaded file:")
    print(original_filename)

    print("Saved as:")
    print(input_path)


    # ========================================================
    # RUN RECONSTRUCTION
    # ========================================================

    try:

        print("\nStarting Poisson reconstruction...\n")


        result = reconstruct_tumor(
            input_path,
            RESULT_DIR
        )


        # ====================================================
        # LOAD GENERATED PLY MESH
        # ====================================================

        mesh_file = result["mesh_file"]

        print("\nLoading generated mesh for visualization...")

        mesh = o3d.io.read_triangle_mesh(
            mesh_file
        )


        if mesh.is_empty():

            raise RuntimeError(
                "Generated Poisson mesh is empty."
            )


        print(
            "Mesh loaded successfully."
        )

        print(
            "Vertices:",
            len(mesh.vertices)
        )

        print(
            "Triangles:",
            len(mesh.triangles)
        )


        # ====================================================
        # EXTRACT MESH DATA
        # ====================================================

        vertices = np.asarray(
            mesh.vertices
        )

        triangles = np.asarray(
            mesh.triangles
        )


        # Convert NumPy arrays to normal Python lists
        vertices_data = vertices.tolist()

        triangles_data = triangles.tolist()


        # Convert to JSON for JavaScript
        vertices_json = json.dumps(
            vertices_data
        )

        triangles_json = json.dumps(
            triangles_data
        )


        # ====================================================
        # DISPLAY RESULT PAGE
        # ====================================================

        return render_template(

            "result.html",

            # Original uploaded file
            original_filename=original_filename,

            # Patient information
            patient_id=result["patient_id"],

            # Tumor statistics
            tumor_voxels=result["tumor_voxels"],

            tumor_volume_mm3=result["tumor_volume_mm3"],

            surface_points=result["surface_points"],

            # Mesh statistics
            vertices=result["vertices"],

            triangles=result["triangles"],

            # Mesh data for Plotly
            vertices_data=vertices_json,

            triangles_data=triangles_json

        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            str(e)
        )

        print("=" * 60)


        flash(
            f"Reconstruction failed: {str(e)}"
        )

        return redirect(
            url_for("index")
        )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "HELIOS POISSON RECONSTRUCTION WEB APPLICATION"
    )
    print("=" * 60)

    print("\nOpen the following address in your browser:")

    print(
        "http://127.0.0.1:5000"
    )

    print("\nStarting Flask server...\n")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
