from poisson_engine import reconstruct_tumor


input_file = "/mnt/d/Saniya/Helios/Helios/predictions/LUNG1-001_prediction.nii.gz"

output_dir = "/mnt/d/Saniya/Helios/Helios/web_app/test_output"


result = reconstruct_tumor(
    input_file,
    output_dir
)


print("\nRESULT")
print("=" * 40)

for key, value in result.items():
    print(f"{key}: {value}")
