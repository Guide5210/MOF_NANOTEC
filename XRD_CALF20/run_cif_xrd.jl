using PyCall

# เรียกใช้โมดูลจาก pymatgen
pmg_core = pyimport("pymatgen.core")
pmg_xrd = pyimport("pymatgen.analysis.diffraction.xrd")

function convert_cif_to_xy(cif_file, output_xy)
    try
        mof_structure = pmg_core.Structure.from_file(cif_file)
        calculator = pmg_xrd.XRDCalculator(wavelength="CuKa")
        pattern = calculator.get_pattern(mof_structure)

        open(output_xy, "w") do f
            for i in 1:length(pattern.x)
                println(f, "$(pattern.x[i])\t$(pattern.y[i])")
            end
        end
        println("✅ Successfully converted: $cif_file -> $output_xy")
    catch e
        println("❌ Error processing $cif_file: $e")
    end
end

println("Starting XRD Pattern simulation for CALF-20 phases...")

# 1. จัดการเฟส Alpha
if isfile("alpha_calf20_ref.cif")
    convert_cif_to_xy("alpha_calf20_ref.cif", "alpha_calf20.xy")
else
    println("⚠️ File not found: alpha_calf20_ref.cif")
end

# 2. จัดการเฟส Gamma
if isfile("gamma_calf20_ref.cif")
    convert_cif_to_xy("gamma_calf20_ref.cif", "gamma_calf20.xy")
else
    println("⚠️ File not found: gamma_calf20_ref.cif")
end

# 3. จัดการเฟส Tau (และทำสำเนาเป็นเฟส Beta ให้ด้วย)
if isfile("tau_calf20_ref.cif")
    convert_cif_to_xy("tau_calf20_ref.cif", "tau_calf20.xy")
    # นำข้อมูลจากไฟล์ tau ไปสร้างเป็นไฟล์ beta ให้โดยอัตโนมัติ
    convert_cif_to_xy("tau_calf20_ref.cif", "beta_calf20.xy")
else
    println("⚠️ File not found: tau_calf20_ref.cif")
end

println("All selected conversions finished!")