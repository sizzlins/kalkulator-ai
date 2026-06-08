import sympy as sp

print("==================================================")
print("KALKULATOR-AI: EXPLORING THE RIEMANN HYPOTHESIS")
print("==================================================")

# Sumbu kritis (Critical Line): Re(s) = 1/2
# Kita coba mengecek beberapa nilai nol non-trivial (Non-trivial zeros) pertama
# dari fungsi Riemann Zeta di sumbu kritis ini.

print("\n[1] Menghitung nilai Riemann Zeta Function pada titik-titik kritis...")

# Nilai imaginer 't' untuk akar-akar pertama fungsi Zeta
t_values = [
    14.134725141734693790,
    21.022039638771554992,
    25.010857580145688763,
    30.424876125859512725
]

for n, t in enumerate(t_values, 1):
    # s = 1/2 + t*i
    s = 0.5 + t * sp.I
    
    # Menghitung Zeta(s)
    # Gunakan evalf untuk mendapatkan nilai numerik dari kompleks
    zeta_val = sp.zeta(s).evalf()
    
    # Menghitung magnitudo (jarak ke nol)
    magnitude = abs(zeta_val)
    
    print(f"Akar ke-{n}: t = {t:.5f}")
    print(f"   Zeta(1/2 + {t:.5f}*i) = {zeta_val}")
    print(f"   Jarak ke 0 (Magnitude): {magnitude:.5e}\n")


print("[2] Kesimpulan AI Kalkulator:")
print("Aplikasi Kalkulator-AI (melalui sympy) memvalidasi bahwa nilai Zeta pada")
print("titik-titik tersebut memang SANGAT mendekati 0.")
print("NAMUN, AI dan komputer hanya bisa 'mengecek' angka secara berurutan.")
print("Untuk MEMBUKTIKAN Hipotesis Riemann (bahwa SEMUA akar ada di garis 1/2),")
print("dibutuhkan pembuktian logika deduktif murni, bukan komputasi numerik.")
print("AI saat ini belum memiliki kemampuan untuk melakukan pembuktian matematika")
print("tingkat lanjut sekelas Millenium Prize secara mandiri.")
print("==================================================")
