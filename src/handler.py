from difflib import SequenceMatcher

# 6. Handle input categories with Pengeluaran + Kategori
def handle_input_category(input_category, dataset_categories):
    best_match = None
    best_ratio = 0.0
    
    # Pencocokan input pengguna dengan data kombinasi 'Pengeluaran + Kategori Asli'
    for pengeluaran, kategori_asli in zip(dataset_categories['Pengeluaran'], dataset_categories['Kategori_Asli']):
        combined_category = f"{pengeluaran} {kategori_asli}"  # Gabungkan Pengeluaran + Kategori Asli
        ratio = SequenceMatcher(None, input_category.lower(), combined_category.lower()).ratio()  # Case insensitive matching
        
        if ratio > best_ratio and ratio > 0.45:  # Ambil kecocokan terbaik di atas threshold
            best_match = kategori_asli
            best_ratio = ratio

    if best_match:
        print(f"Kategori ditemukan: '{best_match}', Ratio: {best_ratio:.2f}")
        return input_category, best_match  # Kategori asli ditemukan
    else:
        print(f"Kategori tidak ditemukan. Menggunakan input pengguna sebagai 'Lain lain', Ratio: {best_ratio:.2f}")
        return input_category, "Lain lain"  # Kategori backend "Lain lain", tetapi tampilkan input asli pengguna
