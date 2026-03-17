import time

def loop_timer():
    # ----------------------------------------
    # Mulai pencatatan total waktu
    # ----------------------------------------
    start_total = time.time()

    # Menyimpan waktu terakhir (untuk menghitung selisih per loop)
    last = time.time()
    
    # Generator menunggu perintah pertama
    command = yield
    
    # ----------------------------------------
    # Loop utama generator
    # ----------------------------------------
    while True:

        # ----------------------
        # MODE: hitung waktu iterasi
        # ----------------------
        if command == "tick":
            now = time.time()        # waktu sekarang
            dt = now - last          # selisih dari iterasi sebelumnya
            last = now               # update waktu terakhir

            # Kembalikan durasi iterasi
            command = yield dt


        # ----------------------
        # MODE: selesai → hitung total waktu
        # ----------------------
        elif command == "done":
            end_total = time.time()

            # Kembalikan dictionary info waktu
            yield {
                "total_time": end_total - start_total,   # total durasi semua iterasi
                "avg_time": (end_total - start_total),   # avg dihitung caller di luar
            }

            return    # hentikan generator


        # ----------------------
        # MODE: command tidak dikenal
        # ----------------------
        else:
            command = yield None     # tidak mengembalikan apa-apa

def format_time(seconds):
    seconds = int(seconds)

    jam = seconds // 3600
    menit = (seconds % 3600) // 60
    detik = seconds % 60

    if jam > 0:
        return f"{jam} jam {menit} menit {detik} detik"
    elif menit > 0:
        return f"{menit} menit {detik} detik"
    else:
        return f"{detik} detik"