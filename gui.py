import cv2
import numpy as np
from skimage.filters import frangi

# =====================================================
# Fungsi: menerapkan satu operasi preprocessing
# img : citra input (grayscale / RGB)
# key : nama preprocessing yang akan dijalankan
# =====================================================
def apply_preprocess(img, key):
    # =================================================
    # 1. GREEN CHANNEL
    # =================================================
    # Mengambil kanal hijau dari citra RGB
    # Alasan:
    # - Kanal hijau memiliki kontras terbaik untuk pembuluh darah & MA
    # - Noise lebih rendah dibanding kanal merah & biru
    if key == "green_channel":
        return img[:, :, 1] if len(img.shape) == 3 else img
    
    # =================================================
    # 2. CROP RETINA (Contour-based)
    # =================================================
    elif key == "crop_retina":
        # Pastikan input berupa grayscale (green channel)
        green_channel = img[:, :, 1] if len(img.shape) == 3 else img
        # Threshold sederhana untuk memisahkan area retina dan background hitam
        # Pixel > 30 dianggap bagian retina
        _, thresh = cv2.threshold(
            green_channel, 20, 255, cv2.THRESH_BINARY
        )
        # Mencari kontur dari mask biner
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Jika tidak ada kontur, kembalikan gambar asli
        if not contours:
            return img
        # Ambil kontur terbesar (diasumsikan sebagai area retina)
        c = max(contours, key=cv2.contourArea)
        # Bounding box dari kontur retina
        x, y, w, h = cv2.boundingRect(c)
        # Membuat background hitam dengan ukuran gambar asli
        black_bg = np.zeros_like(img)
        # Menggambar kontur retina dan mengisinya (mask ROI)
        cv2.drawContours(black_bg, [c], -1, (255), thickness=cv2.FILLED)
        # Mengambil hanya area retina (ROI) dari gambar asli
        roi = cv2.bitwise_and(img, black_bg)
        # Crop berdasarkan bounding box retina
        return roi[y:y+h, x:x+w]

    # =================================================
    # 3. NORMALISASI INTENSITAS
    # =================================================
    elif key.startswith("normalize"):
        if key == "normalize_0_255":
            return cv2.normalize(
                img, None, 0, 255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
        elif key == "normalize_0_1":
            return cv2.normalize(
                img, None, 0, 1,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )


    # =================================================
    # 4. ILLUMINATION CORRECTION
    # =================================================
    # Menghilangkan pencahayaan tidak merata
    # Prinsip:
    # - Estimasi background menggunakan Gaussian Blur
    # - Kurangi background dari citra asli
    elif key == "illumination_correction":
        H, W = img.shape
        # Adaptive sigma (8-12% dari ukuran minimum citra)
        sigma = int(0.1 * min(H, W))
        sigma = np.clip(sigma, 25, 60)
        # Background illumination
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        bg[bg < 1] = 1.0  # menghindari divide by zero
        # Division-based correction
        corrected = img / bg
        corrected = corrected * np.mean(bg)
        # Normalize
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        return corrected.astype(np.uint8)

    # =================================================
    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # =================================================
    # Meningkatkan kontras lokal tanpa memperbesar noise
    # Sangat efektif untuk menonjolkan MA kecil
    elif key == "clahe":
        h, w = img.shape

        tile = max(8, int(min(h, w) / 64))
        tileGridSize = (tile, tile)

        std = np.std(img)
        clip = np.clip(std / 40, 1.1, 3.0)

        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=tileGridSize
        )
        return clahe.apply(img) 

    # =================================================
    # 6. FILTER / BLUR
    # =================================================
    # Gaussian Blur: mereduksi noise halus
    elif key == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)

    # Difference of Gaussian (DoG)
    # Menonjolkan blob kecil seperti microaneurysm
    elif key == "difference_of_gaussian":
        blur1 = cv2.GaussianBlur(img, (3, 3), 0)
        blur2 = cv2.GaussianBlur(img, (7, 7), 0)
        return cv2.subtract(blur1, blur2)

    # Median Blur
    # Efektif menghilangkan noise salt-and-pepper
    elif key == "median_blur":
        return cv2.medianBlur(img, 3)

    # Bilateral Filter
    # Menghaluskan noise tanpa merusak edge
    elif key == "bilateral":
        return cv2.bilateralFilter(img, d=3, sigmaColor=30, sigmaSpace=30)


    # =================================================
    # 7. MORPHOLOGICAL OPERATIONS
    # =================================================
    elif key.startswith("morph_"):
        # Ambil jenis operasi morphology
        op = key.replace("morph_", "")

        # Kernel elips (cocok untuk struktur retina)
        k_size = 5 if op == "tophat" else 3
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size)
        )

        morph_ops = {
            "erode"   : lambda: cv2.erode(img, kernel, iterations=1),
            "dilate"  : lambda: cv2.dilate(img, kernel, iterations=1),
            "open"    : lambda: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
            "close"   : lambda: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
            "gradient": lambda: cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
            "tophat"  : lambda: cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel),
            "blackhat": lambda: cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel),
        }

        if op in morph_ops:
            return morph_ops[op]()
        else:
            print(f"Operasi morphology `{op}` tidak dikenal!")
            return None

    # =================================================
    # 8. VESSEL REMOVAL
    # - Menghilangkan pembuluh darah besar
    # - Mempertahankan struktur microaneurysm (MA)
    # =================================================
    elif key == "vessel_remove":
        # Konversi ke float dan normalisasi
        img_f = img.astype(np.float32) / 255.0

        # ----------------------------------------------------------
        # Deteksi pembuluh darah menggunakan Frangi filter
        # ----------------------------------------------------------
        sigmas = np.arange(1.5, 4.5, 1)

        vesselness = frangi(
            img_f,
            sigmas=sigmas,
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=True
        )

        # Normalisasi vesselness ke [0, 1]
        vesselness = cv2.normalize(vesselness, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)

        # ----------------------------------------------------------
        # Adaptive vessel mask
        # ----------------------------------------------------------
        thr = np.percentile(vesselness, 85)
        vessel_mask = vesselness > thr

        # Perhalus mask agar inpainting stabil
        vessel_mask_u8 = (vessel_mask * 255).astype(np.uint8)
        vessel_mask_u8 = cv2.GaussianBlur(vessel_mask_u8, (3, 3), 0)

        # ----------------------------------------------------------
        # Vessel suppression menggunakan inpainting
        # ----------------------------------------------------------
        suppressed = cv2.inpaint(img, vessel_mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # ----------------------------------------------------------
        # Gentle border suppression
        # ----------------------------------------------------------
        retina_mask = img > 10
        retina_mask = cv2.morphologyEx(retina_mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

        # mengambil ring tipis di tepi retina
        border = cv2.morphologyEx(retina_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)

        # menghaluskan hanya area border
        blur = cv2.GaussianBlur(suppressed, (11, 11), 0)
        suppressed[border] = blur[border]

        return cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # =================================================
    # 9. OPTIC DISC SUPPRESSION
    # =================================================
    # Menghilangkan optic disc (terang besar)
    # agar tidak mendominasi fitur
    elif key == "od_suppress":
        # ===============================
        # 1. Threshold adaptif (bukan fixed 230)
        # ===============================
        thresh = np.percentile(img, 98)
        _, od = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        # ===============================
        # 2. Ambil hanya area terang TERBESAR (OD)
        # ===============================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            od.astype(np.uint8), connectivity=8
        )

        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            od = (labels == largest).astype(np.uint8) * 255
        else:
            return img

        # ===============================
        # 3. Dilasi ringan (ikut bentuk bulat)
        # ===============================
        od = cv2.dilate(
            od,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        )

        # ===============================
        # 4. Inpainting
        # ===============================
        return cv2.inpaint(img, od, 5, cv2.INPAINT_TELEA)

    # =================================================
    # 10. MA ENHANCEMENT
    # =================================================
    # Kombinasi TopHat + DoG untuk menonjolkan microaneurysm
    elif key == "ma_enhance":
        # TopHat menonjolkan objek kecil terang
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_small)

        # Difference of Gaussian
        blur_small = cv2.GaussianBlur(img, (3, 3), 0)
        blur_large = cv2.GaussianBlur(img, (9, 9), 0)
        dog = cv2.subtract(blur_small, blur_large)

        tophat = np.maximum(tophat, 0)
        dog = np.maximum(dog, 0)

        # Kombinasi linear
        ma = cv2.addWeighted(tophat, 0.6, dog, 0.4, 0)
        
        return cv2.normalize(ma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # =================================================
    # 11. RESIZE
    # =================================================
    elif key.startswith("resize_"):

        # Resize dengan menjaga rasio + padding hitam
        if "thumbnail" in key:
            size = key.replace("resize_thumbnail_", "")
            target_w, target_h = map(int, size.split("x"))

            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)

            interpolation = (cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

            # Canvas hitam
            canvas_shape = ((target_h, target_w, 3) if len(img_resized.shape) == 3 else (target_h, target_w))
            canvas = np.zeros(canvas_shape, dtype=np.uint8)

            # Centering
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

            return canvas

        # Resize langsung (tanpa padding)
        else:
            size = key.replace("resize_", "")
            target_w, target_h = map(int, size.split("x"))
            return cv2.resize(img, (target_w, target_h))

    print(f"Preprocessing `{key}` tidak tersedia! Melewati...")
    return None

# =====================================================
# Fungsi preprocessing pipeline
# =====================================================
def preprocessing(img, params):
    # Validasi parameter
    if isinstance(params, dict):
        raise TypeError("Parameter 'params' tidak boleh dictionary! Gunakan list.")
    if not isinstance(params, list):
        raise TypeError("Parameter 'params' harus berupa list, misalnya: ['normalize', 'green_channel'].")

    # Load gambar jika berupa path string
    if isinstance(img, str):
        img = cv2.imread(img)

    # Jika gambar berupa PIL Image
    elif isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Simpan hasil setiap langkah
    result = {'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB)}

    # Terapkan preprocessing sesuai urutan
    for key in params:
        last_key = list(result.keys())[-1]
        last_img = result[last_key]
        processed = apply_preprocess(last_img, key)
        if processed is None:
            continue

        # Hindari duplikasi key
        new_key = key
        count = 1
        while new_key in result:
            new_key = f"{key}_{count}"
            count += 1

        result[new_key] = processed

    return result

from scipy.ndimage import map_coordinates, uniform_filter, sobel
from skimage.util import view_as_windows
from skopt import gp_minimize
from skopt.space import Real

# =====================================================================
# 1. Bilinear Interpolation Sampling
# =====================================================================
def bilinear_interpolate_full(image, x, y):
    return map_coordinates(image, [[y], [x]], order=1, mode='reflect')[0]

# =====================================================================
# 2. Noise-Aware Thresholding
# =====================================================================
def compute_noise_threshold_map(img, size=3, k=0.5):
    mu = uniform_filter(img, size=size, mode='reflect')
    mu2 = uniform_filter(img * img, size=size, mode='reflect')
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    # return mu + k * sigma
    return mu - k * sigma   # MA lebih gelap dari background

# =====================================================================
# 3. Graph Connectivity Weighting
# =====================================================================
def get_graph_neighbors(P, R):
    angles = np.linspace(0, 2*np.pi, P, endpoint=False)
    coords = [(R*np.cos(a), R*np.sin(a)) for a in angles]
    return coords

def adaptive_normalize(img, eps=1e-8):
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()

    # Normalisasi hanya jika belum pada rentang stabil
    if vmax > 1.0 + eps:
        img = (img - vmin) / (vmax - vmin + eps)


    return img

# Jarak
def distance_weight(dx, dy):
    d = np.sqrt(dx**2 + dy**2)
    return 1.0 / (d + 1e-8)

# Gradien Lokal
def gradient_weight(img, size=3, eps=1e-8):
    img = adaptive_normalize(img)

    gx = sobel(img, axis=1, mode='reflect')
    gy = sobel(img, axis=0, mode='reflect')

    grad_mag = np.sqrt(gx**2 + gy**2)

    # Normalisasi berbasis statistik lokal
    local_mean = uniform_filter(grad_mag, size=size)
    grad_w = grad_mag / (local_mean + eps)

    return grad_w

# Konsistensi pola MA
def consistency_weight(img, size=3, eps=1e-8):
    img = adaptive_normalize(img)

    # --------------------------------------------------
    # 2) MA consistency
    # --------------------------------------------------
    local_mean = uniform_filter(img, size=size)
    diff = np.abs(img - local_mean)

    # Normalisasi berbasis statistik lokal
    local_diff_mean = uniform_filter(diff, size=size)

    konstitusi = 1.0 / (1.0 + diff / (local_diff_mean + eps))

    return konstitusi

# =====================================================================
# 5. Graph-Based LBP Calculation
    # - Bilinear interpolation (Modifikasi 1)
    # - Noise-aware threshold (Modifikasi 2)
    # - Graph connectivity weighting (Modifikasi 3)
# =====================================================================
def graph_based_lbp(image, P=8, R=1, k_thresh=0.5, ent_win=3, n_bins=8):
    img = image.astype(np.float32)
    # NORMALISASI GLOBAL
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    H, W = img.shape

    center = img
    # Meshgrid piksel
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Entropy Map
    entropy_map = compute_entropy_map(img, win=ent_win, n_bins=n_bins)
    # Konstitusi pola MA
    konstitusi = consistency_weight(entropy_map)
    # Modifikasi 2: Noise Aware Thresholding
    noise_aware = compute_noise_threshold_map(center, k=k_thresh)
    # Modifikasi 3: Graph Connectivity Weighting
    # Gradien Lokal
    grad = gradient_weight(img)
    # Hitung Bobot | Memberi bobot lebih besar pada area penting (edge + pola konsisten)
    W = (1 + grad) * konstitusi
    W = W / (np.mean(W) + 1e-8)
    # W = np.clip(W, 0.5, 2.0)

    # Neighbor sampling
    neighbors = get_graph_neighbors(P, R)

    bits = []
    for i, (dx, dy) in enumerate(neighbors):
        # Modifikasi 1: Bilinear Interpolation
        intensitas = bilinear_interpolate_full(img, xx + dx, yy + dy)
        
        # Tambahkan bobot pada tetangga
        diff = intensitas - noise_aware
        bit = (diff * W >= 0).astype(np.uint8)

        bits.append(bit)

    # Encode LBP
    bits = np.stack(bits, axis=-1)

    return bits

# ==============================================================
# 6. Entropy Map
# ==============================================================
def compute_entropy_map(img, win=3, n_bins=8, show_mask=False):
    # Buat patch sliding window: shape (H-win+1, W-win+1, win, win)
    patches = view_as_windows(img, (win, win))
    ph, pw = patches.shape[:2]

    # Flatten 1 patch = win*win piksel
    flat = patches.reshape(-1, win * win)

    # Histogram quantization
    bins = np.linspace(img.min(), img.max() + 1e-6, n_bins + 1)
    inds = np.digitize(flat, bins) - 1

    # Hitung histogram 8 bin
    hist = np.zeros((len(flat), n_bins), dtype=np.float32)
    for b in range(n_bins):
        hist[:, b] = (inds == b).sum(axis=1)

    # Normalisasi: jumlah piksel per window = win*win
    probs = hist / (win * win)

    # Hitung entropy tiap window
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0), axis=1)

    # Kembalikan ke bentuk spasial
    entropy_map = ent.reshape(ph, pw)
    # Ubah nilai kurang dari 0 menjadi 0
    entropy_map[entropy_map < 1e-8] = 0.0

    # Padding Agar ukuran Kembali
    pad = win // 2
    entropy_map = np.pad(entropy_map, pad_width=pad, mode='reflect')

    # Kandidat MA
    mask = entropy_map > (entropy_map.mean() + 0.5 * entropy_map.std())
    entropy_map = np.where(mask, entropy_map, 0.3 * entropy_map)

    if show_mask:
        return mask, entropy_map

    return entropy_map

# ==============================================================
# 7. Encode G-LBP (UNIFORM, LBP^u2)
# ==============================================================
def remove_border(green_channel, glbp_img, R=1, P=8, method='default'):
    h, w = green_channel.shape
    mask = np.zeros((h, w), dtype=bool)

    # area dalam
    mask[R:h-R, R:w-R] = True

    # threshold adaptif
    threshold = 10
    mask &= (green_channel > threshold)

    # set background agar selalu terang
    if method == 'uniform':
        glbp_img[~mask] = P + 1
    else:
        glbp_img[~mask] = 2 ** P - 1

    return glbp_img

def encode_glbp(green_channel, bits, R=1, P=8, method="uniform"):
    if method == 'uniform':
        # uniform LBP → P + 2 bin
        transitions = np.sum(bits != np.roll(bits, 1, axis=-1), axis=-1)
        ones_count = np.sum(bits, axis=-1)
        glbp_image = np.where(transitions <= 2, ones_count, P + 1).astype(np.uint8)
    else:  # default
        powers_of_two = 2 ** np.arange(P)
        glbp_image = np.tensordot(bits, powers_of_two, axes=([2], [0])).astype(np.uint8)

    # Remove Border
    glbp_image = remove_border(
        green_channel=green_channel,
        glbp_img=glbp_image,
        R=R,
        P=P,
        method=method
    )

    return glbp_image


# ==============================================================
# 7. Histogram G-LBP (UNIFORM, LBP^u2)
# ==============================================================
def glbp_histogram(glbp_image, P=8, method="uniform"):
    if method == 'uniform':
        # uniform LBP → P + 2 bin
        n_bins = P + 2
    else:
        n_bins = 2**P

    # Histogram
    hist, _ = np.histogram(
        glbp_image.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Normalisasi
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)

    feature_names = [f"LBP_{i}" for i in range(n_bins)]
    features = dict(zip(feature_names, hist))

    return features

# ==============================================================
# 8. Adaptive Radius
# ==============================================================

# Kontras Lokal
def local_contrast(img, win=3):
    img = img.astype(np.float32)
    mean = uniform_filter(img, win)
    mean2 = uniform_filter(img**2, win)
    std = np.sqrt(np.maximum(mean2 - mean**2, 0))
    return std

# Adaptive Radius
def adaptive_radius_map(img, 
                        R_list=[1, 2, 3],
                        ent_win=3, n_bins=8):
    img = img.astype(np.uint8)

    # Step 1: Kontras lokal
    contrast = local_contrast(img)

    # Step 2: Entropy Map
    entropy_map = compute_entropy_map(img, win=ent_win, n_bins=n_bins)

    # Skor
    # Normalisasi skor (0–1)
    contrast_n = (contrast - contrast.min()) / (np.ptp(contrast) + 1e-8)
    entropy_n  = (entropy_map - entropy_map.min()) / (np.ptp(entropy_map) + 1e-8)
    score = 0.5 * contrast_n + 0.5 * entropy_n

    # Quantization ke beberapa radius
    bins = np.linspace(0, 1, len(R_list)+1)

    # Step 3: Buat peta radius | default = radius besar
    R_map = np.zeros_like(img, dtype=np.int32)

    # Skor kecil -> radius kecil/besar tergantung urutan
    for i in range(len(R_list)):
        mask = (score >= bins[i]) & (score < bins[i+1])
        R_map[mask] = R_list[i]

    return R_map, contrast, entropy_map

# =====================================================================
# 7. Adaptive Threshold Factor (k)
# =====================================================================
def adaptive_threshold_factor(image, P, R_map, 
                              k_list = [0.2, 0.3, 0.4, 0.5], 
                              n_calls=12,
                              ent_win=3, n_bins=8,
                              method_opt="grid", method_glbp="uniform"):
    method_opt = method_opt.lower()
    best_score = -np.inf
    best_k = None
    glbp_best = None
    method_name = None

    # -------------------------------------------------
    # Helper: GLBP + Adaptive Radius + Score
    # -------------------------------------------------
    # Step 1
    # Kontras lokal
    contrast = local_contrast(image)
    # # --- Threshold berdasarkan distribusi---
    t_contrast = contrast.mean() + 1.5 * contrast.std()

    def compute_glbp_adaptive_radius(k_thresh):
        glbp_dict = {}
        # Hitung semua radius
        for R in np.unique(R_map):
            # G-LBP untuk masing-masing radius
            bits = graph_based_lbp(image, P=P, R=R, k_thresh=k_thresh)
            # Encode G-LBP
            glbp = encode_glbp(
                green_channel=image,
                bits=bits,
                R=R,
                P=P,
                method=method_glbp
            )

            glbp_dict[R] = glbp

        # Gabungkan sesuai R_map
        glbp_img = np.zeros_like(image, dtype=np.uint8)

        for R, glbp in glbp_dict.items():
            glbp_img[R_map == R] = glbp[R_map == R]

        # MENDAPATKAN SKOR
        # Entropy G-LBP
        mask_ent, entropy_glbp = compute_entropy_map(glbp_img, win=ent_win, n_bins=n_bins, show_mask=True)
        # Mask kandidat MA: entropy tinggi + kontras tinggi
        ma_mask = mask_ent & (contrast > t_contrast)
        bg_mask = ~ma_mask

        # Hitung kontras MA vs background
        if np.sum(ma_mask) == 0 or np.sum(bg_mask) == 0:
            contrast_score = 0
        else:
            mu1 = np.mean(entropy_glbp[ma_mask])
            mu0 = np.mean(entropy_glbp[bg_mask])

            std1 = np.std(entropy_glbp[ma_mask])
            std0 = np.std(entropy_glbp[bg_mask])

            contrast_score = (mu1 - mu0) / (std1 + std0 + 1e-8)


        return glbp_img, contrast_score
    # =================================================
    # GRID SEARCH
    # =================================================
    if method_opt in ["grid", "hybrid"]:
        for k in k_list:
            glbp_img, score = compute_glbp_adaptive_radius(k_thresh=k)
            if score > best_score:
                best_score = score
                best_k = k
                glbp_best = glbp_img
                method_name = "grid"

    # =========================
    # Bayesian Optimization
    # =========================
    if method_opt in ["bayesian", "hybrid"]:
        # Fungsi objektif
        cache = {}
        def objective(k_val):
            k = k_val[0]
            if k in cache:
                return cache[k]
            
            _, score = compute_glbp_adaptive_radius(k_thresh=k)
            cache[k] = -score
            return cache[k]


        k_min, k_max = min(k_list), max(k_list)
        space = [Real(k_min, k_max, name="k")]
        res = gp_minimize(objective, space, n_calls=n_calls, x0=[[k_min], [k_max]], random_state=42)

        # Hasil terbaik Bayes
        bayes_k = res.x[0]
        bayes_score = -res.fun
        # dapakan glbp terbaik
        glbp_bayes, _ = compute_glbp_adaptive_radius(k_thresh=bayes_k)

        if method_opt == "bayesian" or bayes_score > best_score:
            best_score = bayes_score
            best_k = bayes_k
            glbp_best = glbp_bayes
            method_name = "bayesian"

    return glbp_best, best_k, best_score, method_name

# =====================================================================
# 7. Pipeline Ekstraksi Fitur GLBP (Untuk Green Channel)
# =====================================================================
def extract_glbp_adaptive(image_input,  
                          P=8, 
                          R_list=[1, 2, 3], 
                          k_list = [0.2, 0.3, 0.4, 0.5],
                          ent_win=3, n_bins=8,
                          method_opt="hybrid", method_glbp="default"):
    # Muat Gambar
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"File tidak ditemukan: {image_input}")
    else:
        img = image_input
    # Gunakan green channel (sesuai standar citra fundus)
    if img.ndim == 3:
        green_channel = img[:, :, 1]
    else:
        green_channel = img.astype(np.uint8)

    # --------------------------------------------------
    # Adaptive Radius
    # --------------------------------------------------
    R_map, contrast, entropy_map = adaptive_radius_map(
        img=green_channel, 
        R_list=[1, 2, 3],
        ent_win=ent_win, n_bins=n_bins
    )

    # --------------------------------------------------
    # Adaptive Threshold Factor (K)
    # --------------------------------------------------
    glbp_img, best_k, best_score, best_method = adaptive_threshold_factor(
        image=green_channel, 
        P=P, 
        R_map=R_map,
        k_list=k_list,
        ent_win=ent_win, n_bins=n_bins,
        method_opt=method_opt, method_glbp=method_glbp
    )

    # Histogram
    hist = glbp_histogram(glbp_img, P=P, method=method_glbp)
    
    return {
        "glbp_image": glbp_img,
        "entropy_map": entropy_map,
        "glbp_histogram": hist,
        "best_k": best_k,
        "best_score": best_score,
        "best_method": best_method
    }

import tkinter as tk
from tkinter import ttk, filedialog, font
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import json
import os
import pickle
import tensorflow as tf
from openpyxl import Workbook

class ImageViewerCanvas(tk.Canvas):
    def __init__(self, parent):

        super().__init__(parent, bg="black", highlightthickness=0)

        self.image = None
        self.img_tk = None

        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0

        self.pan_x = 0
        self.pan_y = 0

        self.display_w = 0
        self.display_h = 0

        self.start_x = 0
        self.start_y = 0

        self.bind("<Configure>", self.redraw)
        self.bind("<MouseWheel>", self.zoom_image)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan_image)
        self.bind("<Double-Button-1>", self.reset_view)

    # ==========================
    # LOAD IMAGE
    # ==========================
    def load_image(self, img):

        self.image = img
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.redraw()

    # ==========================
    # DRAW IMAGE
    # ==========================
    def redraw(self, event=None):

        if self.image is None:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_w, img_h = self.image.size

        scale = min(canvas_w / img_w, canvas_h / img_h) * self.zoom

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        img = self.image.resize((new_w, new_h), Image.LANCZOS)

        self.display_w = new_w
        self.display_h = new_h

        self.img_tk = ImageTk.PhotoImage(img)
        self.image_ref = self.img_tk

        self.delete("all")

        x = canvas_w // 2 + self.pan_x
        y = canvas_h // 2 + self.pan_y

        self.create_image(x, y, image=self.img_tk, anchor="center")

    # ==========================
    # ZOOM
    # ==========================
    def zoom_image(self, event):

        if self.image is None:
            return

        if event.delta > 0:
            new_zoom = self.zoom * 1.1
        else:
            new_zoom = self.zoom * 0.9

        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        if new_zoom != self.zoom:
            self.zoom = new_zoom
            self.redraw()

    # ==========================
    # PAN START
    # ==========================
    def start_pan(self, event):

        self.start_x = event.x
        self.start_y = event.y

    # ==========================
    # PAN MOVE
    # ==========================
    def pan_image(self, event):

        if self.image is None:
            return

        dx = event.x - self.start_x
        dy = event.y - self.start_y

        self.pan_x += dx
        self.pan_y += dy

        self.start_x = event.x
        self.start_y = event.y

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        max_x = max((self.display_w - canvas_w) // 2, 0)
        max_y = max((self.display_h - canvas_h) // 2, 0)

        self.pan_x = max(-max_x, min(self.pan_x, max_x))
        self.pan_y = max(-max_y, min(self.pan_y, max_y))

        self.redraw()

    # ==========================
    # RESET VIEW
    # ==========================
    def reset_view(self, event=None):

        if self.image is None:
            return

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.redraw()

    def clear(self):
        self.image = None
        self.img_tk = None
        self.delete("all")


class RetinaApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Sistem Klasifikasi Retina Diabetes")
        self.root.configure(bg="#f5f6fa")

        self.setup_style()

        self.load_parameters()

        self.create_title()
        self.create_main_layout()

    def load_parameters(self):
        self.result_img = []
        self.model = None
        self.model_info = None
        self.data_extract = None
        self.history_count = 0

        # Load Settings
        try:
            with open('settings.json', 'r') as f:
                self.dict_settings = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", "settings.json tidak ditemukan!")
            self.root.quit()
            return

        self.PATH_FOLDER = self.dict_settings['PATH_FOLDER']
        self.PATH_FILE = self.dict_settings['PATH_FILE']
        self.IMAGE_SIZE = self.dict_settings['IMAGE_SIZE']
        self.FEATURE_SELECTION = self.dict_settings['FEATURE_SELECTION']
        
    # =============================
    # STYLE
    # =============================
    def setup_style(self):
        style = ttk.Style()
        style.theme_use("default")

        # background utama
        style.configure("Custom.TFrame", background="#f5f6fa")
        style.configure("Custom.TLabelframe", background="#f5f6fa")
        style.configure("Custom.TLabelframe.Label", background="#f5f6fa")

    # =============================
    # TITLE
    # =============================
    def create_title(self):
        title = tk.Label(
            self.root,
            text="SISTEM KLASIFIKASI RETINA DIABETES",
            font=("Segoe UI",16,"bold"),
            bg="#f5f6fa"
        )

        title.pack(pady=10)

    # =============================
    # MAIN LAYOUT
    # =============================
    def create_main_layout(self):
        main_frame = tk.Frame(self.root,bg="#f5f6fa")
        main_frame.pack(fill="both",expand=True,padx=15)

        main_frame.columnconfigure(0,weight=1,minsize=350)
        main_frame.columnconfigure(1,weight=2)

        self.create_left_panel(main_frame)
        self.create_right_panel(main_frame)

        # self.create_bottom_buttons(container)

    # =============================
    # PANEL KIRI
    # =============================
    def create_left_panel(self,parent):

        left_panel = tk.Frame(parent,bg="#f5f6fa")
        left_panel.grid(row=0,column=0,sticky="nsew",padx=10)

        # MODEL PANEL
        frame_model = ttk.LabelFrame(left_panel, text="Model", style="Custom.TLabelframe")
        frame_model.pack(fill="x",pady=5)

        ttk.Button(
            frame_model,
            text="Upload Model",
            command=self.upload_model
        ).grid(row=0,column=0,padx=10,pady=10)

        self.frame_model_info = self.create_table(
            frame_model,
            ("Parameter","Nilai"),
            height=5
        )

        self.frame_model_info.grid(row=0,column=1,sticky="nsew")

        # PREVIEW
        frame_preview = ttk.LabelFrame(left_panel, text="Preview Gambar", style="Custom.TLabelframe")
        frame_preview.pack(fill="both",pady=10)

        self.btn_upload_img = ttk.Button(
            frame_preview,
            text="Upload Gambar",
            command=self.upload_image,
            state="disabled"
        )

        self.btn_upload_img.grid(row=0,column=0)

        self.preview = ImageViewerCanvas(frame_preview)
        self.preview.grid(row=1,column=0,sticky="nsew",padx=10,pady=10)

        # RESULT
        frame_result = ttk.LabelFrame(left_panel, text="Hasil Prediksi", style="Custom.TLabelframe")
        frame_result.pack(fill="x", pady=10)

        # Label hasil utama
        self.label_dr = ttk.Label(
            frame_result,
            text="Hasil : -",
            font=("Segoe UI", 12, "bold")
        )
        self.label_dr.pack(anchor="w", padx=10, pady=5)

        # =========================
        # Tabel Probabilitas
        # =========================
        columns = ("Kelas", "Probabilitas")

        self.table_result = ttk.Treeview(
            frame_result,
            columns=columns,
            show="headings",
            height=5
        )

        self.table_result.heading("Kelas", text="Kelas")
        self.table_result.heading("Probabilitas", text="Probabilitas (%)")

        self.table_result.column("Kelas", anchor="center", width=120)
        self.table_result.column("Probabilitas", anchor="center", width=120)

        self.table_result.pack(fill="x", padx=10, pady=5)

    # =============================
    # PANEL KANAN
    # =============================
    def create_right_panel(self,parent):

        right_panel = tk.Frame(parent,bg="#f5f6fa")
        right_panel.grid(row=0,column=1,sticky="nsew")

        # PREPROCESS GRID
        frame_grid = ttk.LabelFrame(right_panel, text="Hasil Preprocessing", style="Custom.TLabelframe")
        frame_grid.pack(fill="both",expand=True,pady=5)

        grid_container = tk.Frame(frame_grid)
        grid_container.pack(fill="both",expand=True,padx=5,pady=5)

        for i in range(4):
            grid_container.columnconfigure(i,weight=1)

        for i in range(2):
            grid_container.rowconfigure(i,weight=1)

        self.image_boxes = []
        self.image_titles = []

        for r in range(2):
            for c in range(4):

                cell = tk.Frame(grid_container)
                cell.grid(row=r,column=c,sticky="nsew",padx=4,pady=4)

                cell.rowconfigure(1,weight=1)

                title = tk.Label(
                    cell,
                    text="-",
                    font=("Segoe UI",9,"bold"),
                    bg="#f0f0f0"
                )

                title.grid(row=0,column=0,sticky="ew")

                canvas = ImageViewerCanvas(cell)
                canvas.grid(row=1,column=0,sticky="nsew")

                self.image_titles.append(title)
                self.image_boxes.append(canvas)

        # HISTORY
        frame_history = ttk.LabelFrame(right_panel, text="Riwayat Prediksi", style="Custom.TLabelframe")
        frame_history.pack(fill="both",expand=True,pady=10)

        self.tree = self.create_table(
            frame_history,
            ("No","Nama File","Model","Hasil","Confidence"),
            height=6
        )

        self.tree.pack(fill="both",expand=True, padx=5,pady=5)

        # Button
        self.create_bottom_buttons(frame_history)

    # =============================
    # BUTTONS
    # =============================
    def create_bottom_buttons(self, parent):
        frame_buttons = ttk.Frame(parent, style="Custom.TFrame")
        frame_buttons.pack(fill="x", pady=10)

        ttk.Button(frame_buttons, text="Simpan Hasil", width=18, command=self.save_to_excel).pack(side="left", padx=10)

        ttk.Button(frame_buttons, text="Reset", width=18, command=self.reset_app).pack(side="left", padx=10)

        ttk.Button(frame_buttons, text="Keluar", width=18, command=self.exit_app).pack(side="right", padx=10)

    # =============================
    # HELPER TABLE
    # =============================
    def create_table(self,parent,columns,height=5):

        tree = ttk.Treeview(parent,columns=columns,show="headings",height=height)

        for col in columns:

            tree.heading(col,text=col)

            if col == "Nama File":
                tree.column(col,width=200)
            else:
                tree.column(col,width=120)

        return tree

    def save_to_excel(self):
        # cek apakah ada data
        items = self.tree.get_children()
        if not items:
            messagebox.showwarning("Warning", "Belum ada data untuk disimpan!")
            return

        # pilih lokasi simpan
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel File", "*.xlsx")],
            title="Simpan ke Excel"
        )

        if not file_path:
            return

        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Hasil Prediksi"

            # =========================
            # HEADER
            # =========================
            columns = ("No","Nama File","Model","Hasil","Confidence")
            ws.append(columns)

            # =========================
            # DATA
            # =========================
            for item in items:
                values = self.tree.item(item, "values")
                ws.append(values)

            # =========================
            # AUTO WIDTH
            # =========================
            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter

                for cell in col:
                    try:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass

                ws.column_dimensions[col_letter].width = max_length + 2

            # =========================
            # SAVE
            # =========================
            wb.save(file_path)

            messagebox.showinfo("Sukses", "Data berhasil disimpan ke Excel!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def reset_app(self):
        if not messagebox.askyesno("Reset", "Yakin ingin reset semua data?"):
            return

        try:
            # =========================
            # RESET PREVIEW UTAMA
            # =========================
            self.preview.image = None
            self.preview.delete("all")

            # =========================
            # RESET GRID IMAGE
            # =========================
            self.preview.clear()
            for box in self.image_boxes:
                box.clear()

            for title in self.image_titles:
                title.config(text="-")

            # =========================
            # RESET HASIL
            # =========================
            self.label_dr.config(text="Hasil : -")

            for item in self.table_result.get_children():
                self.table_result.delete(item)

            # =========================
            # RESET DATA INTERNAL
            # =========================
            self.data_extract = None
            self.result_img = []
            self.original_img = None
            self.filename_img = None

            # =========================
            # RESET HISTORY
            # =========================
            for item in self.tree.get_children():
                self.tree.delete(item)

            messagebox.showinfo("Reset", "Aplikasi berhasil di-reset")

            # RESET NOMOR
            self.history_count = 0

        except Exception as e:
            messagebox.showerror("Error Reset", str(e))

    def exit_app(self):
        if messagebox.askyesno("Keluar", "Yakin ingin keluar aplikasi?"):
            self.root.quit()

    # =============================
    # Upload model
    # =============================
    def upload_model(self):
        try:
            if not messagebox.askyesno("Konfirmasi", "Upload model baru?"):
                return
            
            # Request Upload File
            file = filedialog.askopenfilename(
                title="Pilih Model",
                filetypes=[("Model Files", "*.pkl *.keras")]
            )

            if not file:
                return
            
            messagebox.showinfo("Sukses", "Model berhasil dimuat!")

            filename = file
            _, ext = os.path.splitext(os.path.basename(filename))

            model = None
            info = {}

            # =========================
            # LOAD MODEL
            # =========================
            if ext == ".pkl":
                model = pickle.load(open(filename, "rb"))

                info = {
                    "file": os.path.basename(filename),
                    "framework": "scikit-learn",
                    "model_type": type(model).__name__,
                }

                if hasattr(model, "n_features_in_"):
                    info["num_features"] = model.n_features_in_

                if hasattr(model, "classes_"):
                    info["classes"] = list(model.classes_)

                info["extension"] = ext

            elif ext == ".keras":
                model = tf.keras.models.load_model(filename)

                info = {
                    "file": os.path.basename(filename),
                    "framework": "Keras / TensorFlow",
                    "model_type": "Deep Learning",
                    "input_shape": model.input_shape,
                    "output_shape": model.output_shape,
                    "num_layers": len(model.layers),
                    "extension": ext
                }

            # =========================
            # MODEL TIDAK VALID
            # =========================
            if model is None:
                return

            # simpan ke class
            self.model = model
            self.model_info = info

            # hapus isi lama
            for item in self.frame_model_info.get_children():
                self.frame_model_info.delete(item)

            # isi tabel
            for k, v in info.items():
                self.frame_model_info.insert("", "end", values=(k, v))

            # Aktifkan button upload gambar
            self.btn_upload_img.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    # =============================
    # Upload gambar
    # =============================
    def upload_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Silakan upload model terlebih dahulu!")
            return

        file = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )

        if not file:
            return

        self.filename_img = file
        try:
            self.original_img = Image.open(file)
        except Exception as e:
            messagebox.showerror("Error", "Gambar tidak valid!")
            return

        def process():
            self.preview.load_image(self.original_img)
            # Preprocessing Image
            self.preprocessing()
            # Ekstraksi Fitur
            self.extract_features()
            # Prediksi Model
            self.model_predictions()

        self.run_with_loading(process, "Memproses gambar...")

    def preprocessing(self):
        # Urutan preprocessing citra retina
        params = [
            'green_channel',            # Ambil kanal hijau (kontras MA terbaik)
            'crop_retina',              # Hilangkan background, fokus area retina
            'illumination_correction',  # Koreksi pencahayaan tidak merata
            'bilateral',                # Denoise sambil jaga edge MA
            'vessel_remove',            # Hilangkan pembuluh darah
            'ma_enhance',               # Tonjolkan microaneurysm
            'clahe',                    # Tingkatkan kontras lokal (MA lebih jelas)
            f'resize_thumbnail_{self.IMAGE_SIZE[0]}x{self.IMAGE_SIZE[1]}',  # Samakan ukuran input
        ]

        try:
            img_pre = preprocessing(self.original_img, params)
            self.result_img = list(img_pre.values())[-1]
            self.show_preprocessing(img_pre)
        except Exception as e:
            messagebox.showerror("Error Preprocessing", str(e))

    def show_preprocessing(self, img_list):
        for i, (title, img) in enumerate(img_list.items()):

            if i >= len(self.image_boxes):
                break

            # set title
            self.image_titles[i].config(text=title)

            # numpy -> PIL
            if isinstance(img, np.ndarray):

                if len(img.shape) == 2:
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            else:
                pil_img = img

            self.image_boxes[i].load_image(pil_img)

    def extract_features(self):
        try:
            image_input = self.result_img if len(self.result_img) > 0 else None
            extract_img = extract_glbp_adaptive(image_input=image_input)
            
            self.data_extract = extract_img
        except Exception as e:
            messagebox.showerror("Error Feature", str(e))

    def features_selection(self):
        if self.data_extract is None:
            return None
        
        data = self.data_extract['glbp_histogram']
        data_select = {k: data[k] for k in self.FEATURE_SELECTION}

        return data_select

    def model_predictions(self):
        try:
            if self.model is None or self.model_info is None:
                messagebox.showwarning("Warning", "Model belum dimuat!")
                return
            model_ext = self.model_info['extension']

            # =========================
            # MODEL ML (.pkl)
            # =========================
            if model_ext == ".pkl":

                X = self.features_selection()
                if X is None:
                    return

                X = np.array(list(X.values())).reshape(1, -1)

                probs = self.model.predict_proba(X)[0]

            # =========================
            # MODEL CNN (.keras)
            # =========================
            elif model_ext == ".keras":

                X = self.convert_to_tensor()
                probs = self.model.predict(X)[0]

                # jika sigmoid
                if len(probs) == 1:
                    probs = [1 - probs[0], probs[0]]
            else:
                return

            classes = ["Non MA", "MA"]

            # =========================
            # FORMAT PROBABILITAS
            # =========================
            prob_text = []
            for cls, prob in zip(classes, probs):
                prob_text.append(f"{cls} : {prob*100:.2f} %")

            prob_text = "\n".join(prob_text)


            # =========================
            # PREDIKSI TERBESAR
            # =========================
            idx = np.argmax(probs)
            result_class = classes[idx]
            confidence = probs[idx] * 100


            # =========================
            # TAMPILKAN HASIL
            # =========================
            self.label_dr.config(text=f"Hasil : {result_class}")

            # hapus isi lama
            for item in self.table_result.get_children():
                self.table_result.delete(item)

            # isi tabel probabilitas
            for cls, prob in zip(classes, probs):

                self.table_result.insert(
                    "",
                    "end",
                    values=(cls, f"{prob*100:.2f}")
                )

            # =========================
            # SIMPAN KE RIWAYAT
            # =========================
            filename = self.filename_img
            model_name = self.model_info["file"]

            self.history_count += 1
            no = self.history_count

            self.tree.insert(
                "",
                "end",
                values=(
                    no,
                    filename,
                    model_name,
                    result_class,
                    f"{confidence:.2f} %"
                )
            )
        except Exception as e:
            messagebox.showerror("Error Prediksi", str(e))
        
    def convert_to_tensor(self):
        model_name = os.path.splitext(self.model_info['file'])[0].lower()
        image_size = self.IMAGE_SIZE

        # pilih preprocess_input
        if model_name.startswith("efficientnet"):
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif model_name.startswith("resnet"):
            from tensorflow.keras.applications.resnet import preprocess_input
        elif model_name.startswith("mobilenet_v3"):
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        else:
            raise ValueError(f"{model_name} tidak dikenal")

        img = self.data_extract['glbp_image']
        # resize
        img = cv2.resize(img, image_size)

        # convert ke array
        img = np.array(img)

        # 
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)


        # tambah batch dimension
        img = np.expand_dims(img, axis=0)

        # preprocess
        img = preprocess_input(img)

        return img
    
    def show_loading(self, text="Processing..."):
        self.loading_win = tk.Toplevel(self.root)
        self.loading_win.title("Loading")
        self.loading_win.geometry("300x120")
        self.loading_win.resizable(False, False)
        self.loading_win.grab_set()

        tk.Label(self.loading_win, text=text, font=("Segoe UI", 10)).pack(pady=10)

        self.progress = ttk.Progressbar(
            self.loading_win,
            orient="horizontal",
            length=200,
            mode="indeterminate"
        )
        self.progress.pack(pady=10)
        self.progress.start(10)


    def hide_loading(self):
        if hasattr(self, "loading_win"):
            self.progress.stop()
            self.loading_win.destroy()

    def run_with_loading(self, func, text="Processing..."):
        def task():
            try:
                func()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.hide_loading)

        self.show_loading(text)
        threading.Thread(target=task, daemon=True).start()

# =============================
# ERROR HANDLER GLOBAL
# =============================
import sys
import traceback

def global_exception_handler(exc_type, exc_value, exc_traceback):

    error_msg = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    print(error_msg)  # tetap tampil di console

    messagebox.showerror(
        "Terjadi Error",
        f"{exc_value}\n\nLihat console untuk detail."
    )

if __name__ == "__main__":
    # aktifkan handler
    sys.excepthook = global_exception_handler

    # =============================
    # RUN APP
    # =============================
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+0+0")

    app = RetinaApp(root)
    root.mainloop()