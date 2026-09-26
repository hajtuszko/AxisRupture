# Jodelka: wykrywanie ukladu skosnych desek (herringbone) z map tekstury.
# Laczona mapa szczelin (AO, Cavity, Specular, Roughness, Displacement, Normal), siec jodelki
# na drobnej siatce (szerokosc i dlugosc deski jako wielokrotnosci jednostki) dopasowana do szczelin.
import bpy, math, os, re

MAP_KEYS = ("AO", "Cavity", "Specular", "Roughness", "Displacement", "Normal", "BaseColor", "Gloss", "Bump")


def _box(a, rad):
    import numpy as np
    k = 2 * rad + 1
    c = np.cumsum(np.cumsum(np.pad(a, ((rad + 1, rad), (rad + 1, rad)), mode='wrap'), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _pixels(img, res):
    """RGBA przeskalowanego obrazu (na kopii - obraz uzytkownika bez zmian)"""
    import numpy as np
    im = img.copy()
    try:
        if im.size[0] != res or im.size[1] != res:
            im.scale(res, res)
        px = np.empty(res * res * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(im)
    return px.reshape(res, res, 4)


def sibling_maps(img):
    """inne mapy tego samego materialu (ten sam katalog i nazwa, inny rodzaj mapy)"""
    path = bpy.path.abspath(img.filepath)
    folder, name = os.path.split(path)
    found = {}
    for key in MAP_KEYS:
        m = re.search(key, name, re.IGNORECASE)
        if m:
            for other in MAP_KEYS:
                cand = os.path.join(folder, name[:m.start()] + other + name[m.end():])
                if os.path.exists(cand):
                    found[other] = cand
            break
    if not found:
        found["?"] = path
    return found


def seam_map(img, res=1024):
    """laczona mapa szczelin: kazda mapa daje swoj sygnal szczeliny, znormalizowany i usredniony"""
    import numpy as np
    maps = sibling_maps(img)
    ev, used = [], []

    def z(a):
        med = np.median(a); mad = np.median(np.abs(a - med)) + 1e-9
        return np.minimum(np.clip((a - med) / (1.4826 * mad), 0, None), 8.0)

    for key, path in maps.items():
        try:
            im = bpy.data.images.load(path, check_existing=True)
        except RuntimeError:
            continue
        px = _pixels(im, res)
        L = 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]
        if key in ("AO", "Cavity", "Specular", "?"):          # szczelina = ciemna linia
            ev.append(z(np.maximum(_box(L, 3) - L, 0)))
        elif key in ("Roughness",):                           # szczelina = jasna (szorstka) linia
            ev.append(z(np.maximum(L - _box(L, 3), 0)))
        elif key in ("Gloss",):
            ev.append(z(np.maximum(_box(L, 3) - L, 0)))
        elif key in ("Displacement", "Bump"):                 # szczelina = prog wysokosci miedzy deskami
            g = np.hypot(np.roll(L, -1, 1) - np.roll(L, 1, 1), np.roll(L, -1, 0) - np.roll(L, 1, 0))
            ev.append(z(np.maximum(g - _box(g, 4), 0)))
        elif key == "Normal":                                 # szczelina = mocne pochylenie normalnej
            t = np.hypot(px[..., 0] - np.median(px[..., 0]), px[..., 1] - np.median(px[..., 1]))
            ev.append(z(np.maximum(t - _box(t, 3), 0)))
        else:
            continue
        used.append(key)
    S = np.mean(ev, 0) if ev else None
    return S, used


def _comb(T, s, per):
    import numpy as np
    b = np.floor((T % per) / per * 48).astype(np.int64)
    prof = np.bincount(b, weights=s, minlength=48) / np.maximum(np.bincount(b, minlength=48), 1)
    k = int(np.argmax(prof))
    return np.sort(prof)[-2:].mean() - np.median(prof), (k + 0.5) / 48 * per


def detect_lattice(S):
    """Siatka desek z widma (FFT) mapy szczelin. Kafel jest bezszwowy, wiec siec ma calkowite wektory
    czestotliwosci: dwa najsilniejsze szczyty = linie bokow desek w dwoch kierunkach (okres = szerokosc
    deski), faza szczytu = polozenie linii. Zwraca (N1, N2, okres, faza1, faza2, k=1, n_sugerowane)."""
    import numpy as np
    h, w = S.shape
    F = np.fft.fft2(S - S.mean())
    P = np.abs(F) ** 2
    KX, KY = np.meshgrid(np.fft.fftfreq(w) * w, np.fft.fftfreq(h) * h)
    R = np.hypot(KX / w, KY / h)                        # cykle na piksel
    lo, hi = 15.0, min(w, h) / 5                        # szerokosc deski: 15 px .. 1/5 kafla
    band = (R > 1 / hi) & (R < 1 / lo) & ((KY > 0) | ((KY == 0) & (KX > 0)))
    order = np.argsort(-np.where(band, P, 0), axis=None)
    peaks = []
    for flat in order[:400]:
        y, x = np.unravel_index(flat, P.shape)
        kx, ky = KX[y, x], KY[y, x]
        ang = math.degrees(math.atan2(ky / h, kx / w)) % 180
        if not peaks:
            peaks.append((kx, ky, ang, y, x))
        elif all(min(abs(ang - q[2]), 180 - abs(ang - q[2])) > 20 for q in peaks):
            peaks.append((kx, ky, ang, y, x))
        if len(peaks) == 2:
            break
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for kx, ky, ang, y, x in peaks:
        fvec = np.array([kx / w, ky / h])               # cykle na piksel
        per = 1.0 / np.linalg.norm(fvec)
        N = fvec * per                                   # jednostkowa normalna linii
        # faza: profil mapy zlozony modulo okres; szczelina = maksimum profilu. (Faza samego szczytu FFT
        # bywa przesunieta o pol okresu, gdy szczelina ma dwie krawedzie - rowek w mapie wysokosci.)
        t = ((X * N[0] + Y * N[1]) / per % 1.0).ravel()
        b = np.minimum((t * 48).astype(np.int64), 47)
        prof = np.bincount(b, weights=S.ravel(), minlength=48) / np.maximum(np.bincount(b, minlength=48), 1)
        prof = (np.roll(prof, 1) + prof + np.roll(prof, -1)) / 3
        out.append((N, per, (int(np.argmax(prof)) + 0.5) / 48 * per))
    (N1, per1, p1), (N2, per2, p2) = out
    per = (per1 + per2) / 2
    # dlugosc deski: powtarzanie wzoru wzdluz kierunku kratek (1,-1) co n kratek
    Minv = np.linalg.inv(np.array([N1, N2]))
    vb = Minv @ np.array([per, -per])                   # przesuniecie o (1,-1) kratke w pikselach
    best_n, best_v = 0, -1.0
    for n in range(2, 17):
        v = 0.0
        for mlt in (1, 2, 3):
            f = vb / (np.dot(vb, vb) * n) * mlt          # czestotliwosc okresu n*vb
            ix = int(round(f[0] * w)) % w; iy = int(round(f[1] * h)) % h
            v += P[iy, ix]
        if v > best_v:
            best_n, best_v = n, v
    return (N1, N2, per, p1, p2, 1), best_n


def edge_scores(S, lat, samples=5):
    """sila szczeliny na kazdym boku kratki jednostkowej: E1[i,j] - bok na linii t1=i (miedzy (i-1,j) a (i,j)),
    E2[i,j] - bok na linii t2=j; zwraca tez zakresy indeksow"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    h, w = S.shape
    m = S.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                m = np.maximum(m, np.roll(np.roll(S, dy, 0), dx, 1))
    Minv = np.linalg.inv(np.array([N1, N2]))
    cor = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    t1 = cor @ N1; t2 = cor @ N2
    i0, i1 = int(math.floor((t1.min() - p1) / u)), int(math.ceil((t1.max() - p1) / u))
    j0, j1 = int(math.floor((t2.min() - p2) / u)), int(math.ceil((t2.max() - p2) / u))
    I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    fs = np.linspace(0.2, 0.8, samples)

    def sample(A, B):                                  # A,B: tablice t1,t2 (wiele probek)
        PX = Minv[0, 0] * A + Minv[0, 1] * B; PY = Minv[1, 0] * A + Minv[1, 1] * B
        return m[np.round(PY).astype(int) % h, np.round(PX).astype(int) % w]
    # mediana probek: szczelina musi biec wzdluz calego boku (pojedyncza plamka/rysa nie wystarcza)
    E1 = np.median([sample(p1 + I * u, p2 + (J + f) * u) for f in fs], 0)
    E2 = np.median([sample(p1 + (I + f) * u, p2 + J * u) for f in fs], 0)
    return E1, E2, (i0, j0)


def _ids(X, Y, k, n, cx, cy):
    """identyfikator deski jodelki (siec: (k,k), (n,-n); para: H [0,n)x[0,k), V [n,n+k)x[k-n,k))
    dla srodkow kratek X+0.5, Y+0.5 (identyfikatory >= 0)"""
    import numpy as np
    qx = X + 0.5 - cx; qy = Y + 0.5 - cy
    fa = np.floor((qx + qy) / (2 * k)).astype(np.int64)
    fb = np.floor((qx - qy) / (2 * n)).astype(np.int64)
    out = np.zeros(X.shape, dtype=np.int64)
    done = np.zeros(X.shape, dtype=bool)
    ra = n // k + 2
    for da in range(-ra, 2):
        for db in (-2, -1, 0, 1):
            a = fa + da; b = fb + db
            rx = qx - a * k - b * n; ry = qy - a * k + b * n
            H = (rx >= 0) & (rx < n) & (ry >= 0) & (ry < k) & ~done
            V = (rx >= n) & (rx < n + k) & (ry >= k - n) & (ry < k) & ~done
            key = ((a + (1 << 20)) * (1 << 21) + (b + (1 << 20))) * 2
            out = np.where(H, key, np.where(V, key + 1, out))
            done |= H | V
    return out


VARIANTS = ((0, 1), (0, -1), (1, 1), (1, -1))


def _var_xy(I, J, var):
    swap, sig = var
    X, Y = (J, I) if swap else (I, J)
    return X, sig * Y


def fit(E1, E2, origin, k, n_user=0, var_user=None, off_user=None):
    """najlepsza siec jodelki: (wynik, n, (cx, cy), wariant)"""
    import numpy as np
    i0, j0 = origin
    I, J = np.meshgrid(np.arange(E1.shape[0]) + i0, np.arange(E1.shape[1]) + j0, indexing='ij')
    best = None
    ns = [n_user] if n_user else range(k + 2, 12 * k + 1)
    for var in ([var_user] if var_user else VARIANTS):
        X, Y = _var_xy(I, J, var)
        for n in ns:
            offs = [off_user] if off_user else [(cx, cy) for cx in range(2 * n) for cy in range(2 * k)]
            for cx, cy in offs:
                B = _ids(X, Y, k, n, cx, cy)
                in1 = B[1:, :] == B[:-1, :]            # bok E1[i] miedzy (i-1) a i
                in2 = B[:, 1:] == B[:, :-1]
                e1 = E1[1:, :]; e2 = E2[:, 1:]
                inner = np.concatenate([e1[in1], e2[in2]]); seam = np.concatenate([e1[~in1], e2[~in2]])
                if inner.size == 0 or seam.size == 0:
                    continue
                # wynik rangowy (odporny na pojedyncze mocne linie): ile bokow wewnetrznych jest slabszych
                # od typowej szczeliny + ile szczelin mocniejszych od typowego boku wewnetrznego (maks. 2)
                sc = float((inner < np.median(seam)).mean() + (seam > np.median(inner)).mean())
                if best is None or sc > best[0]:
                    best = (sc, n, (cx, cy), var)
    return best


def layout(lat, model, uv_min, uv_max, size):
    """deski na obszarze UV: lista (i0, i1, j0, j1 w kratkach jednostkowych, dlugosc wzdluz i?)"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    _, n, (cx, cy), var = model
    w, h = size
    cor = np.array([[uv_min[0] * w, uv_min[1] * h], [uv_max[0] * w, uv_min[1] * h],
                    [uv_min[0] * w, uv_max[1] * h], [uv_max[0] * w, uv_max[1] * h]])
    t1 = cor @ N1; t2 = cor @ N2
    i0, i1 = int(math.floor((t1.min() - p1) / u)) - 2, int(math.ceil((t1.max() - p1) / u)) + 2
    j0, j1 = int(math.floor((t2.min() - p2) / u)) - 2, int(math.ceil((t2.max() - p2) / u)) + 2
    I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    X, Y = _var_xy(I, J, var)
    B = _ids(X, Y, k, n, cx, cy)
    boards = {}
    for bid in np.unique(B):
        if bid < 0:
            continue
        m = B == bid
        ii = I[m]; jj = J[m]
        boards[int(bid)] = (int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1)
    return boards


def cell_quad(lat, size, i0, i1, j0, j1, shrink):
    """czworokat UV (CCW) prostokata kratek, pomniejszony o shrink (w jednostkach) z kazdej strony"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    w, h = size
    Minv = np.linalg.inv(np.array([N1, N2]))
    pts = []
    for a, b in ((i0 + shrink, j0 + shrink), (i1 - shrink, j0 + shrink),
                 (i1 - shrink, j1 - shrink), (i0 + shrink, j1 - shrink)):
        P = Minv @ np.array([p1 + a * u, p2 + b * u])
        pts.append((float(P[0] / w), float(P[1] / h)))
    area = sum(pts[q - 1][0] * pts[q][1] - pts[q][0] * pts[q - 1][1] for q in range(4))
    return pts if area > 0 else pts[::-1]


def along_uv(lat, size, i0, i1, j0, j1):
    """kierunek dlugiej osi deski w UV (na jedna jednostke)"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    Minv = np.linalg.inv(np.array([N1, N2]))
    d = Minv @ (np.array([u, 0.0]) if (i1 - i0) >= (j1 - j0) else np.array([0.0, u]))
    return float(d[0] / size[0]), float(d[1] / size[1])
