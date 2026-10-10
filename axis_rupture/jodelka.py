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


def _valley(A, sigmas, weights=None):
    """sila 'doliny' (ciemnej/niskiej linii) w obrazie: maks. po skalach z sigma^2 * LoG (liczone przez FFT -
    kafel bezszwowy, wiec splot okresowy jest dokladny). Dodatnie w srodku linii, niezaleznie od jej szerokosci."""
    import numpy as np
    h, w = A.shape
    FA = np.fft.rfft2(A - A.mean())
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    r2 = fx * fx + fy * fy
    out = None
    for s in sigmas:
        Gk = np.exp(-2 * math.pi ** 2 * s * s * r2)
        lap = np.fft.irfft2(FA * (-4 * math.pi ** 2 * r2) * Gk, s=(h, w)) * s * s   # sigma^2 * LoG
        out = lap if out is None else np.maximum(out, lap)
    return out


def _normal_valley(px, sigmas):
    """dolina wysokosci z mapy normalnych: laplasjan wysokosci = -dywergencja (nx, ny); konwencja osi Y
    (OpenGL / DirectX) nieznana - zwraca obie wersje"""
    import numpy as np
    nx = px[..., 0] * 2 - 1; ny = px[..., 1] * 2 - 1
    h, w = nx.shape
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    r2 = fx * fx + fy * fy
    Fx = np.fft.rfft2(nx - nx.mean()); Fy = np.fft.rfft2(ny - ny.mean())
    outs = []
    for sy in (1, -1):
        out = None
        for s in sigmas:
            Gk = np.exp(-2 * math.pi ** 2 * s * s * r2)
            div = np.fft.irfft2((2j * math.pi * fx * Fx + sy * 2j * math.pi * fy * Fy) * Gk, s=(h, w)) * s
            v = -div                                           # dolina: laplasjan wysokosci > 0
            out = v if out is None else np.maximum(out, v)
        outs.append(out)
    return outs


def run(gen):
    """wykonaj generator do konca i zwroc jego wynik (wersja bez paska postepu)"""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def seam_map(img, res=1024):
    return run(seam_map_iter(img, res))


def seam_map_iter(img, res=1024):
    """Laczona mapa szczelin: kazda mapa materialu zamieniona na miare 'doliny' w srodku szczeliny
    (AO/Cavity/Specular/Gloss/wysokosc/kolor - ciemniej/nizej, Roughness - jasniej, Normal - wklesla
    krzywizna), wieloskalowo, odpornie znormalizowana i usredniona. Szczelina = jeden ostry grzbiet
    na srodku, niezaleznie od rozdzielczosci, szerokosci szczeliny i rodzaju map."""
    import numpy as np
    maps = sibling_maps(img)
    base = res / 1024
    sig = [s * base for s in (0.7, 1.4, 2.8, 5.6)]
    ev, used, wts = [], [], []

    def z(a):
        med = np.median(a); mad = np.median(np.abs(a - med)) + 1e-9
        return np.minimum(np.clip((a - med) / (1.4826 * mad), 0, None), 8.0)

    normal_px = None
    n_maps = len(maps)
    for i_map, (key, path) in enumerate(maps.items()):
        yield (i_map / n_maps, f"map {i_map + 1}/{n_maps}: {key}")
        try:
            im = bpy.data.images.load(path, check_existing=True)
        except RuntimeError:
            continue
        px = _pixels(im, res).astype(np.float64)
        L = 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]
        if key in ("AO", "Cavity", "Specular", "Gloss", "Displacement", "Bump", "?"):
            ev.append(z(_valley(L, sig))); wts.append(1.0)
        elif key == "Roughness":
            ev.append(z(_valley(-L, sig))); wts.append(1.0)
        elif key == "BaseColor":                              # kolor: odcienie desek myla - mniejsza waga
            ev.append(z(_valley(L, sig))); wts.append(0.5)
        elif key == "Normal":
            normal_px = px; continue
        else:
            continue
        used.append(key)
    if normal_px is not None:
        yield (0.95, "normal map")
        a, b = _normal_valley(normal_px, sig)
        if ev:                                                # konwencja Y: zgodna z pozostalymi mapami
            ref = np.average(ev, 0, weights=wts)
            pick = a if np.corrcoef(a.ravel()[::7], ref.ravel()[::7])[0, 1] >= \
                np.corrcoef(b.ravel()[::7], ref.ravel()[::7])[0, 1] else b
        else:                                                 # sama mapa normalnych: ostrzejsze linie
            pick = a if np.percentile(a, 99) - np.median(a) >= np.percentile(b, 99) - np.median(b) else b
        ev.append(z(pick)); wts.append(1.0); used.append("Normal")
    S = np.average(ev, 0, weights=wts).astype(np.float32) if ev else None
    return S, used


def _comb(T, s, per):
    import numpy as np
    b = np.floor((T % per) / per * 48).astype(np.int64)
    prof = np.bincount(b, weights=s, minlength=48) / np.maximum(np.bincount(b, minlength=48), 1)
    k = int(np.argmax(prof))
    return np.sort(prof)[-2:].mean() - np.median(prof), (k + 0.5) / 48 * per


def _fold(S, X, Y, f, bins=48):
    """profil mapy zlozony wzdluz wektora czestotliwosci f (cykle/piksel): (ostrosc szczytu, faza 0..1)"""
    import numpy as np
    t = (X * f[0] + Y * f[1]) % 1.0
    b = np.minimum((t * bins).astype(np.int64), bins - 1)
    prof = np.bincount(b, weights=S, minlength=bins) / np.maximum(np.bincount(b, minlength=bins), 1)
    prof = (np.roll(prof, 1) + prof + np.roll(prof, -1)) / 3
    k = int(np.argmax(prof))
    return float(prof[k] - np.median(prof)), (k + 0.5) / bins


def detect_lattice(S):
    """Siatka desek z mapy szczelin - niezalezna od rozdzielczosci i rodzaju tekstury.
    Kafel jest bezszwowy, wiec linie szczelin maja calkowite wektory czestotliwosci. Kandydaci = najsilniejsze
    szczyty widma + ich podwielokrotnosci (harmoniczne rowka / slojow bywaja silniejsze niz podstawowa).
    Ocena kandydata = ostrosc szczytu profilu mapy zlozonej modulo okres: prawdziwe szczeliny daja jeden
    ostry szczyt, sloje / szum - rozmyty. Dwa kierunki > 20 st. od siebie; kazdy z wlasnym odstepem
    (znieksztalcone zdjecie), sprowadzonym do wspolnej jednostki przez skalowanie normalnej.
    Zwraca ((N1, N2, u, faza1, faza2, k=1), None)."""
    import numpy as np
    h, w = S.shape
    F = np.fft.rfft2(S - S.mean())
    P = np.abs(F) ** 2
    KY = (np.fft.fftfreq(h) * h)[:, None] * np.ones((1, P.shape[1]))
    KX = np.arange(P.shape[1])[None, :] * np.ones((h, 1))
    R = np.hypot(KX / w, KY / h)
    lo, hi = 6.0, min(w, h) / 3                         # szerokosc deski: 6 px .. 1/3 kafla
    band = (R > 1 / hi) & (R < 1 / lo) & ((KX > 0) | (KY > 0))
    order = np.argsort(-np.where(band, P, 0), axis=None)[:40]
    cands = set()
    for flat in order:
        y, x = np.unravel_index(flat, P.shape)
        kx, ky = int(KX[y, x]), int(KY[y, x])
        for m in (1, 2, 3, 4, 5, 6):
            c = (int(round(kx / m)), int(round(ky / m)))
            r = math.hypot(c[0] / w, c[1] / h)
            if c != (0, 0) and 1 / hi < r < 1 / lo:
                cands.add(c)
    # podprobkowanie pikseli (ok. 0.5 mln) - szybka ocena wielu kandydatow
    st = max(1, int(round(math.sqrt(h * w / 5e5))))
    Y, X = np.mgrid[0:h:st, 0:w:st].astype(np.float64)
    Ss = S[::st, ::st]
    Ss = (Ss - Ss.mean()) / (Ss.std() + 1e-9)
    X = X.ravel(); Y = Y.ravel(); Ss = Ss.ravel()
    scored = []
    for kx, ky in cands:
        f = (kx / w, ky / h)
        sharp, ph = _fold(Ss, X, Y, f)
        scored.append((sharp, kx, ky, ph))
    scored.sort(reverse=True)

    def ang(c):
        return math.degrees(math.atan2(c[2] / h, c[1] / w)) % 180

    def per(c):
        return 1.0 / math.hypot(c[1] / w, c[2] / h)

    def pick(pool):
        best = pool[0]
        # wielokrotnosc prawdziwego okresu ma podobna ostrosc - wybierz najmniejszy okres o ostrosci >= 85%
        for c in pool:
            if abs(ang(c) - ang(best)) % 180 < 2 or abs(ang(c) - ang(best)) % 180 > 178:
                if c[0] >= 0.85 * best[0] and per(c) < per(best) - 1e-6:
                    ratio = per(best) / per(c)
                    if abs(ratio - round(ratio)) < 0.08:
                        best = c
        return best

    c1 = pick(scored)
    rest = [c for c in scored if 20 < abs(ang(c) - ang(c1)) % 180 < 160]
    c2 = pick(rest)
    out = []
    for sharp, kx, ky, _ in (c1, c2):
        f = np.array([kx / w, ky / h])
        # faza na pelnej mapie (dokladniej)
        Yf, Xf = np.mgrid[0:h, 0:w].astype(np.float64)
        _, ph = _fold(S.ravel(), Xf.ravel(), Yf.ravel(), f, bins=96)
        out.append((f, ph))
    (f1, ph1), (f2, ph2) = out
    u = (1 / np.linalg.norm(f1) + 1 / np.linalg.norm(f2)) / 2   # wspolna jednostka (px)
    N1 = f1 * u; N2 = f2 * u                                   # x.N / u = liczba okresow
    return (N1, N2, u, ph1 * u, ph2 * u, 1), None


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
    return run(fit_iter(E1, E2, origin, k, n_user, var_user, off_user))


def fit_iter(E1, E2, origin, k, n_user=0, var_user=None, off_user=None):
    """najlepsza siec jodelki: (wynik, n, (cx, cy), wariant)"""
    import numpy as np
    i0, j0 = origin
    I, J = np.meshgrid(np.arange(E1.shape[0]) + i0, np.arange(E1.shape[1]) + j0, indexing='ij')
    best = None
    ns = [n_user] if n_user else range(k + 2, 12 * k + 1)
    vs = [var_user] if var_user else VARIANTS
    for i_var, var in enumerate(vs):
        yield (i_var / len(vs), f"pattern fit {i_var + 1}/{len(vs)}")
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


def cell_quad(lat, size, i0, i1, j0, j1, shrink, field=None):
    """czworokat UV (CCW) prostokata kratek, pomniejszony o shrink (w jednostkach) z kazdej strony"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    w, h = size
    Minv = np.linalg.inv(np.array([N1, N2]))
    pts = []
    for a, b in ((i0 + shrink, j0 + shrink), (i1 - shrink, j0 + shrink),
                 (i1 - shrink, j1 - shrink), (i0 + shrink, j1 - shrink)):
        P = Minv @ np.array([p1 + a * u, p2 + b * u])
        if field is not None:
            P = P + field_at(field, P[0:1], P[1:2])[:, 0]
        pts.append((float(P[0] / w), float(P[1] / h)))
    area = sum(pts[q - 1][0] * pts[q][1] - pts[q][0] * pts[q - 1][1] for q in range(4))
    return pts if area > 0 else pts[::-1]


def cell_poly(lat, size, i0, i1, j0, j1, shrink, field=None, raw=False):
    """obrys deski (CCW, UV) z punktem w kazdym wezle siatki na bokach; kazdy punkt przesuniety tym samym
    polem korekty -> sasiednie deski maja wspolna (odsunieta o szczeline) linie szczeliny, bez nachodzenia"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    w, h = size
    Minv = np.linalg.inv(np.array([N1, N2]))
    a0, a1, b0, b1 = i0 + shrink, i1 - shrink, j0 + shrink, j1 - shrink

    def ticks(lo, hi, c0, c1):
        inner = [float(t) for t in range(c0 + 1, c1)]
        return [lo] + inner + [hi]
    ta = ticks(a0, a1, i0, i1); tb = ticks(b0, b1, j0, j1)
    pts = [(a, b0) for a in ta] + [(a1, b) for b in tb[1:]] + [(a, b1) for a in ta[-2::-1]] + [(a0, b) for b in tb[-2:0:-1]]
    A = np.array([p1 + a * u for a, _ in pts]); B = np.array([p2 + b * u for _, b in pts])
    X = Minv[0, 0] * A + Minv[0, 1] * B; Y = Minv[1, 0] * A + Minv[1, 1] * B
    if field is not None:
        Dp = field_at(field, X, Y); X = X + Dp[0]; Y = Y + Dp[1]
    out = [(float(x / w), float(y / h)) for x, y in zip(X, Y)]
    if raw:
        return out, len(ta), len(tb)
    area = sum(out[q - 1][0] * out[q][1] - out[q][0] * out[q - 1][1] for q in range(len(out)))
    return out if area > 0 else out[::-1]


def along_uv(lat, size, i0, i1, j0, j1, field=None):
    """kierunek dlugiej osi deski w UV (z uwzglednieniem pola korekty)"""
    import numpy as np
    q = cell_quad(lat, size, i0, i1, j0, j1, 0.0, field)
    a = np.subtract(q[1], q[0]); b = np.subtract(q[3], q[0])
    la = math.hypot(a[0] * size[0], a[1] * size[1]); lb = math.hypot(b[0] * size[0], b[1] * size[1])
    d = a if la >= lb else b
    return float(d[0]), float(d[1])


# ---------------------------------------------------------------- lokalna kalibracja
# Tekstura to zdjecie - deski nie leza idealnie na siatce. Globalna siec daje punkt startowy;
# potem kazdy odcinek szczeliny (bok kratki) jest mierzony lokalnie (przesuniecie w poprzek, z pewnoscia),
# a z pewnych pomiarow liczone jest gladkie pole korekty (okresowe - kafel bezszwowy), ktore
# interpoluje poprawke tam, gdzie szczelina jest slabo widoczna.

def _bilinear(A, x, y):
    import numpy as np
    h, w = A.shape
    x0 = np.floor(x); y0 = np.floor(y)
    fx = x - x0; fy = y - y0
    x0 = x0.astype(np.int64) % w; y0 = y0.astype(np.int64) % h
    x1 = (x0 + 1) % w; y1 = (y0 + 1) % h
    return (A[y0, x0] * (1 - fx) * (1 - fy) + A[y0, x1] * fx * (1 - fy)
            + A[y1, x0] * (1 - fx) * fy + A[y1, x1] * fx * fy)


def lattice_points(lat, a, b):
    """piksele wezlow siatki (a, b) - tablice"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    Minv = np.linalg.inv(np.array([N1, N2]))
    A = p1 + np.asarray(a, float) * u; B = p2 + np.asarray(b, float) * u
    return np.stack([Minv[0, 0] * A + Minv[0, 1] * B, Minv[1, 0] * A + Minv[1, 1] * B])


def seam_edges(lat, model, size):
    """jednostkowe odcinki szczelin wzoru (srodek w obrebie kafla): (P0, P1) - tablice 2 x m"""
    import numpy as np
    N1, N2, u, p1, p2, k = lat
    _, n, (cx, cy), var = model
    w, h = size
    cor = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    t1 = cor @ N1; t2 = cor @ N2
    i0, i1 = int(math.floor((t1.min() - p1) / u)) - 2, int(math.ceil((t1.max() - p1) / u)) + 2
    j0, j1 = int(math.floor((t2.min() - p2) / u)) - 2, int(math.ceil((t2.max() - p2) / u)) + 2
    I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    X, Y = _var_xy(I, J, var)
    B = _ids(X, Y, k, n, cx, cy)
    a0, b0, a1, b1 = [], [], [], []
    s1 = B[1:, :] != B[:-1, :]                              # linia t1 = i: od (i, j) do (i, j+1)
    a0.append(I[1:, :][s1]); b0.append(J[1:, :][s1]); a1.append(I[1:, :][s1]); b1.append(J[1:, :][s1] + 1)
    s2 = B[:, 1:] != B[:, :-1]                              # linia t2 = j: od (i, j) do (i+1, j)
    a0.append(I[:, 1:][s2]); b0.append(J[:, 1:][s2]); a1.append(I[:, 1:][s2] + 1); b1.append(J[:, 1:][s2])
    P0 = lattice_points(lat, np.concatenate(a0), np.concatenate(b0))
    P1 = lattice_points(lat, np.concatenate(a1), np.concatenate(b1))
    M = (P0 + P1) / 2
    keep = (M[0] >= 0) & (M[0] < w) & (M[1] >= 0) & (M[1] < h)
    return P0[:, keep], P1[:, keep]


def measure_edges(S, P0, P1, reach, step=0.25, ns=16):
    """przesuniecie szczeliny w poprzek kazdego odcinka (px) i pewnosc pomiaru"""
    import numpy as np
    t = P1 - P0
    L = np.hypot(t[0], t[1]); t = t / L
    nrm = np.stack([-t[1], t[0]])
    s = np.linspace(0.15, 0.85, ns)
    offs = np.arange(-reach, reach + step / 2, step)
    X = (P0[0][:, None, None] + t[0][:, None, None] * (s[None, :, None] * L[:, None, None])
         + nrm[0][:, None, None] * offs[None, None, :])
    Y = (P0[1][:, None, None] + t[1][:, None, None] * (s[None, :, None] * L[:, None, None])
         + nrm[1][:, None, None] * offs[None, None, :])
    prof = np.median(_bilinear(S, X, Y), axis=1)           # mediana wzdluz odcinka: szczelina na calej dlugosci
    k = np.argmax(prof, 1)
    m = np.arange(len(k))
    kk = np.clip(k, 1, len(offs) - 2)
    a, b, c = prof[m, kk - 1], prof[m, kk], prof[m, kk + 1]
    den = a - 2 * b + c
    sub = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    d = offs[kk] + np.clip(sub, -1, 1) * step
    med = np.median(prof, 1)
    mad = np.median(np.abs(prof - med[:, None]), 1) * 1.4826 + 1e-6
    conf = (prof[m, k] - med) / mad
    conf[(k == 0) | (k == len(offs) - 1)] = 0.0             # szczyt na brzegu zakresu - niepewny
    return d, conf, nrm, (P0 + P1) / 2


def _grid_weights(G, size, x, y):
    import numpy as np
    w, h = size
    gx = np.asarray(x) / w * G; gy = np.asarray(y) / h * G
    x0 = np.floor(gx); y0 = np.floor(gy); fx = gx - x0; fy = gy - y0
    x0 = x0.astype(np.int64) % G; y0 = y0.astype(np.int64) % G
    x1 = (x0 + 1) % G; y1 = (y0 + 1) % G
    idx = [y0 * G + x0, y0 * G + x1, y1 * G + x0, y1 * G + x1]
    wts = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
    return idx, wts


def fit_field(mid, nrm, d, conf, size, G=16, lam=1.0, conf_min=3.0, iters=5):
    """gladkie, okresowe pole przesuniec (px) z pomiarow szczelin: n . D(mid) = d.
    Waga = pewnosc, odrzucanie odstajacych (Huber), wygladzanie roznicami sasiednich wezlow.
    Rownania normalne skladane bezposrednio (8 niezerowych na pomiar) - szybkie dla wielu odcinkow."""
    import numpy as np
    use = conf >= conf_min
    mid = mid[:, use]; nrm = nrm[:, use]; d = d[use]
    cw = np.minimum(conf[use], 3 * conf_min) / (3 * conf_min)
    m = len(d); GG = G * G; U = 2 * GG
    idx, wts = _grid_weights(G, size, mid[0], mid[1])
    cols = np.stack([*idx, *[i + GG for i in idx]], 1)                       # m x 8
    vals = np.stack([*[w_ * nrm[0] for w_ in wts], *[w_ * nrm[1] for w_ in wts]], 1)
    # wygladzanie (laplasjan okresowy) dla obu skladowych
    Lap = np.zeros((GG, GG))
    for gy in range(G):
        for gx in range(G):
            a = gy * G + gx
            for b in (gy * G + (gx + 1) % G, ((gy + 1) % G) * G + gx):
                Lap[a, a] += 1; Lap[b, b] += 1; Lap[a, b] -= 1; Lap[b, a] -= 1
    LL = np.zeros((U, U)); LL[:GG, :GG] = Lap * lam * lam; LL[GG:, GG:] = Lap * lam * lam
    rw = np.ones(m)
    D = np.zeros(U)
    hub = 1.0 * max(1.0, size[0] / 1024)                    # prog Hubera ~1 px (na 1024)
    ri = np.repeat(cols, 8, 1).ravel(); ci = np.tile(cols, (1, 8)).ravel()
    for _ in range(iters):
        W = cw * rw
        M = LL + 1e-6 * np.eye(U)
        np.add.at(M, (ri, ci), (vals[:, :, None] * vals[:, None, :] * W[:, None, None]).ravel())
        b = np.zeros(U)
        np.add.at(b, cols.ravel(), (vals * (W * d)[:, None]).ravel())
        D = np.linalg.solve(M, b)
        res = (vals * D[cols]).sum(1) - d
        rw = np.minimum(1.0, hub / np.maximum(np.abs(res), 1e-9))
    return {"G": G, "size": tuple(size), "Dx": D[:GG], "Dy": D[GG:], "n": int(m)}


def field_at(field, x, y):
    """przesuniecie (2 x m) pola w pikselach x, y"""
    import numpy as np
    idx, wts = _grid_weights(field["G"], field["size"], x, y)
    dx = sum(field["Dx"][i] * w for i, w in zip(idx, wts))
    dy = sum(field["Dy"][i] * w for i, w in zip(idx, wts))
    return np.stack([dx, dy])


def calibrate(S, lat, model, G=None, lam=0.2, conf_min=3.0):
    """lokalna kalibracja wzoru do szczelin: pole korekty + statystyki (przed / po, px).
    Wezly pola co ok. 2 szerokosci deski (dobrane walidacja krzyzowa - pole przewiduje polozenie
    pominietych szczelin z bledem ~1 px), niezaleznie od rozdzielczosci i wielkosci desek."""
    import numpy as np
    size = S.shape[::-1]
    if G is None:
        G = int(min(48, max(4, round(min(size) / (2 * lat[2])))))
    P0, P1 = seam_edges(lat, model, size)
    reach = 0.3 * lat[2]
    d, conf, nrm, mid = measure_edges(S, P0, P1, reach)
    field = fit_field(mid, nrm, d, conf, size, G, lam, conf_min)
    Q0 = P0 + field_at(field, P0[0], P0[1]); Q1 = P1 + field_at(field, P1[0], P1[1])
    d2, _, _, _ = measure_edges(S, Q0, Q1, reach)
    good = conf >= conf_min
    field["stats"] = {"odcinki": int(len(d)), "pewne": int(good.sum()),
                      "przed_med": round(float(np.median(np.abs(d[good]))), 2),
                      "po_med": round(float(np.median(np.abs(d2[good]))), 2),
                      "przed_p90": round(float(np.percentile(np.abs(d[good]), 90)), 2),
                      "po_p90": round(float(np.percentile(np.abs(d2[good]), 90)), 2)}
    return field
