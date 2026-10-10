# Etap 2: przygotowanie desek - podzial plaszczyzny podlogi na osobne deski
# wg szczelin wykrytych na teksturze (krawedzie desek), z zachowaniem UV.
import bpy, bmesh, math
from . import jodelka as jd
from mathutils import Vector, Matrix


# ---------------------------------------------------------------- wykrywanie szczelin na teksturze
def image_luminance(img):
    """obraz -> tablica jasnosci (wiersz = v od dolu, kolumna = u)"""
    import numpy as np
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    return 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]


def _highpass(p, win):
    """profil minus srednia kroczaca (cykliczna - tekstura sie powtarza)"""
    import numpy as np
    k = np.ones(win) / win
    ext = np.concatenate([p[-win:], p, p[:win]])
    return p - np.convolve(ext, k, mode='same')[win:-win]


def _dips(hp, sens, min_sep):
    """indeksy wyraznych dolkow profilu (ciemne linie), cyklicznie, co najmniej min_sep od siebie"""
    import numpy as np
    n = len(hp)
    mad = np.median(np.abs(hp - np.median(hp))) * 1.4826 + 1e-9
    cand = [i for i in range(n) if hp[i] < -sens * mad
            and hp[i] <= hp[(i - 1) % n] and hp[i] <= hp[(i + 1) % n]]
    cand.sort(key=lambda i: hp[i])                     # najglebsze najpierw
    picked = []
    for i in cand:
        if all(min(abs(i - j), n - abs(i - j)) >= min_sep for j in picked):
            picked.append(i)
    out = []
    for i in sorted(picked):                           # dokladnosc ponizej piksela (parabola)
        a, b, c = hp[(i - 1) % n], hp[i], hp[(i + 1) % n]
        den = a - 2 * b + c
        out.append((i + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)) / n)
    return out


def _joints(sub, thr, cover, min_sep):
    """laczenia (konce desek) w pasie: wiersze, w ktorych ciemna linia przechodzi przez
    wieksza czesc szerokosci deski (sloje i plamy tego nie robia)"""
    import numpy as np
    n = sub.shape[0]
    rowmin = sub.copy()
    for d in (-2, -1, 1, 2):                           # linia moze byc lekko krzywa / gruba
        rowmin = np.minimum(rowmin, np.roll(sub, d, axis=0))
    base = np.median(sub, axis=0)
    cov = ((base[None, :] - rowmin) > thr).mean(axis=1)
    rows = np.where(cov >= cover)[0]
    if len(rows) == 0:
        return []
    # grupy sasiednich wierszy (cyklicznie) -> srodek wazony pokryciem
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 2:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    if len(groups) > 1 and groups[0][0] + n - groups[-1][-1] <= 2:
        groups[0] = groups.pop() + [r + n for r in groups[0]]
    cands = []
    for g in groups:
        g = np.array(g); wts = cov[g % n]
        cands.append((float(wts.max()), float((g * wts).sum() / wts.sum()) % n))
    cands.sort(reverse=True)
    picked = []
    for c, r in cands:
        if all(min(abs(r - q), n - abs(r - q)) >= min_sep for q in picked):
            picked.append(r)
    return sorted(r / n for r in picked)


def detect_planks(lum, sens=4.0, min_width=0.015, min_length=0.05, cover=0.6, contrast=0.3):
    """Wykrywa uklad desek na (powtarzalnej) teksturze.
    Zwraca (os, szczeliny_wzdluz, lista (od, do, laczenia)) we wspolrzednych 0..1:
      os = 'V' gdy deski biegna wzdluz V (szczeliny to pionowe linie w stalym u), 'U' odwrotnie;
      szczeliny_wzdluz = pozycje dlugich szczelin w poprzek desek;
      dla kazdego pasa miedzy szczelinami - pozycje laczen (koncow desek) wzdluz pasa."""
    import numpy as np
    # mediana zamiast sredniej - pojedyncze ciemne plamy (seki) nie udaja szczelin
    pu = np.median(lum, axis=0)                        # f(u): pionowe linie -> dolki
    pv = np.median(lum, axis=1)                        # f(v): poziome linie -> dolki
    hu = _highpass(pu, max(5, len(pu) // 40))
    hv = _highpass(pv, max(5, len(pv) // 40))
    axis = 'V' if np.std(hu) >= np.std(hv) else 'U'
    L = lum if axis == 'V' else lum.T                  # kolumny = w poprzek desek
    across = hu if axis == 'V' else hv
    n_across, n_along = L.shape[1], L.shape[0]
    seams = _dips(across, sens, max(2, int(min_width * n_across)))
    if len(seams) < 2:
        return axis, seams, []
    # kontrast szczeliny: o ile szczelina jest ciemniejsza od drewna obok -> prog dla laczen
    contr = []
    # "drewno obok" szczeliny: 20% szerokosci elementu (stale 8 px bylo wciaz w szczelinie przy duzej mapie)
    gaps_px = [((seams[(i + 1) % len(seams)] - a) % 1.0) * n_across for i, a in enumerate(seams)]
    dd = max(8, int(0.2 * min(gaps_px)))
    for s in seams:
        x = int(round(s * n_across))
        line = np.min([L[:, (x + d) % n_across] for d in (-1, 0, 1)], axis=0)
        side = np.median([L[:, (x + d) % n_across] for d in (-dd, -(3 * dd) // 4, (3 * dd) // 4, dd)], axis=0)
        contr.append(float(np.median(side - line)))
    thr = contrast * float(np.median(contr))
    bands = []
    for i, a in enumerate(seams):
        b = seams[(i + 1) % len(seams)] + (1.0 if i + 1 == len(seams) else 0.0)
        c0, c1 = a + (b - a) * 0.15, a + (b - a) * 0.85  # srodek pasa (bez samych szczelin)
        cols = sorted({int(round(c * n_across)) % n_across
                       for c in np.linspace(c0, c1, max(3, int((c1 - c0) * n_across)))})
        joints = _joints(L[:, cols], thr, cover, max(2, int(min_length * n_along)))
        bands.append((a, b, joints))
    return axis, seams, bands


def _box(a, rad):
    """srednia w oknie (2*rad+1)^2, cyklicznie (tekstura sie powtarza)"""
    import numpy as np
    k = 2 * rad + 1
    c = np.cumsum(np.cumsum(np.pad(a, ((rad + 1, rad), (rad + 1, rad)), mode='wrap'), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def detect_nail_dots(img, cool_thr=-0.06, dark_thr=0.08):
    """lebki gwozdzi na teksturze koloru: male, ciemne, chlodne (szaroniebieskie) punkty na brazowym drewnie.
    Zwraca liste (u, v) w 0..1."""
    import numpy as np
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cool = b - r                                       # drewno ujemne, stal blisko zera
    dark = _box(lum, 6) - lum                          # ciemniejszy od otoczenia
    score = cool + dark
    m = score.copy()
    for dy in range(-3, 4):                            # lokalne maksimum w oknie 7x7
        for dx in range(-3, 4):
            if dx or dy:
                m = np.maximum(m, np.roll(np.roll(score, dy, 0), dx, 1))
    pts = np.argwhere((score == m) & (cool > cool_thr) & (dark > dark_thr))
    return [((x + 0.5) / w, (y + 0.5) / h) for y, x in pts]


def face_material(obj, poly):
    """material faktycznie widoczny na scianie: slot z indeksu sciany (link Object albo Data)"""
    slots = obj.material_slots
    return slots[poly.material_index].material if poly.material_index < len(slots) else None


def used_materials(obj):
    """materialy uzyte na scianach obiektu, od najczesciej uzywanego"""
    from collections import Counter
    cnt = Counter(face_material(obj, p) for p in obj.data.polygons)
    return [m for m, _ in cnt.most_common() if m is not None]


def _images(obj):
    """obrazy z materialow rzeczywiscie uzytych na plaszczyznie (nie ze wszystkich slotow)"""
    imgs = []
    for m in used_materials(obj):
        if m.node_tree:
            imgs += [n.image for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
    return imgs


def _pick(imgs, keys):
    for key in keys:
        for im in imgs:
            if key in (im.name + im.filepath).lower().replace(" ", ""):
                return im
    return None


def find_texture(obj, override=None):
    """obraz do szczelin: wskazany recznie, albo z materialu - najlepiej AO / Cavity, potem Base Color"""
    if override:
        return override
    imgs = _images(obj)
    return _pick(imgs, ("_ao", "ambient", "occlusion", "cavity", "basecolor", "base_color", "albedo",
                        "diffuse", "color")) or (imgs[0] if imgs else None)


def find_color_texture(obj, override=None):
    """obraz do gwozdzi: kolor (lebki gwozdzi odrozniaja sie kolorem od drewna)"""
    if override:
        return override
    return _pick(_images(obj), ("basecolor", "base_color", "albedo", "diffuse", "color"))


def point_in_poly(pt, poly):
    x, y = pt; inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


# ---------------------------------------------------------------- UV bokow i czol deski
def fit_top_uv(samples):
    """UV wierzchu deski jako funkcja liniowa polozenia na wierzchu: uv = c + A*(x, y).
    samples = [((x, y), (u, v))] w ukladzie, w ktorym wierzch lezy w plaszczyznie xy."""
    import numpy as np
    if len(samples) < 3:
        return None
    X = np.array([[x, y, 1.0] for (x, y), _ in samples])
    U = np.array([uv for _, uv in samples])
    coef, *_ = np.linalg.lstsq(X, U, rcond=None)
    return lambda x, y: (coef[0][0] * x + coef[1][0] * y + coef[2][0], coef[0][1] * x + coef[1][1] * y + coef[2][1])


def top_outline(top_faces, to_frame):
    """krawedzie obrysu wierzchu deski (w ukladzie wierzchu, 2D)"""
    tops = set(top_faces)
    segs = []
    for f in top_faces:
        for e in f.edges:
            if sum(1 for g in e.link_faces if g in tops) == 1:
                a, b = to_frame(e.verts[0].co, False), to_frame(e.verts[1].co, False)
                segs.append((Vector((a.x, a.y)), Vector((b.x, b.y))))
    return segs


def _nearest_on(segs, q):
    best, bd = None, 1e18
    for a, b in segs:
        ab = b - a
        t = 0.0 if ab.length_squared < 1e-18 else max(0.0, min(1.0, (q - a).dot(ab) / ab.length_squared))
        c = a + ab * t
        d = (q - c).length_squared
        if d < bd:
            best, bd = c, d
    return best


def unfold_uv(faces, uvl, to_frame, uvf, z_top, segs=None):
    """Rozklada UV scian bocznych (boki, czola, fazy, pekniecia) i spodu przez 'zagiecie' wierzchu:
    punkt dostaje UV punktu wierzchu lezacego w glab deski od najblizszej krawedzi obrysu, w odleglosci
    rownej drodze po powierzchni (odsuniecie w poziomie + glebokosc). Tekstura przechodzi przez krawedz
    bez szwu, z ta sama gestoscia co na wierzchu i bez ciemnej szczeliny sasiedniej deski. Spod - rzut."""
    for f in faces:
        n = to_frame(f.normal, True)
        h = Vector((n.x, n.y)); hl = h.length
        for lp in f.loops:
            q = to_frame(lp.vert.co, False)
            q2 = Vector((q.x, q.y))
            d = max(0.0, z_top - q.z)
            if hl > 0.2 and n.z < 0.9:
                hn = h / hl
                edge = _nearest_on(segs, q2) if segs else None
                if edge is not None:
                    off = max(0.0, (q2 - edge).dot(hn))            # jak daleko na zewnatrz obrysu
                    p = edge - hn * (off + d)
                else:
                    p = q2 - hn * d
                x, y = p.x, p.y
            else:
                x, y = q.x, q.y
            lp[uvl].uv = uvf(x, y)


def board_uv_frame(nrm, origin):
    """uklad z wierzchem deski w plaszczyznie xy (z = normalna wierzchu)"""
    e3 = nrm.normalized(); e1 = e3.orthogonal().normalized(); e2 = e3.cross(e1)
    M = Matrix((e1, e2, e3))

    def to_frame(v, is_dir):
        return M @ v if is_dir else M @ (v - origin)
    return to_frame


def fix_side_uvs(bm, uvl, to_frame, top_faces, other_faces):
    """dopasuj UV wierzchu i rozloz wg niego pozostale sciany"""
    z_top = max(to_frame(v.co, False).z for f in top_faces for v in f.verts)
    samples = []
    for f in top_faces:
        for lp in f.loops:
            q = to_frame(lp.vert.co, False)
            samples.append(((q.x, q.y), tuple(lp[uvl].uv)))
    uvf = fit_top_uv(samples)
    if uvf:
        unfold_uv(other_faces, uvl, to_frame, uvf, z_top, top_outline(top_faces, to_frame))
    return uvf


# ---------------------------------------------------------------- geometria
def clip_rect(poly, u0, u1, v0, v1):
    """Sutherland-Hodgman: wielokat (lista (u,v)) przyciety do prostokata"""
    def clip(pts, inside, inter):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def ix(k, c):
        def f(p, q):
            t = (c - p[k]) / (q[k] - p[k])
            return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
        return f
    for inside, inter in ((lambda p: p[0] >= u0, ix(0, u0)), (lambda p: p[0] <= u1, ix(0, u1)),
                          (lambda p: p[1] >= v0, ix(1, v0)), (lambda p: p[1] <= v1, ix(1, v1))):
        poly = clip(poly, inside, inter)
        if not poly:
            break
    return poly


def face_uv_affine(me, poly, uv):
    """mapowanie afiniczne UV -> punkt 3D (lokalny) dla plaskiej sciany"""
    li = list(poly.loop_indices)
    P = [me.vertices[me.loops[l].vertex_index].co.copy() for l in li]
    T = [uv[l].uv.copy() for l in li]
    best = None
    for i in range(1, len(li) - 1):                    # najlepiej uwarunkowany trojkat
        du1, du2 = T[i] - T[0], T[i + 1] - T[0]
        det = du1.x * du2.y - du1.y * du2.x
        if best is None or abs(det) > abs(best[0]):
            best = (det, i)
    det, i = best
    if abs(det) < 1e-12:
        return None
    du1, du2 = T[i] - T[0], T[i + 1] - T[0]
    dp1, dp2 = P[i] - P[0], P[i + 1] - P[0]
    # P(u,v) = P0 + a*(u-u0) + b*(v-v0)
    a = (dp1 * du2.y - dp2 * du1.y) / det
    b = (dp2 * du1.x - dp1 * du2.x) / det
    return lambda u, v: P[0] + a * (u - T[0].x) + b * (v - T[0].y), a.length, b.length


# ---------------------------------------------------------------- jodelka (skosne panele na siatce)
def clip_convex(subject, clip):
    """Sutherland-Hodgman: wielokat subject przyciety do wypuklego wielokata clip (CCW)"""
    out = list(subject)
    for k in range(len(clip)):
        a, b = clip[k], clip[(k + 1) % len(clip)]

        def inside(q):
            return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) >= 0

        def inter(q0, q1):
            dx, dy = b[0] - a[0], b[1] - a[1]
            den = (q1[0] - q0[0]) * dy - (q1[1] - q0[1]) * dx
            t = ((a[0] - q0[0]) * dy - (a[1] - q0[1]) * dx) / den if abs(den) > 1e-15 else 0.0
            return (q0[0] + (q1[0] - q0[0]) * t, q0[1] + (q1[1] - q0[1]) * t)
        pts, out = out, []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        if not out:
            break
    return out


def herring_layout(src, grid, model, size):
    """deski jodelki na plaszczyznie: {klucz: (kratki (i0,i1,j0,j1), kierunek wzdluz w UV)}.
    grid = (N1, N2, u, faza1, faza2, k, pole_korekty)"""
    lat, field = grid[:6], grid[6]
    uv = src.data.uv_layers.active.data
    us = [l.uv[0] for l in uv]; vs = [l.uv[1] for l in uv]
    boards = jd.layout(lat, model, (min(us), min(vs)), (max(us), max(vs)), size)
    return {key: (r, jd.along_uv(lat, size, *r, field=field)) for key, r in boards.items()}


def result_collection(context, name, src_name, replace):
    """kolekcja wyniku: przy 'Zastap poprzedni wynik' poprzednia kolekcja tej samej plaszczyzny jest
    oprozniana i uzywana ponownie (zamiast '.001' nakladajacego sie na stary wynik)"""
    def ours(c):                                          # wynik tej plaszczyzny (oznaczony albo po nazwach obiektow)
        if c.get("deski_wynik") == src_name:
            return True
        obs = [o for o in c.objects if not o.get("deski_gwozdz")]
        return bool(obs) and all(o.name.startswith(src_name + "_") for o in obs)
    cands = [c for c in bpy.data.collections if (c.name == name or c.name.startswith(name + ".")) and ours(c)]
    cands.sort(key=lambda c: (c.name != name, c.name))
    old = cands[0] if cands else None
    if replace and old is not None:
        old["deski_wynik"] = src_name
        pg = getattr(context.scene, "deski_podglad", None)
        if pg is not None and pg.active and pg.coll == old:
            from . import podglad
            podglad.end_session(context.scene, True, "")
        objs = set(old.all_objects)
        for ob in [ob for ob in bpy.data.objects if ob.parent in objs and ob not in objs]:
            bpy.data.objects.remove(ob)                   # gwozdzie itp. przypiete do starych elementow
        for ob in list(objs):
            me = ob.data if ob.type == 'MESH' else None
            bpy.data.objects.remove(ob)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        from .wyrwanie import purge_orphans
        purge_orphans()
        if old.name not in context.scene.collection.children and not old.users:
            context.scene.collection.children.link(old)
        return old
    out = bpy.data.collections.new(name)
    context.scene.collection.children.link(out)
    out["deski_wynik"] = src_name
    return out


_CACHE = {}
_SKIP = {"n": 0, "w": 0.0}                              # kawalki odrzucone jako wezsze niz 'Min. kawalek'                                            # analiza tekstur - wspolna dla podzialu i podgladu


def _img_key(img):
    """obraz + plik + czas modyfikacji: ta sama tekstura = ta sama analiza"""
    import os
    path = bpy.path.abspath(img.filepath)
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0.0
    return (img.name, path, mt)


def _sub(gen, lo, hi):
    """postep generatora (0..1, tekst) przeskalowany do zakresu lo..hi; zwraca jego wynik"""
    try:
        while True:
            f, t = next(gen)
            yield (lo + (hi - lo) * f, t)
    except StopIteration as e:
        return e.value


def herring_fit_cached(cache, img, p):
    return jd.run(herring_fit_iter(cache, img, p))


def herring_fit_iter(cache, img, p):
    """Jodelka z tekstury, w trzech krokach:
    1) mapa szczelin ze wszystkich map materialu (AO, Cavity, Roughness, Specular, Displacement, Normal...),
    2) globalna siec z widma (kat, szerokosc) + dopasowanie dlugosci, wariantu i przesuniecia,
    3) kalibracja lokalna: pomiar kazdego odcinka szczeliny i gladkie pole korekty z pewnych pomiarow
       (zdjecie nie jest idealna siatka). Zwraca (grid, model, size)."""
    var = {"A": (0, 1), "B": (0, -1), "C": (1, 1), "D": (1, -1)}.get(p.herr_variant)
    ik = _img_key(img)
    key = ("h", ik, p.herr_length, var, p.herr_offset, p.herr_calibrate)
    if key not in cache:
        if ("S", ik) not in cache or cache[("S", ik)][1] is None:
            res = 2048 if max(img.size) >= 2048 else 1024
            S, used = yield from _sub(jd.seam_map_iter(img, res), 0.0, 0.6)
            print("Prepare planks - herringbone: seam maps", used, "resolution", res)
            yield (0.62, "plank grid (spectrum)")
            lat, _ = jd.detect_lattice(S)
            cache[("S", ik)] = (S, lat, jd.edge_scores(S, lat))
        S, lat, (E1, E2, org) = cache[("S", ik)]
        k = lat[5]
        off = None
        if p.herr_offset >= 0:
            n_off = 2 * k
            off = (p.herr_offset // n_off, p.herr_offset % n_off)
        model = yield from _sub(jd.fit_iter(E1, E2, org, k, p.herr_length, var, off), 0.65, 0.9)
        field = None
        if p.herr_calibrate:
            yield (0.92, "local seam calibration")
            field = jd.calibrate(S, lat, model)
            print("Prepare planks - herringbone: local calibration (px)", field["stats"])
        cache[key] = (tuple(lat) + (field,), model, S.shape[::-1])
    return cache[key]


def cell_quad(grid, size, i0, i1, j0, j1, shrink):
    return jd.cell_quad(grid[:6], size, i0, i1, j0, j1, shrink, grid[6])


def cell_poly(grid, size, i0, i1, j0, j1, shrink):
    return jd.cell_poly(grid[:6], size, i0, i1, j0, j1, shrink, grid[6])


def board_quad(grid, size, i0, i1, j0, j1, shrink, tol=0.0):
    """czworokat deski jodelki (4 wierzcholki) zawarty w komorce wyznaczonej przez pole korekty"""
    pts, na, nb = jd.cell_poly(grid[:6], size, i0, i1, j0, j1, shrink, grid[6], raw=True)
    return contain_quad(pts, na, nb, tol)


def ccw(poly):
    a = sum(poly[q - 1][0] * poly[q][1] - poly[q][0] * poly[q - 1][1] for q in range(len(poly)))
    return list(poly) if a > 0 else list(poly)[::-1]


def simplify_outline(pts, tol_uv):
    """Douglas-Peucker dla zamknietego obrysu (UV); narozniki (najdalsze punkty) zostaja.
    tol < polowy szczeliny -> sasiednie deski nadal nie nachodza na siebie"""
    n = len(pts)
    if n <= 4 or tol_uv <= 0:
        return pts
    import math as _m

    def dist(q, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-30:
            return _m.hypot(q[0] - a[0], q[1] - a[1])
        return abs((q[0] - a[0]) * dy - (q[1] - a[1]) * dx) / _m.sqrt(L2)

    def dp(seq):
        if len(seq) <= 2:
            return seq
        a, b = seq[0], seq[-1]
        i, d = max(((i, dist(seq[i], a, b)) for i in range(1, len(seq) - 1)), key=lambda t: t[1])
        if d <= tol_uv:
            return [a, b]
        return dp(seq[:i + 1])[:-1] + dp(seq[i:])
    # podzial na dwa luki miedzy najbardziej odleglymi punktami
    j = max(range(n), key=lambda k: (pts[k][0] - pts[0][0]) ** 2 + (pts[k][1] - pts[0][1]) ** 2)
    i0 = max(range(n), key=lambda k: (pts[k][0] - pts[j][0]) ** 2 + (pts[k][1] - pts[j][1]) ** 2)
    r = pts[i0:] + pts[:i0]
    j = (j - i0) % n
    out = dp(r[:j + 1])[:-1] + dp(r[j:] + [r[0]])[:-1]
    return out if len(out) >= 3 else pts


def contain_quad(pts, na, nb, tol=0.0):
    """Obrys probkowany (boki: na, nb, na, nb punktow, narozniki wspolne) -> czworokat CCW o 4 wierzcholkach,
    zawarty w obrysie: bok, pod ktorym krzywa szczeliny wchodzi do srodka, cofany rownolegle o te glebokosc.
    Wszystkie deski leza w swoich (rozlacznych) komorkach -> nie nachodza na siebie, bez dodatkowych punktow."""
    ci = [0, na - 1, na - 1 + nb - 1, 2 * (na - 1) + nb - 1]
    n = len(pts)
    area = sum(pts[q - 1][0] * pts[q][1] - pts[q][0] * pts[q - 1][1] for q in range(n))
    sg = 1.0 if area > 0 else -1.0
    lines = []
    for k in range(4):
        a, b = ci[k], (ci[(k + 1) % 4] if k < 3 else n)
        A, B = pts[a], pts[b % n]
        dx, dy = B[0] - A[0], B[1] - A[1]
        L = math.hypot(dx, dy) or 1e-12
        nx, ny = -dy / L * sg, dx / L * sg              # normalna do srodka
        off = 0.0
        for q in range(a + 1, b):
            P = pts[q % n]
            off = max(off, (P[0] - A[0]) * nx + (P[1] - A[1]) * ny)
        off = max(0.0, off - tol)                       # pol szczeliny zapasu: sasiad robi to samo -> bez przeciec
        lines.append(((A[0] + nx * off, A[1] + ny * off), (dx, dy)))
    quad = []
    for k in range(4):
        (p0, d0), (p1, d1) = lines[k - 1], lines[k]
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) < 1e-18:
            quad.append(p1); continue
        t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den
        quad.append((p0[0] + d0[0] * t, p0[1] + d0[1] * t))
    return ccw(quad)


def clip_board(board_poly, face_uv):
    """obrys deski (dowolny, lekko wygiety) przyciety do sciany plaszczyzny (wypukla)"""
    return clip_convex(board_poly, ccw(face_uv))


def calibrate_straight(S, axis, seams, bands):
    """kalibracja lokalna prostych desek: kazda dluga szczelina (w odcinkach ~ szerokosc deski) i kazde
    laczenie (koniec deski) mierzone osobno; z pewnych pomiarow gladkie pole korekty (jak w jodelce)"""
    import numpy as np
    h, w = S.shape
    n_acr, n_alg = (w, h) if axis == 'V' else (h, w)
    Wpx = max(8.0, float(np.median([b - a for a, b, _ in bands])) * n_acr)

    def P(acr, alg):                                   # (w poprzek, wzdluz) w UV -> piksele (x, y)
        return (acr * w, alg * h) if axis == 'V' else (alg * w, acr * h)
    s0, s1 = [], []
    nseg = max(1, int(round(n_alg / Wpx)))
    for a in seams:
        for k in range(nseg):
            s0.append(P(a, k / nseg)); s1.append(P(a, (k + 1) / nseg))
    for a, b, joints in bands:
        for j in joints:
            s0.append(P(a, j)); s1.append(P(b, j))
    P0 = np.array(s0, float).T; P1 = np.array(s1, float).T
    reach = 0.3 * Wpx
    d, conf, nrm, mid = jd.measure_edges(S, P0, P1, reach)
    G = int(min(48, max(4, round(min(w, h) / (2 * Wpx)))))
    field = jd.fit_field(mid, nrm, d, conf, (w, h), G, 0.2)
    Q0 = P0 + jd.field_at(field, P0[0], P0[1]); Q1 = P1 + jd.field_at(field, P1[0], P1[1])
    d2, _, _, _ = jd.measure_edges(S, Q0, Q1, reach)
    good = conf >= 3.0
    if good.any():
        field["stats"] = {"odcinki": int(len(d)), "pewne": int(good.sum()),
                          "przed_med": round(float(np.median(np.abs(d[good]))), 2),
                          "po_med": round(float(np.median(np.abs(d2[good]))), 2),
                          "przed_p90": round(float(np.percentile(np.abs(d[good]), 90)), 2),
                          "po_p90": round(float(np.percentile(np.abs(d2[good]), 90)), 2)}
    return field


def straight_detect_cached(cache, img, p):
    return jd.run(straight_detect_iter(cache, img, p))


def straight_detect_iter(cache, img, p):
    """uklad prostych desek (os, szczeliny, pasy z laczeniami) + pole korekty i rozmiar mapy (lub None)"""
    ik = _img_key(img)
    key = ("s", ik, p.straight_method, p.straight_calibrate, p.sensitivity, p.min_width, p.min_length,
           p.joint_cover, p.joint_contrast)
    if key in cache:
        return cache[key]
    field, size = None, None
    if p.straight_method == 'MAPS':
        res = 2048 if max(img.size) >= 2048 else 1024
        if ("S", ik) not in cache:
            S, used = yield from _sub(jd.seam_map_iter(img, res), 0.0, 0.8)
            print("Prepare planks - straight: seam maps", used, "rozdzielczosc", res)
            cache[("S", ik)] = (S, None, None)
        S = cache[("S", ik)][0]
        yield (0.85, "seams and joints")
        # mapa szczelin: szczelina = jasny grzbiet -> dla wykrywania (ciemne linie) odwrocona
        axis, seams, bands = detect_planks(-S.astype(float), p.sensitivity, p.min_width, p.min_length,
                                           p.joint_cover, p.joint_contrast)
        size = (S.shape[1], S.shape[0])
        if p.straight_calibrate and len(seams) >= 2:
            yield (0.92, "local seam calibration")
            field = calibrate_straight(S, axis, seams, bands)
            print("Prepare planks - straight: local calibration (px)", field.get("stats"))
    else:
        yield (0.5, "seams (brightness)")
        axis, seams, bands = detect_planks(image_luminance(img), p.sensitivity, p.min_width,
                                           p.min_length, p.joint_cover, p.joint_contrast)
    cache[key] = (axis, seams, bands, field, size)
    return cache[key]


def warp_rect(field, size, r, step=None, tol=0.0):
    """prostokat UV (u0, u1, v0, v1) -> obrys CCW przesuniety polem korekty; boki probkowane co 'step'
    (UV), zeby sasiednie deski mialy wspolna linie szczeliny"""
    import numpy as np
    u0, u1, v0, v1 = r

    def ticks(a, b):
        if not step or b - a <= step:
            return [a, b]
        n = int(math.ceil((b - a) / step))
        return [a + (b - a) * i / n for i in range(n + 1)]
    tu, tv = ticks(u0, u1), ticks(v0, v1)
    pts = [(u, v0) for u in tu] + [(u1, v) for v in tv[1:]] + [(u, v1) for u in tu[-2::-1]] + [(u0, v) for v in tv[-2:0:-1]]
    cs = np.array(pts, float)
    Dp = jd.field_at(field, cs[:, 0] * size[0], cs[:, 1] * size[1])
    out = [(float(cs[i, 0] + Dp[0, i] / size[0]), float(cs[i, 1] + Dp[1, i] / size[1])) for i in range(len(cs))]
    return contain_quad(out, len(tu), len(tv), tol)


# ---------------------------------------------------------------- nieregularne (kamienie, bruk)
def _label_components(mask):
    """spojne obszary maski (4-sasiedztwo, z zawijaniem - kafel bezszwowy): etykieta = indeks piksela, -1 poza"""
    import numpy as np
    h, w = mask.shape
    BIG = h * w + 1
    lab = np.where(mask, np.arange(h * w).reshape(h, w), BIG).astype(np.int64)
    for _ in range(4000):
        m = lab.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            m = np.minimum(m, np.roll(lab, sh, ax))
        m = np.where(mask, m, BIG)
        flat = m.ravel()
        v = flat < BIG
        jumped = flat.copy()
        jumped[v] = np.minimum(flat[v], flat[flat[v]])       # skok: etykieta piksela-etykiety
        new = jumped.reshape(h, w)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.where(lab < BIG, lab, -1)


def _grow(lab, S):
    """etykiety rosna w szczeliny (sasiedzi stykaja sie w srodku fugi); najpierw piksele o nizszej mapie szczelin"""
    import numpy as np
    for _ in range(500):
        empty = lab < 0
        if not empty.any():
            break
        best = np.full(lab.shape, -1, np.int64)
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb = np.roll(lab, sh, ax)
            best = np.where((best < 0) & (nb >= 0), nb, best)
        lab = np.where(empty & (best >= 0), best, lab)
    return lab


def _loops(lab, x_off=0, y_off=0):
    """obrysy obszarow z krawedzi pikseli (CCW, os y w gore): {etykieta: [petle [(x, y)...]]}"""
    import numpy as np
    edges = {}
    for (dy, dx), (a, b) in (((0, 1), ((1, 0), (1, 1))), ((0, -1), ((0, 1), (0, 0))),
                             ((1, 0), ((1, 1), (0, 1))), ((-1, 0), ((0, 0), (1, 0)))):
        nb = np.roll(lab, (-dy, -dx), (0, 1))            # nb[y, x] = lab[y + dy, x + dx]
        ys, xs = np.nonzero((nb != lab) & (lab >= 0))
        for y, x, L in zip(ys.tolist(), xs.tolist(), lab[ys, xs].tolist()):
            X, Y = x + x_off, y + y_off
            edges.setdefault(L, {})[(X + a[0], Y + a[1])] = (X + b[0], Y + b[1])
    out = {}
    for L, nxt in edges.items():
        loops = []
        while nxt:
            start, cur = next(iter(nxt.items()))
            loop = [start]
            del nxt[start]
            while cur != start and cur in nxt:
                loop.append(cur)
                cur = nxt.pop(cur)
            if len(loop) >= 4:
                loops.append(loop)
        out[L] = loops
    return out


def _dp_open(pts, tol):
    import math as _m
    if len(pts) <= 2:
        return pts
    a, b = pts[0], pts[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = _m.hypot(dx, dy) or 1e-12
    i, d = max(((i, abs((pts[i][0] - a[0]) * dy - (pts[i][1] - a[1]) * dx) / L) for i in range(1, len(pts) - 1)),
               key=lambda t: t[1])
    if d <= tol:
        return [a, b]
    return _dp_open(pts[:i + 1], tol)[:-1] + _dp_open(pts[i:], tol)


def _offset_in(poly, dist):
    """wielokat (CCW) cofniety do srodka o dist (wierzcholki wzdluz dwusiecznej, ograniczone)"""
    n = len(poly); out = []
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        e1 = Vector((p1[0] - p0[0], p1[1] - p0[1])); e2 = Vector((p2[0] - p1[0], p2[1] - p1[1]))
        if e1.length < 1e-12 or e2.length < 1e-12:
            out.append(p1); continue
        n1 = Vector((-e1.y, e1.x)).normalized(); n2 = Vector((-e2.y, e2.x)).normalized()
        bis = n1 + n2
        if bis.length < 1e-6:
            bis = n1
        bis.normalize()
        k = min(dist / max(bis.dot(n1), 0.3), 3 * dist)
        out.append((p1[0] + bis.x * k, p1[1] + bis.y * k))
    return out


def stones_detect_iter(cache, img, p):
    """kamienie / bruk nieregularny: wielokaty w ukladzie kafla (UV 0..1, moga wychodzic poza) + rozmiar mapy"""
    import numpy as np
    ik = _img_key(img)
    key = ("k", ik, p.stone_sens, p.stone_min, p.stone_simplify)
    if key in cache:
        return cache[key]
    res = 1024
    if ("S1024", ik) not in cache:
        S, used = yield from _sub(jd.seam_map_iter(img, res), 0.0, 0.6)
        print("Prepare planks - stones: seam maps", used)
        cache[("S1024", ik)] = S
    S = cache[("S1024", ik)].astype(np.float64)
    yield (0.62, "stone regions")
    inner = S < p.stone_sens
    for _ in range(2):                                   # erozja: slabe polaczenia miedzy kamieniami znikaja
        e = inner.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            e &= np.roll(inner, sh, ax)
        inner = e
    lab = _label_components(inner)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    small = ids[counts < p.stone_min * res * res]
    lab[np.isin(lab, small)] = -1
    yield (0.75, "grout (region growing)")
    lab = _grow(lab, S)
    # obrysy na obrazie 3x3 (kamienie na brzegu kafla w calosci), wybierz kopie ze srodkiem w srodkowym kaflu
    yield (0.82, "stone outlines")
    step = 2                                             # obrysy na 512 px (wystarczy, szybciej)
    lab2 = lab[::step, ::step]
    h2, w2 = lab2.shape
    big = np.tile(lab2, (3, 3))
    mg = int(0.35 * max(w2, h2))                         # kamienie na brzegu kafla: margines zamiast calego 3x3
    sub = big[h2 - mg:2 * h2 + mg, w2 - mg:2 * w2 + mg]
    loops = _loops(sub, x_off=w2 - mg, y_off=h2 - mg)
    polys = []
    for L, lps in loops.items():
        for lp in lps:
            ax = np.array(lp, float)
            area = 0.5 * float(np.sum(ax[:, 0] * np.roll(ax[:, 1], -1) - np.roll(ax[:, 0], -1) * ax[:, 1]))
            if area <= 0:
                continue                                 # otwor (dziura w obszarze)
            cx, cy = ax[:, 0].mean(), ax[:, 1].mean()
            if not (w2 <= cx < 2 * w2 and h2 <= cy < 2 * h2):
                continue
            # uproszczenie: rozciecie petli na dwa luki
            j = int(np.argmax((ax[:, 0] - ax[0, 0]) ** 2 + (ax[:, 1] - ax[0, 1]) ** 2))
            lpp = [tuple(q) for q in lp]
            simp = _dp_open(lpp[:j + 1], p.stone_simplify)[:-1] + _dp_open(lpp[j:] + [lpp[0]], p.stone_simplify)[:-1]
            if len(simp) < 3:
                continue
            polys.append([((x - w2) / w2, (y - h2) / h2) for x, y in simp])
    out = (polys, (w2, h2))
    cache[key] = out
    return out


# ---------------------------------------------------------------- dlugie zadania: porcjami, z paskiem postepu
JOB = {"running": False, "frac": 0.0, "text": "", "cancel": False, "label": ""}


class JobOperator:
    """operator wykonujacy generator job(context) porcjami (~0.1 s na krok) z paskiem postepu w panelu;
    Esc albo 'Przerwij' konczy prace (zrobione deski zostaja). Ze skryptow (execute) - od razu w calosci."""
    _timer = None

    def execute(self, context):
        lvl, msg = jd.run(self.job(context))
        self.report({lvl}, msg)
        return {'CANCELLED'} if lvl == 'ERROR' else {'FINISHED'}

    def invoke(self, context, event):
        if JOB["running"]:
            self.report({'WARNING'}, "The previous task is still running"); return {'CANCELLED'}
        self._gen = self.job(context)
        JOB.update(running=True, frac=0.0, text="start", cancel=False, label=self.bl_label)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context, lvl, msg):
        context.window_manager.event_timer_remove(self._timer)
        JOB.update(running=False, frac=1.0, text="", cancel=False)
        context.workspace.status_text_set(None)
        _redraw(context)
        self.report({lvl}, msg)
        return {'CANCELLED'} if lvl == 'ERROR' else {'FINISHED'}

    def modal(self, context, event):
        import time
        if event.type == 'ESC' or JOB["cancel"]:
            self._gen.close()
            return self._finish(context, 'WARNING', f"{self.bl_label}: cancelled ({JOB['text']})")
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        t0 = time.time()
        try:
            while time.time() - t0 < 0.1:
                f, txt = next(self._gen)
                JOB["frac"], JOB["text"] = max(0.0, min(1.0, f)), txt
        except StopIteration as e:
            lvl, msg = e.value
            return self._finish(context, lvl, msg)
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._finish(context, 'ERROR', f"{self.bl_label}: error - {e}")
        context.workspace.status_text_set(f"{self.bl_label}: {int(JOB['frac'] * 100)}% - {JOB['text']} (Esc = cancel)")
        _redraw(context)
        return {'RUNNING_MODAL'}


def _redraw(context):
    for w in context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


class DESKI_OT_job_cancel(bpy.types.Operator):
    bl_idname = "object.axis_rupture_przerwij"
    bl_label = "Cancel"
    bl_description = "Cancels the running task (finished planks are kept)"

    def execute(self, context):
        JOB["cancel"] = True
        return {'FINISHED'}


def draw_job(layout, op_idname, icon, text=None):
    """przycisk albo - w trakcie pracy - pasek postepu z opisem i przyciskiem Przerwij"""
    if JOB["running"]:
        box = layout.box()
        box.progress(factor=JOB["frac"], type='BAR', text=f"{int(JOB['frac'] * 100)}%  {JOB['text']}")
        box.operator(DESKI_OT_job_cancel.bl_idname, icon='CANCEL')
    else:
        layout.operator(op_idname, icon=icon, text=text) if text else layout.operator(op_idname, icon=icon)


# ---------------------------------------------------------------- operator
class DESKI_OT_split(JobOperator, bpy.types.Operator):
    bl_idname = "object.axis_rupture_przygotuj"
    bl_label = "Split into Planks"
    bl_description = ("Splits the plane into separate planks along seams detected on the texture "
                      "(UVs preserved), adds thickness and puts them in a new collection")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_przygotowanie.source is not None

    def job(self, context):
        """generator: (postep 0..1, tekst) ... -> (poziom, komunikat)"""
        p = context.scene.deski_przygotowanie
        src_objs = [o for o in (p.source,) if o and o.type == 'MESH' and o.data.polygons and o.data.uv_layers]
        if not src_objs:
            return 'ERROR', "Pick a plane (mesh with UVs and a plank texture)"
        out = result_collection(context, p.output_name or (p.source.name + " - planks"), p.source.name, p.replace_output)
        total, msgs = 0, []
        cache = _CACHE
        _SKIP.update(n=0, w=0.0)
        ns = len(src_objs)
        for si, src in enumerate(src_objs):
            lo, mid, hi = si / ns, (si + 0.85) / ns, (si + 1) / ns        # analiza tekstury trwa najdluzej
            img = find_texture(src, p.texture)
            if img is None:
                msgs.append(f"{src.name}: no texture (set it in the Texture field)"); continue
            if img.size[0] == 0:
                return 'ERROR', f"Cannot load image {img.name} ({img.filepath})"
            if p.layout == 'IRREGULAR':
                polys, size = yield from _sub(stones_detect_iter(cache, img, p), lo, mid)
                n = yield from _sub(self.split_stones(src, out, polys, p), mid, hi)
                total += n
                msgs.append(f"{src.name}: {n} stones ({img.name}, per tile {len(polys)})")
                if p.hide_source:
                    src.hide_set(True); src.hide_render = True
                continue
            if p.layout == 'HERRINGBONE':
                grid, model, size = yield from _sub(herring_fit_iter(cache, img, p), lo, mid)
                n = yield from _sub(self.split_herringbone(src, out, grid, model, size, p), mid, hi)
                total += n
                msgs.append(f"{src.name}: {n} herringbone planks ({img.name}, width {grid[2]:.1f} px, "
                            f"length {model[1]} widths, variant {model[3]}, offset {model[2]})")
                if p.hide_source:
                    src.hide_set(True); src.hide_render = True
                continue
            axis, seams, bands, field, fsize = yield from _sub(straight_detect_iter(cache, img, p), lo, mid)
            if len(seams) < 2:
                msgs.append(f"{src.name}: no seams detected on {img.name}"); continue
            dots, cimg = [], None
            if p.detect_nails:
                cimg = find_color_texture(src, p.nail_texture)
                if cimg is not None and cimg.size[0] > 0:
                    nk = ("n", _img_key(cimg), p.nail_dark)
                    if nk not in cache:
                        yield (mid, "nails on texture")
                        cache[nk] = detect_nail_dots(cimg, dark_thr=p.nail_dark)
                    dots = cache[nk]
            n, nn = yield from _sub(self.split_object(context, src, out, axis, bands, p, dots,
                                                      warp=(lambda r, tl=0.0, f=field, z=fsize, st=min(b - a for a, b, _ in bands):
                                                            warp_rect(f, z, r, st, tl)) if field else None), mid, hi)
            total += n
            msgs.append(f"{src.name}: {n} planks ({img.name}, seams {len(seams)}, axis {axis})"
                        + (f", nails {nn} ({cimg.name})" if p.detect_nails and cimg else
                           (", nails: no color texture" if p.detect_nails else "")))
            if p.hide_source:
                src.hide_set(True); src.hide_render = True
        for m in msgs:
            print("Prepare planks -", m)
        if _SKIP["n"]:
            msgs.append(f"skipped {_SKIP['n']} pieces narrower than 'Min. Piece' ({p.min_piece * 100:.1f} cm; "
                        f"widest skipped {_SKIP['w'] * 100:.1f} cm)"
                        + (" - lower 'Min. Piece' in Plank Dimensions" if not total else ""))
        return ('INFO' if total else 'WARNING'), f"Created {total} planks in collection '{out.name}'. " + "; ".join(msgs)

    def split_stones(self, src, out, polys, p):
        """plaszczyzna -> kamienie: wielokaty z kafla powtorzone na kolejnych kaflach UV, cofniete o pol fugi"""
        me = src.data
        uv = me.uv_layers.active.data
        faces = []
        for poly in me.polygons:
            aff = face_uv_affine(me, poly, uv)
            if aff:
                faces.append(([tuple(uv[l].uv) for l in poly.loop_indices], aff, face_material(src, poly)))
        if not faces:
            return 0
        sc = max(src.matrix_world.to_scale())
        m_uv = 0.5 * (faces[0][1][1] + faces[0][1][2]) * sc          # metry na jednostke UV (srednio)
        g = (p.gap / 2) / max(m_uv, 1e-9)
        us = [q[0] for f, _, _ in faces for q in f]; vs = [q[1] for f, _, _ in faces for q in f]
        u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
        items = []
        for tu in range(math.floor(u0) - 1, math.ceil(u1) + 1):
            for tv in range(math.floor(v0) - 1, math.ceil(v1) + 1):
                for pl in polys:
                    q = [(x + tu, y + tv) for x, y in pl]
                    if max(x for x, _ in q) < u0 or min(x for x, _ in q) > u1 or \
                            max(y for _, y in q) < v0 or min(y for _, y in q) > v1:
                        continue
                    items.append(q)
        count = 0
        for bi, q in enumerate(items):
            if bi % 10 == 0:
                yield (bi / max(len(items), 1), f"stones {count} ({bi}/{len(items)})")
            q = _offset_in(ccw(q), g)
            pls = []
            for fuv, aff, mat in faces:
                cl = clip_board(q, fuv)
                if len(cl) >= 3:
                    pls.append(([(aff[0](u, v), (u, v)) for u, v in cl], mat))
            if pls and self.make_board(src, out, pls, p, count + 1) is not None:
                count += 1
        return count

    def split_herringbone(self, src, out, grid, model, size, p):
        """plaszczyzna -> deski jodelki (kazda deska z obrotem zgodnym z jej kierunkiem)"""
        me = src.data
        uv = me.uv_layers.active.data
        faces = []
        for poly in me.polygons:
            aff = face_uv_affine(me, poly, uv)
            if aff:
                faces.append(([tuple(uv[l].uv) for l in poly.loop_indices], aff[0], face_material(src, poly)))
        if not faces:
            return 0
        yield (0.0, "plank layout")
        # ile metrow ma jedna kratka (szerokosc deski) - do szczeliny miedzy deskami
        import numpy as np
        N1, N2, per = grid[:3]
        Minv = np.linalg.inv(np.array([N1, N2]))
        d = Minv @ np.array([per, 0.0])
        fuv0, f30, _ = faces[0]
        u0, v0 = fuv0[0]
        m_cell = (f30(u0 + d[0] / size[0], v0 + d[1] / size[1]) - f30(u0, v0)).length * \
            max(src.matrix_world.to_scale())
        shrink = (p.gap / 2) / max(m_cell, 1e-9)
        tol_uv = (p.gap / 2) / max(m_cell, 1e-9) * (np.linalg.norm(d) / max(size))   # pol szczeliny w UV
        count = 0
        items = sorted(herring_layout(src, grid, model, size).items())
        for bi, (key, ((i0, i1, j0, j1), (du, dv))) in enumerate(items):
            if bi % 10 == 0:
                yield (bi / max(len(items), 1), f"planks {count} (layout {bi}/{len(items)})")
            quad = board_quad(grid, size, i0, i1, j0, j1, shrink, tol_uv)
            polys, along = [], None
            for fuv, f3, mat in faces:
                cl = clip_board(quad, fuv)
                if len(cl) < 3:
                    continue
                polys.append(([(f3(u, v), (u, v)) for u, v in cl], mat))
                if along is None:
                    uc, vc = cl[0]
                    along = f3(uc + du, vc + dv) - f3(uc, vc)
            if not polys:
                continue
            if self.make_board(src, out, polys, p, count + 1, along=along) is not None:
                count += 1
        return count

    def split_object(self, context, src, out, axis, bands, p, dots=(), warp=None):
        me = src.data
        uv = me.uv_layers.active.data
        pieces = {}                                    # klucz deski -> lista wielokatow [(P3d, uv)]
        rects = {}                                     # klucz deski -> (A, B w poprzek, c0, c1 wzdluz) w UV
        faces_uv = {}                                  # klucz deski -> [(wielokat UV, mapowanie UV->3D)]
        # granice desek wzdluz pasa - liczone raz dla calego obiektu (deska przez kilka scian = jedna deska)
        all_alg = [l.uv[1] if axis == 'V' else l.uv[0] for l in uv]
        glo, ghi = min(all_alg) - 1, max(all_alg) + 1
        band_edges = []
        for a, b, joints in bands:
            cuts = sorted(j + m for j in joints for m in range(math.floor(glo) - 1, math.ceil(ghi) + 2)
                          if glo < j + m < ghi)
            band_edges.append([glo] + cuts + [ghi])

        # dodatkowy podzial zbyt dlugich desek w proporcji 40/60 (naprzemiennie 60/40 w sasiednich pasach)
        alo, ahi = min(all_alg), max(all_alg)          # rzeczywisty zasieg plaszczyzny wzdluz desek
        mw = src.matrix_world
        m_per_uv = 0.0
        for poly in me.polygons:
            aff = face_uv_affine(me, poly, uv)
            if aff:
                u0, v0 = uv[poly.loop_indices[0]].uv
                q0, q1 = aff[0](u0, v0), (aff[0](u0, v0 + 1) if axis == 'V' else aff[0](u0 + 1, v0))
                m_per_uv = (mw @ q1 - mw @ q0).length   # metry na jednostke UV wzdluz desek
                break
        edge_cache = {}

        def edges_for(t, bi):
            if (t, bi) in edge_cache:
                return edge_cache[(t, bi)]
            base = band_edges[bi]
            if p.max_length <= 0 or m_per_uv <= 0:
                edge_cache[(t, bi)] = base
                return base
            res = [base[0]]
            flip = (t * len(bands) + bi) % 2 == 1
            for c0, c1 in zip(base, base[1:]):
                lo_, hi_ = max(c0, alo), min(c1, ahi)   # deska przycieta do plaszczyzny
                inner = []

                def split(x0, x1, fl):
                    if (x1 - x0) * m_per_uv <= p.max_length:
                        return
                    c = x0 + (x1 - x0) * (0.6 if fl else 0.4)
                    split(x0, c, not fl); inner.append(c); split(c, x1, not fl)
                if hi_ > lo_:
                    split(lo_, hi_, flip)
                res += sorted(inner) + [c1]
            edge_cache[(t, bi)] = res
            return res

        yield (0.0, "plane split")
        for poly in me.polygons:
            aff = face_uv_affine(me, poly, uv)
            if aff is None:
                continue
            f3, su, sv = aff
            fuv = [tuple(uv[l].uv) for l in poly.loop_indices]
            # os "wzdluz desek" / "w poprzek" w UV
            if axis == 'V':
                acr = [q[0] for q in fuv]; alg = [q[1] for q in fuv]; s_acr, s_alg = su, sv
            else:
                acr = [q[1] for q in fuv]; alg = [q[0] for q in fuv]; s_acr, s_alg = sv, su
            g_acr = p.gap / 2 / max(s_acr, 1e-9)       # polowa szczeliny w jednostkach UV
            g_alg = p.gap / 2 / max(s_alg, 1e-9)
            for t in range(math.floor(min(acr)) - 1, math.ceil(max(acr)) + 1):
                for bi, (a, b, joints) in enumerate(bands):
                    A, Bb = a + t, b + t
                    if Bb < min(acr) or A > max(acr):
                        continue
                    lo, hi = min(alg), max(alg)
                    edges = edges_for(t, bi)
                    for c0, c1 in zip(edges, edges[1:]):
                        if c1 < lo or c0 > hi:
                            continue
                        if axis == 'V':
                            r = (A + g_acr, Bb - g_acr, c0 + g_alg, c1 - g_alg)
                        else:
                            r = (c0 + g_alg, c1 - g_alg, A + g_acr, Bb - g_acr)
                        if r[0] >= r[1] or r[2] >= r[3]:
                            continue
                        cl = clip_board(warp(r, (p.gap / 2) / max(m_per_uv, 1e-9)), fuv) if warp else clip_rect(fuv, *r)
                        if len(cl) < 3:
                            continue
                        pts = [(f3(u, v), (u, v)) for u, v in cl]
                        # klucz: pas + odcinek (bez numeru kafla wzdluz - te same deski w roznych scianach)
                        key = (t, bi, round(c0, 6))
                        pieces.setdefault(key, []).append((pts, face_material(src, poly)))
                        rects[key] = (A, Bb, c0, c1)
                        faces_uv.setdefault(key, []).append((cl, f3))
        count = 0
        n_nails = 0
        items = sorted(pieces.items())
        for bi, (key, polys) in enumerate(items):
            if bi % 10 == 0:
                yield (0.1 + 0.9 * bi / max(len(items), 1), f"planks {bi}/{len(items)}")
            res = self.make_board(src, out, polys, p, count + 1)
            if res is None:
                continue
            ob, c = res
            count += 1
            if dots:
                loc = self.board_nails(key, rects[key], faces_uv[key], dots, axis, m_per_uv, alo, ahi)
                if loc:
                    ob["deski_gwozdzie_tex"] = [x for q in loc for x in (q - c)]   # lokalnie, na wierzchu deski
                    n_nails += len(loc)
        return count, n_nails

    def make_board(self, src, out, polys, p, idx, along=None):
        """deska z wielokatow wierzchu [(pts, material)]: grubosc, faza, UV bokow, origin w srodku.
        along = kierunek deski (lokalnie, w plaszczyznie) - obiekt dostaje wtedy obrot zgodny z deska.
        Zwraca (obiekt, srodek w ukladzie plaszczyzny) albo None."""
        me = src.data
        mw = src.matrix_world
        bm = bmesh.new()
        uvl = bm.loops.layers.uv.new(me.uv_layers.active.name)
        mats = []                                  # materialy tej deski = te ze scian plaszczyzny
        for pts, mat in polys:
            vs = [bm.verts.new(co) for co, _ in pts]
            try:
                f = bm.faces.new(vs)
            except ValueError:
                continue
            if mat not in mats:
                mats.append(mat)
            f.material_index = mats.index(mat)
            for lp, (_, t) in zip(f.loops, pts):
                lp[uvl].uv = t
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
        if not bm.faces:
            bm.free(); return None
        if len(bm.faces) > 1:
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1), verts=bm.verts[:], edges=bm.edges[:],
                                     delimit={'UV', 'MATERIAL'})
        ref = src_normal(src)                          # normalna plaszczyzny (podloga, sciana, dach)
        for f in bm.faces:
            if f.normal.dot(ref) < 0:
                f.normal_flip()
        # n-gony (deska przycieta skosnie brzegiem plaszczyzny) -> czworokaty + trojkat
        ngons = [f for f in bm.faces if len(f.verts) > 4]
        if ngons:
            r = bmesh.ops.triangulate(bm, faces=ngons, ngon_method='BEAUTY')
            bmesh.ops.join_triangles(bm, faces=r["faces"], angle_face_threshold=math.pi, angle_shape_threshold=math.pi,
                                     cmp_uvs=False, cmp_materials=True)
        bm.faces.ensure_lookup_table()
        nrm = bm.faces[0].normal.copy()
        # uklad deski: X wzdluz deski, Z normalna plaszczyzny (dla desek prostych - uklad plaszczyzny)
        R = Matrix.Identity(3)
        if along is not None:
            X = (along - nrm * along.dot(nrm)).normalized()
            R = Matrix((X, nrm.cross(X), nrm)).transposed()
            bm.transform(R.inverted().to_4x4())
            nrm = Vector((0, 0, 1))
        # odrzuc drzazgi na brzegach plaszczyzny
        vs = [v.co for v in bm.verts]
        ext = sorted((max(c[k] for c in vs) - min(c[k] for c in vs)) for k in range(3))
        if ext[1] < p.min_piece:
            _SKIP["n"] += 1; _SKIP["w"] = max(_SKIP["w"], ext[1])
            bm.free(); return None
        top = bm.faces[:]
        if p.thickness > 0:
            r = bmesh.ops.extrude_face_region(bm, geom=top)
            nv = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
            bmesh.ops.translate(bm, verts=nv, vec=-nrm * p.thickness)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            if p.chamfer > 0:
                # faza gornych krawedzi (jak w oryginalnych deskach)
                zt = max(w.co.dot(nrm) for w in bm.verts)
                top_edges = [e for e in bm.edges if all(v.co.dot(nrm) > zt - 1e-6 for v in e.verts) and e.is_manifold
                             and len(e.link_faces) == 2 and abs(e.link_faces[0].normal.dot(e.link_faces[1].normal)) < 0.5]
                if top_edges:
                    bmesh.ops.bevel(bm, geom=top_edges, offset=min(p.chamfer, p.thickness * 0.45),
                                    offset_type='OFFSET', segments=1, affect='EDGES', clamp_overlap=True)
            # UV bokow, czol, fazy i spodu - rozlozone z wierzchu (inaczej sa sciagniete w linie)
            to_frame = board_uv_frame(nrm, next(iter(bm.verts)).co.copy())
            z_max = max(to_frame(v.co, False).z for v in bm.verts)
            tops = [f for f in bm.faces if to_frame(f.normal, True).z > 0.999
                    and all(to_frame(v.co, False).z > z_max - 1e-6 for v in f.verts)]
            tset = set(tops)
            fix_side_uvs(bm, uvl, to_frame, tops, [f for f in bm.faces if f not in tset])
        bm.faces.ensure_lookup_table()
        dm = bpy.data.meshes.new(f"{src.name}_{PIECE.get(getattr(p, 'surface', 'DESKI'), 'Plank')}")
        bm.to_mesh(dm); bm.free()
        for mat in mats:
            dm.materials.append(mat)
        for poly in dm.polygons:
            poly.use_smooth = False
        # origin w srodku deski (Bounds Center), jak oczekuje etap 1
        vs = [v.co for v in dm.vertices]
        c = Vector([(min(v[k] for v in vs) + max(v[k] for v in vs)) / 2 for k in range(3)])
        dm.transform(Matrix.Translation(-c))
        ob = bpy.data.objects.new(f"{src.name}_{PIECE.get(getattr(p, 'surface', 'DESKI'), 'Plank')}_{idx:03d}", dm)
        ob.matrix_world = mw @ R.to_4x4() @ Matrix.Translation(c)
        out.objects.link(ob)
        return ob, R @ c

    @staticmethod
    def board_nails(key, rect, fpolys, dots, axis, m_per_uv, alo, ahi):
        """gwozdzie z tekstury na tej desce: blisko bocznych krawedzi (narozniki) i parami w poprzek"""
        A, Bb, c0, c1 = rect
        lo, hi = max(c0, alo), min(c1, ahi)
        cand = []
        for du, dv in dots:
            ac0, al0 = (du, dv) if axis == 'V' else (dv, du)
            for i in range(math.floor(A) - 1, math.ceil(Bb) + 1):
                acr = ac0 + i
                if not (A < acr < Bb):
                    continue
                f = (acr - A) / (Bb - A)               # polozenie w poprzek deski (0..1)
                if not (0.04 < f < 0.42 or 0.58 < f < 0.96):
                    continue
                for j in range(math.floor(lo) - 1, math.ceil(hi) + 1):
                    alg = al0 + j
                    if lo < alg < hi:
                        cand.append((acr, alg, f < 0.5))
        pair_tol = 0.06 / max(m_per_uv, 1e-9)          # para gwozdzi: do 6 cm roznicy wzdluz deski
        out = []
        for acr, alg, left in cand:
            if not any(l2 != left and abs(a2 - alg) < pair_tol for _, a2, l2 in cand):
                continue                               # samotna kropka (plama, sek) - nie gwozdz
            u, v = (acr, alg) if axis == 'V' else (alg, acr)
            for cl, f3 in fpolys:
                if point_in_poly((u, v), cl):
                    out.append(f3(u, v))
                    break
        return out


# ---------------------------------------------------------------- ustawienia + panel
PIECE = {"DESKI": "Plank", "PLYTKI": "Tile", "CEGLY": "Brick", "DACHOWKI": "RoofTile", "KOSTKA": "Paver",
         "LOD": "Floe"}

# presety powierzchni: grubosc, fuga, faza, gwozdzie, maks. dlugosc, min. dlugosc do giecia (etap 2)
PRESETS = {
    "DESKI":    dict(thickness=0.012, gap=0.001, chamfer=0.002, detect_nails=True, max_length=2.5),
    "PLYTKI":   dict(thickness=0.008, gap=0.003, chamfer=0.001, detect_nails=False, max_length=0.0),
    "CEGLY":    dict(thickness=0.06, gap=0.008, chamfer=0.004, detect_nails=False, max_length=0.0),
    "DACHOWKI": dict(thickness=0.015, gap=0.002, chamfer=0.001, detect_nails=False, max_length=0.0),
    "KOSTKA":   dict(thickness=0.06, gap=0.006, chamfer=0.008, detect_nails=False, max_length=0.0),
}


def _apply_preset(self, context):
    for k, v in PRESETS.get(self.surface, {}).items():
        setattr(self, k, v)


def make_piece(src, out, polys, p, idx, along=None):
    """element (deska / plytka / kra...) z wielokatow wierzchu - do uzycia z innych modulow"""
    return DESKI_OT_split.make_board(None, src, out, polys, p, idx, along=along)


_NRM = {}


def src_normal(src):
    """srednia normalna scian plaszczyzny zrodlowej (lokalnie) - kierunek 'na zewnatrz' elementow"""
    key = (src.name, len(src.data.polygons), src.data.name)
    if key not in _NRM:
        n = Vector()
        for poly in src.data.polygons:
            n += poly.normal * poly.area
        _NRM[key] = n.normalized() if n.length > 1e-12 else Vector((0, 0, 1))
    return _NRM[key]


def _plane_poll(self, o):
    return o.type == 'MESH' and len(o.data.uv_layers) > 0


class DeskiPrepSettings(bpy.types.PropertyGroup):
    surface: bpy.props.EnumProperty(name="Surface", default='DESKI', update=_apply_preset, items=(
        ('DESKI', "Planks", "Planks / parquet: nailed down", 'MESH_PLANE', 0),
        ('PLYTKI', "Tiles", "Tiles / coffers: thin, wide grout, no nails", 'MESH_GRID', 1),
        ('CEGLY', "Bricks", "Bricks / masonry / panels: thick, mortar joints", 'MOD_BUILD', 2),
        ('DACHOWKI', "Roof Tiles", "Roof tiles / shingles / roofing", 'MOD_ARRAY', 3),
        ('KOSTKA', "Pavers", "Paving stones / pavement slabs: thick, bevelled", 'SNAP_VOLUME', 4)),
        description="Sets thickness, grout, bevel and nails for the surface type (you can change them later)")
    source: bpy.props.PointerProperty(name="Plane", type=bpy.types.Object, poll=_plane_poll,
                                      description="Floor plane to split (mesh with UVs and a plank texture) - "
                                                  "pick from the list or with the eyedropper in the viewport")
    texture: bpy.props.PointerProperty(name="Texture (optional)", type=bpy.types.Image,
                                       description="Image used to detect seams. Empty = from the material (AO/Cavity, then Base Color)")
    output_name: bpy.props.StringProperty(name="Output Collection", default="",
                                          description="Empty = '<collection> - planks'")
    layout: bpy.props.EnumProperty(name="Plank Layout", default='STRAIGHT', items=(
        ('STRAIGHT', "Rows", "Elements in rows (planks, tiles, bricks): seams along texture U or V + joints"),
        ('IRREGULAR', "Irregular", "Stones / cobbles / irregular paving: each element detected as a region "
                                      "surrounded by grout (any shape)"),
        ('HERRINGBONE', "Herringbone (diagonal)", "Diagonal herringbone panels: angle and width detected from the texture, "
                                            "plank length set by hand")))
    herr_length: bpy.props.IntProperty(name="Plank Length (in widths)", default=0, min=0, max=30,
                                       description="How many plank widths make up its length. 0 = pick from the texture "
                                                   "(best fit to the seams, 3..12) - check with Preview")
    herr_variant: bpy.props.EnumProperty(name="Pattern Variant", default='AUTO', items=(
        ('AUTO', "Auto", "Fit to texture"), ('A', "A", ""), ('B', "B (mirrored)", ""),
        ('C', "C (rotated)", ""), ('D', "D (rotated, mirrored)", "")),
        description="Herringbone orientation. If Auto misses, change it and check with Preview")
    herr_calibrate: bpy.props.BoolProperty(name="Local Calibration", default=True,
                                           description="Fit every seam separately (a photo is not a perfect grid): "
                                                       "confident seams correct the position, faint ones are interpolated")
    herr_offset: bpy.props.IntProperty(name="Pattern Offset", default=-1, min=-1, max=60,
                                       description="Herringbone offset index (successive pattern positions on the grid). "
                                                   "-1 = fit to texture")
    stone_sens: bpy.props.FloatProperty(name="Grout Threshold", default=1.5, min=0.2, max=8.0,
                                        description="How distinct the grout must be to separate stones. Lower = more, "
                                                    "smaller stones")
    stone_min: bpy.props.FloatProperty(name="Min. Stone", default=0.0006, min=0.00001, max=0.05, precision=5,
                                       description="Smaller regions (fraction of a tile) are merged into neighbours")
    stone_simplify: bpy.props.FloatProperty(name="Outline Simplification", default=1.2, min=0.3, max=6.0,
                                            description="Outline simplification tolerance (px of a 512 map) - higher = fewer faces")
    straight_method: bpy.props.EnumProperty(name="Detection", default='MAPS', items=(
        ('MAPS', "All Maps", "Seams from all material maps (AO, Cavity, Roughness, Normal, height...) - "
                                   "robust to wood grain and stains"),
        ('LUMA', "Single Map Brightness", "Previous method: dark lines on one texture (AO / Base Color)")))
    straight_calibrate: bpy.props.BoolProperty(name="Local Calibration", default=True,
                                               description="Fit every seam and every joint separately (a photo is not "
                                                           "a perfect grid). Only with the 'All Maps' method")
    sensitivity: bpy.props.FloatProperty(name="Sensitivity (deviations)", default=4.0, min=1.0, max=20.0,
                                         description="How distinct a dark line must be to count as a seam. "
                                                     "Lower = more seams")
    min_width: bpy.props.FloatProperty(name="Min. Plank Width", default=0.015, min=0.001, max=0.5,
                                       subtype='FACTOR', description="As a fraction of the texture width")
    min_length: bpy.props.FloatProperty(name="Min. Plank Length", default=0.05, min=0.001, max=1.0,
                                        subtype='FACTOR', description="As a fraction of the texture length")
    joint_contrast: bpy.props.FloatProperty(name="Joints: Contrast", default=0.3, min=0.05, max=1.0,
                                            description="How dark (relative to the seam between planks) the "
                                                        "cross line at a plank end must be. Lower = more joints")
    joint_cover: bpy.props.FloatProperty(name="Joints: Coverage", default=0.6, min=0.1, max=1.0, subtype='FACTOR',
                                         description="How much of the plank width the end line must cross")
    max_length: bpy.props.FloatProperty(name="Max. Plank Length", default=2.5, min=0.0, unit='LENGTH',
                                        description="Planks longer than this are split 40/60 "
                                                    "(alternating in neighbouring rows). 0 = no split")
    thickness: bpy.props.FloatProperty(name="Thickness", default=0.012, min=0.0, unit='LENGTH',
                                       description="Plank extruded downward from the plane (the top stays in place)")
    gap: bpy.props.FloatProperty(name="Gap", default=0.001, min=0.0, unit='LENGTH',
                                 description="Gap between neighbouring planks")
    chamfer: bpy.props.FloatProperty(name="Top edge bevel", default=0.002, min=0.0, unit='LENGTH')
    min_piece: bpy.props.FloatProperty(name="Min. Piece", default=0.02, min=0.0, unit='LENGTH',
                                       description="Skip slivers narrower than this (at the plane edges)")
    hide_source: bpy.props.BoolProperty(name="Hide source plane", default=True)
    replace_output: bpy.props.BoolProperty(name="Replace Previous Result", default=True,
                                           description="Splitting the same plane again replaces the previous pieces "
                                                       "(instead of creating a new '.001' collection on top of the old ones)")
    detect_nails: bpy.props.BoolProperty(name="Detect Nails", default=True,
                                         description="Nail heads from the color texture (dark, cool dots in pairs near "
                                                     "the plank edges) - used for tearing and details")
    nail_texture: bpy.props.PointerProperty(name="Nail Texture (optional)", type=bpy.types.Image,
                                            description="Empty = Base Color from the material")
    nail_dark: bpy.props.FloatProperty(name="Nails: Contrast", default=0.08, min=0.01, max=0.5,
                                       description="How much darker a nail head must be than the wood next to it. Lower = more nails")


class DESKI_OT_split_preview(JobOperator, bpy.types.Operator):
    bl_idname = "object.axis_rupture_podglad"
    bl_label = "Preview Split"
    bl_description = ("Draws plank outlines (detected herringbone) on the planes - to check the fit "
                      "to the texture before cutting. The 'Split Preview' object can be deleted afterwards")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_przygotowanie.source is not None

    def job(self, context):
        p = context.scene.deski_przygotowanie
        old = bpy.data.objects.get("Split Preview")
        if old:
            bpy.data.objects.remove(old)
        bm = bmesh.new()
        cache, info = _CACHE, []
        srcs = [o for o in (p.source,) if o and o.type == 'MESH' and o.data.polygons and o.data.uv_layers]
        for si, src in enumerate(srcs):
            img = find_texture(src, p.texture)
            if img is None or img.size[0] == 0:
                info.append(f"{src.name}: no texture"); continue
            grid, model, size = yield from _sub(herring_fit_iter(cache, img, p), si / len(srcs), (si + 0.9) / len(srcs))
            info.append(f"{src.name}: plank length {model[1]} widths, variant {model[3]}, offset {model[2]}, "
                        f"match {model[0]:.4f}")
            me = src.data; uv = me.uv_layers.active.data; mw = src.matrix_world
            for (i0, i1, j0, j1), _ in herring_layout(src, grid, model, size).values():
                quad = board_quad(grid, size, i0, i1, j0, j1, 0.0)
                for poly in me.polygons:
                    aff = face_uv_affine(me, poly, uv)
                    if not aff:
                        continue
                    cl = clip_board(quad, [tuple(uv[l].uv) for l in poly.loop_indices])
                    if len(cl) < 3:
                        continue
                    nrm = (mw.to_3x3() @ poly.normal).normalized()
                    vs = [bm.verts.new(mw @ aff[0](u, v) + nrm * 0.002) for u, v in cl]
                    for k in range(len(vs)):
                        bm.edges.new((vs[k], vs[k - 1]))
        me = bpy.data.meshes.new("Split Preview")
        bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new("Split Preview", me)
        ob.show_in_front = True; ob.hide_render = True
        context.scene.collection.objects.link(ob)
        return 'INFO', "; ".join(info) or "No planes"


classes = (DeskiPrepSettings, DESKI_OT_split, DESKI_OT_split_preview, DESKI_OT_job_cancel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_przygotowanie = bpy.props.PointerProperty(type=DeskiPrepSettings)


def unregister():
    del bpy.types.Scene.deski_przygotowanie
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
