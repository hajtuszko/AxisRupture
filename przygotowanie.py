# Etap 2: przygotowanie desek - podzial plaszczyzny podlogi na osobne deski
# wg szczelin wykrytych na teksturze (krawedzie desek), z zachowaniem UV.
import bpy, bmesh, math
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
    for s in seams:
        x = int(round(s * n_across))
        line = np.min([L[:, (x + d) % n_across] for d in (-1, 0, 1)], axis=0)
        side = np.median([L[:, (x + d) % n_across] for d in (-8, -6, 6, 8)], axis=0)
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


def _images(obj):
    imgs = []
    for slot in obj.material_slots:
        m = slot.material
        if m and m.node_tree:
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


# ---------------------------------------------------------------- operator
class DESKI_OT_split(bpy.types.Operator):
    bl_idname = "object.deski_przygotuj"
    bl_label = "Podziel na deski"
    bl_description = ("Dzieli plaszczyzny z kolekcji na osobne deski wg szczelin wykrytych na teksturze "
                      "(z zachowaniem UV), nadaje grubosc i wstawia do nowej kolekcji")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_przygotowanie.source is not None

    def execute(self, context):
        p = context.scene.deski_przygotowanie
        src_objs = [o for o in p.source.all_objects if o.type == 'MESH' and o.data.polygons and o.data.uv_layers]
        if not src_objs:
            self.report({'ERROR'}, "W kolekcji nie ma plaszczyzn z UV"); return {'CANCELLED'}
        out = bpy.data.collections.new(p.output_name or (p.source.name + " - deski"))
        context.scene.collection.children.link(out)
        total, msgs = 0, []
        cache = {}
        for src in src_objs:
            img = find_texture(src, p.texture)
            if img is None:
                msgs.append(f"{src.name}: brak tekstury"); continue
            if img.name not in cache:
                if img.size[0] == 0:
                    self.report({'ERROR'}, f"Nie mozna wczytac obrazu {img.name} ({img.filepath})")
                    return {'CANCELLED'}
                cache[img.name] = detect_planks(image_luminance(img), p.sensitivity, p.min_width,
                                                p.min_length, p.joint_cover, p.joint_contrast)
            axis, seams, bands = cache[img.name]
            if len(seams) < 2:
                msgs.append(f"{src.name}: nie wykryto szczelin na {img.name}"); continue
            dots, cimg = [], None
            if p.detect_nails:
                cimg = find_color_texture(src, p.nail_texture)
                if cimg is not None and cimg.size[0] > 0:
                    if ("n", cimg.name) not in cache:
                        cache[("n", cimg.name)] = detect_nail_dots(cimg, dark_thr=p.nail_dark)
                    dots = cache[("n", cimg.name)]
            n, nn = self.split_object(context, src, out, axis, bands, p, dots)
            total += n
            msgs.append(f"{src.name}: {n} desek ({img.name}, szczelin {len(seams)}, os {axis})"
                        + (f", gwozdzi {nn} ({cimg.name})" if p.detect_nails and cimg else
                           (", gwozdzie: brak tekstury koloru" if p.detect_nails else "")))
            if p.hide_source:
                src.hide_set(True); src.hide_render = True
        for m in msgs:
            print("Przygotowanie desek -", m)
        self.report({'INFO'} if total else {'WARNING'}, f"Utworzono {total} desek w kolekcji '{out.name}'. " + "; ".join(msgs))
        return {'FINISHED'} if total else {'CANCELLED'}

    def split_object(self, context, src, out, axis, bands, p, dots=()):
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
                        cl = clip_rect(fuv, *r)
                        if len(cl) < 3:
                            continue
                        pts = [(f3(u, v), (u, v)) for u, v in cl]
                        # klucz: pas + odcinek (bez numeru kafla wzdluz - te same deski w roznych scianach)
                        key = (t, bi, round(c0, 6))
                        pieces.setdefault(key, []).append(pts)
                        rects[key] = (A, Bb, c0, c1)
                        faces_uv.setdefault(key, []).append((cl, f3))
        count = 0
        n_nails = 0
        mw = src.matrix_world
        for key, polys in sorted(pieces.items()):
            bm = bmesh.new()
            uvl = bm.loops.layers.uv.new(me.uv_layers.active.name)
            for pts in polys:
                vs = [bm.verts.new(co) for co, _ in pts]
                try:
                    f = bm.faces.new(vs)
                except ValueError:
                    continue
                for lp, (_, t) in zip(f.loops, pts):
                    lp[uvl].uv = t
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
            if not bm.faces:
                bm.free(); continue
            if len(bm.faces) > 1:
                bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1), verts=bm.verts[:], edges=bm.edges[:],
                                         delimit={'UV', 'MATERIAL'})
            # odrzuc drzazgi na brzegach plaszczyzny
            vs = [v.co for v in bm.verts]
            ext = sorted((max(c[k] for c in vs) - min(c[k] for c in vs)) for k in range(3))
            if ext[1] < p.min_piece:
                bm.free(); continue
            for f in bm.faces:
                if f.normal.z < 0:
                    f.normal_flip()
            bm.faces.ensure_lookup_table()
            nrm = bm.faces[0].normal.copy()
            top = bm.faces[:]
            if p.thickness > 0:
                r = bmesh.ops.extrude_face_region(bm, geom=top)
                nv = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
                bmesh.ops.translate(bm, verts=nv, vec=-nrm * p.thickness)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
                if p.chamfer > 0:
                    # faza gornych krawedzi (jak w oryginalnych deskach)
                    top_edges = [e for e in bm.edges if all(v.co.dot(nrm) > max(w.co.dot(nrm) for w in bm.verts) - 1e-6
                                                            for v in e.verts) and e.is_manifold
                                 and len(e.link_faces) == 2 and abs(e.link_faces[0].normal.dot(e.link_faces[1].normal)) < 0.5]
                    if top_edges:
                        bmesh.ops.bevel(bm, geom=top_edges, offset=min(p.chamfer, p.thickness * 0.45),
                                        offset_type='OFFSET', segments=1, affect='EDGES', clamp_overlap=True)
                # UV bokow, czol, fazy i spodu - rozlozone z wierzchu (inaczej sa sciagniete w linie)
                to_frame = board_uv_frame(nrm, next(iter(bm.verts)).co.copy())
                z_max = max(to_frame(v.co, False).z for v in bm.verts)
                tops = [f for f in bm.faces if to_frame(f.normal, True).z > 0.999
                        and all(to_frame(v.co, False).z > z_max - 1e-6 for v in f.verts)]
                fix_side_uvs(bm, uvl, to_frame, tops, [f for f in bm.faces if f not in set(tops)])
            bm.faces.ensure_lookup_table()
            dm = bpy.data.meshes.new(f"{src.name}_Deska")
            bm.to_mesh(dm); bm.free()
            for mat in me.materials:
                dm.materials.append(mat)
            for poly in dm.polygons:
                poly.use_smooth = False
            # origin w srodku deski (Bounds Center), jak oczekuje etap 1
            vs = [v.co for v in dm.vertices]
            c = Vector([(min(v[k] for v in vs) + max(v[k] for v in vs)) / 2 for k in range(3)])
            dm.transform(Matrix.Translation(-c))
            count += 1
            ob = bpy.data.objects.new(f"{src.name}_Deska_{count:03d}", dm)
            ob.matrix_world = mw @ Matrix.Translation(c)
            out.objects.link(ob)
            if dots:
                loc = self.board_nails(key, rects[key], faces_uv[key], dots, axis, m_per_uv, alo, ahi)
                if loc:
                    ob["deski_gwozdzie_tex"] = [x for q in loc for x in (q - c)]   # lokalnie, na wierzchu deski
                    n_nails += len(loc)
        return count, n_nails

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
class DeskiPrepSettings(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(name="Plaszczyzny", type=bpy.types.Collection,
                                      description="Kolekcja z plaszczyznami podlogi (z UV i tekstura desek)")
    texture: bpy.props.PointerProperty(name="Tekstura (opcj.)", type=bpy.types.Image,
                                       description="Obraz do wykrycia szczelin. Puste = z materialu (AO/Cavity, potem Base Color)")
    output_name: bpy.props.StringProperty(name="Kolekcja wynikowa", default="",
                                          description="Puste = '<kolekcja> - deski'")
    sensitivity: bpy.props.FloatProperty(name="Czulosc (odchylenia)", default=4.0, min=1.0, max=20.0,
                                         description="Jak wyrazna musi byc ciemna linia, by uznac ja za szczeline. "
                                                     "Mniej = wiecej szczelin")
    min_width: bpy.props.FloatProperty(name="Min. szerokosc deski", default=0.015, min=0.001, max=0.5,
                                       subtype='FACTOR', description="Jako ulamek szerokosci tekstury")
    min_length: bpy.props.FloatProperty(name="Min. dlugosc deski", default=0.05, min=0.001, max=1.0,
                                        subtype='FACTOR', description="Jako ulamek dlugosci tekstury")
    joint_contrast: bpy.props.FloatProperty(name="Laczenia: kontrast", default=0.3, min=0.05, max=1.0,
                                            description="Jak ciemna (wzgledem szczeliny miedzy deskami) musi byc "
                                                        "poprzeczna linia konca deski. Mniej = wiecej laczen")
    joint_cover: bpy.props.FloatProperty(name="Laczenia: pokrycie", default=0.6, min=0.1, max=1.0, subtype='FACTOR',
                                         description="Jaka czesc szerokosci deski musi przecinac linia konca deski")
    max_length: bpy.props.FloatProperty(name="Maks. dlugosc deski", default=2.5, min=0.0, unit='LENGTH',
                                        description="Deski dluzsze niz tyle sa dzielone w proporcji 40/60 "
                                                    "(naprzemiennie w sasiednich pasach). 0 = bez podzialu")
    thickness: bpy.props.FloatProperty(name="Grubosc", default=0.012, min=0.0, unit='LENGTH',
                                       description="Deska wyciagana w dol od plaszczyzny (gora zostaje na miejscu)")
    gap: bpy.props.FloatProperty(name="Szczelina", default=0.001, min=0.0, unit='LENGTH',
                                 description="Odstep miedzy sasiednimi deskami")
    chamfer: bpy.props.FloatProperty(name="Faza gornej krawedzi", default=0.002, min=0.0, unit='LENGTH')
    min_piece: bpy.props.FloatProperty(name="Min. kawalek", default=0.02, min=0.0, unit='LENGTH',
                                       description="Pomijaj drzazgi wezsze niz tyle (przy brzegach plaszczyzny)")
    hide_source: bpy.props.BoolProperty(name="Ukryj plaszczyzne zrodlowa", default=True)
    detect_nails: bpy.props.BoolProperty(name="Wykrywaj gwozdzie", default=True,
                                         description="Lebki gwozdzi z tekstury koloru (ciemne, chlodne punkty parami przy "
                                                     "krawedziach deski) - uzywane przy wyrywaniu i w detalach")
    nail_texture: bpy.props.PointerProperty(name="Tekstura gwozdzi (opcj.)", type=bpy.types.Image,
                                            description="Puste = Base Color z materialu")
    nail_dark: bpy.props.FloatProperty(name="Gwozdzie: kontrast", default=0.08, min=0.01, max=0.5,
                                       description="O ile lebek musi byc ciemniejszy od drewna obok. Mniej = wiecej gwozdzi")


class DESKI_PT_prep(bpy.types.Panel):
    bl_label = "Przygotowanie desek"
    bl_idname = "DESKI_PT_prep"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Deski"
    bl_order = 0

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = self.layout.column()
        col.prop(p, "source"); col.prop(p, "texture"); col.prop(p, "output_name")
        box = col.box(); box.label(text="Wykrywanie szczelin")
        box.prop(p, "sensitivity"); box.prop(p, "min_width"); box.prop(p, "min_length")
        box.prop(p, "joint_contrast"); box.prop(p, "joint_cover")
        box = col.box(); box.prop(p, "detect_nails")
        sub = box.column(); sub.enabled = p.detect_nails
        sub.prop(p, "nail_texture"); sub.prop(p, "nail_dark")
        box = col.box(); box.label(text="Deski")
        box.prop(p, "max_length"); box.prop(p, "thickness"); box.prop(p, "gap"); box.prop(p, "chamfer"); box.prop(p, "min_piece")
        col.prop(p, "hide_source")
        col.separator()
        col.operator(DESKI_OT_split.bl_idname, icon='MOD_EDGESPLIT')


classes = (DeskiPrepSettings, DESKI_OT_split, DESKI_PT_prep)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_przygotowanie = bpy.props.PointerProperty(type=DeskiPrepSettings)


def unregister():
    del bpy.types.Scene.deski_przygotowanie
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
