"""Remove the gripper from the iiwa photo (bare media flange, as during the dataset recordings).

Usage: python remove_gripper.py photos/iiwa_real_original.jpg photos/iiwa_real_bare_flange.png

The geometry below (mask region, panel edges, flange ellipse) is specific to
that photo (1932 x 2576).  Steps: (1) mask the gripper (dark connected component + hole fill, dilated);
(2) fill each masked column by vertical interpolation between the photo just
above and just below the mask (continues the wall, the blue pipe, the conduit
and the pilaster edge); (3) rebuild the electrical panel under the finger tip
(lid band and top edge from the visible lid profile, paper texture copied
from the clean band below the handwriting, top-right screw copied from the
top-left one); (4) behind the adapter,
continue the background above/below the flange face and draw the bare
ISO 9409-1-A50 face as an ellipse fitted to the visible rim of the flange.
"""
import numpy as np, cv2, sys
src_path = sys.argv[1] if len(sys.argv) > 1 else 'iiwa_real_orig.jpg'
out_path = sys.argv[2] if len(sys.argv) > 2 else 'iiwa_real_noGripper.png'
rng = np.random.default_rng(0)
orig = cv2.imread(src_path)
im = orig.astype(np.float32)
H, W = im.shape[:2]

# ---- (1) mask
v = orig.max(axis=2).astype(int)
dark = np.zeros((H, W), np.uint8)
dark[250:560, 470:1110] = (v[250:560, 470:1110] < 95)
dark[470:, 1040:] = 0                       # the flange cable below the adapter is not part of the gripper
n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
m = (lab == k).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
ff = m.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1); m = m | (1 - ff)
M = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
M[466:481, 1034:1064] = True                # the adapter's soft lower edge on the blue pipe

# ---- (2) column-wise vertical interpolation
out = im.copy()
for x in range(W):
    col = M[:, x]
    if not col.any():
        continue
    y = 0
    while y < H:
        if not col[y]:
            y += 1; continue
        y1 = y
        while y < H and col[y]:
            y += 1
        y2 = y - 1
        a = im[max(y1 - 9, 0):y1 - 3, x].mean(0); b = im[y2 + 4:min(y2 + 10, H), x].mean(0)
        t = ((np.arange(y1, y2 + 1) - y1 + 1) / (y2 - y1 + 2))[:, None]
        out[y1:y2 + 1, x] = a * (1 - t) + b * t

# ---- (3) electrical panel under the finger tip
Y_TOP, Y_PAPER, PX0, PX1, PR = 480, 504, 476, 664, 705
lid_prof = np.median(im[Y_TOP - 6:Y_PAPER + 3, 450:466], axis=1)
lid_col = np.median(im[565:600, PX1 + 6:PR - 4].reshape(-1, 3), axis=0)
D = 72                                       # clean paper band below the handwriting
for x in range(470, PR + 3):
    rows = np.nonzero(M[:, x])[0]
    if len(rows) == 0:
        continue
    y1 = rows.min()
    if y1 < Y_TOP - 6:                       # wall above the lid edge: colour above the mask only
        out[y1:Y_TOP - 6, x] = im[max(y1 - 9, 0):y1 - 3, x].mean(0)
    y2 = rows.max()
    if PX0 <= x <= PX1:                      # brightness of the copied paper matched to the paper just below the mask
        ref = im[y2 + 4:y2 + 14, x].mean(0)
        srcv = im[y2 + 4 + D:y2 + 14 + D, x].mean(0) if x < 600 else im[576:600, x].mean(0)
        off = ref - srcv
    for y in rows[rows >= Y_TOP - 6]:
        if y <= Y_PAPER + 2:
            out[y, x] = lid_prof[y - (Y_TOP - 6)]
        elif PX0 <= x <= PX1:
            out[y, x] = (im[y + D, x] if x < 600 else im[576 + (y - 505) % 24, x]) + off
        elif x <= PR:
            out[y, x] = lid_col
edge = np.median(im[Y_PAPER - 2:Y_PAPER + 3, 478:495], axis=1)
for x in range(PX0, PX1 + 1):
    if M[Y_PAPER, x]:
        out[Y_PAPER - 2:Y_PAPER + 3, x] = edge
seam = np.median(im[548:556, PR - 3:PR + 3], axis=0)
for y in range(Y_TOP, 560):
    if M[y, PR]:
        out[y, PR - 3:PR + 3] = seam
# top-right screw = copy of the top-left one
s = im[480:501, 438:459]
lid_r = np.median(im[483:500, 600:640].reshape(-1, 3), axis=0); lid_l = np.median(im[483:500, 462:468].reshape(-1, 3), axis=0)
yy, xx = np.mgrid[-10:11, -10:11]; w = np.clip(1.5 - np.sqrt(xx ** 2 + yy ** 2) / 7, 0, 1)[..., None]
w = w * 0.8
out[480:501, 674:695] = out[480:501, 674:695] * (1 - w) + (s - lid_l + lid_r) * w

# grain + soft transition at the mask edge
out[M] += rng.normal(0, 1.6, (int(M.sum()), 1)).astype(np.float32)
out = np.clip(out, 0, 255)
blur = cv2.GaussianBlur(out, (0, 0), 0.8)
Ms = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.5)[..., None]
res = out * (1 - Ms) + blur * Ms
res[~M] = im[~M]
res = np.clip(res, 0, 255).astype(np.uint8)
img = res.astype(np.float32)

# ---- (4) flange: background around the face, then the bare face
c = np.array([1066.0, 381.0]); u = np.array([0.19, 0.982]); u /= np.linalg.norm(u); vv = np.array([u[1], -u[0]])
A, B = 95.0, 30.0
gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
du = (gx - c[0]) * u[0] + (gy - c[1]) * u[1]; dv = (gx - c[0]) * vv[0] + (gy - c[1]) * vv[1]
E = np.sqrt((du / A) ** 2 + (dv / B) ** 2) <= 1.0
WALL = np.median(im[495:515, 1057:1062].reshape(-1, 3), axis=0)
for x in range(980, 1110):
    rows = np.nonzero(M[:, x])[0]
    ein = np.nonzero(E[:, x])[0]
    if len(rows) == 0 or len(ein) == 0:
        continue
    ytop, ybot = ein.min(), ein.max(); y1, y2 = rows.min(), rows.max()
    above = im[max(y1 - 8, 0):y1 - 3, x].mean(0); below = im[y2 + 4:y2 + 9, x].mean(0)
    if below.max() < 90:
        below = WALL
    for y in rows:
        if y < ytop: img[y, x] = above
        elif y > ybot: img[y, x] = below
S = 4
x0, x1, y0, y1 = int(c[0] - 110), int(c[0] + 110), int(c[1] - 110), int(c[1] + 110)
h, w = (y1 - y0) * S, (x1 - x0) * S
qy, qx = np.mgrid[0:h, 0:w].astype(np.float32)
X = x0 + (qx + 0.5) / S - 0.5; Y = y0 + (qy + 0.5) / S - 0.5
DU = (X - c[0]) * u[0] + (Y - c[1]) * u[1]; DV = (X - c[0]) * vv[0] + (Y - c[1]) * vv[1]
R = np.sqrt((DU / A) ** 2 + (DV / B) ** 2)
t = DU / A
base = 156 - 26 * t - 6 * (DV / B)
face = np.stack([base * 0.93, base * 0.965, base], -1)
for (r0, r1, val, al) in [(0.95, 1.01, 118, 0.7), (0.91, 0.95, 200, 0.35), (0.58, 0.61, 150, 0.25)]:
    mm = ((R >= r0) & (R < r1)).astype(np.float32)[..., None] * al
    face = face * (1 - mm) + val * mm
Rb = 0.26
mb = (R < Rb).astype(np.float32)[..., None]
face = face * (1 - mb) + np.clip(np.stack([100 + 35 * (DU / A / Rb)] * 3, -1), 40, 120) * mb
mb2 = ((R >= Rb) & (R < Rb + 0.03)).astype(np.float32)[..., None]
face = face * (1 - mb2 * 0.4) + 190 * mb2 * 0.4
for kk in range(8):
    ang = 2 * np.pi * kk / 8 + 0.2
    cu, cv_ = 0.5 * np.cos(ang), 0.5 * np.sin(ang)
    rr = np.sqrt(((DU / A - cu) / 0.04) ** 2 + ((DV / B - cv_) / 0.04) ** 2)
    mh = np.clip(((0.75 if kk == 3 else 1.0) - rr) * 3, 0, 1)[..., None]
    face = face * (1 - mh * 0.8) + 80 * mh * 0.8
alpha = np.clip((1.0 - R) * A * S / 3, 0, 1)[..., None]
face = cv2.GaussianBlur(face, (0, 0), 2.5) + rng.normal(0, 1.2, face.shape)
face = cv2.resize(face, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
alpha = cv2.resize(alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)[..., None]
img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + face * alpha
cv2.imwrite(out_path, np.clip(img, 0, 255).astype(np.uint8))
print('wrote', out_path)
