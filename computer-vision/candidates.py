import cv2
import numpy as np
from collections import deque, Counter

def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]        # TL
    rect[2] = pts[np.argmax(s)]        # BR
    rect[1] = pts[np.argmin(diff)]     # TR
    rect[3] = pts[np.argmax(diff)]     # BL
    return rect

def warp_quad(frame, quad, out_w=360, out_h=240):
    rect = order_points(quad)
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)
    warp = cv2.warpPerspective(frame, M, (out_w, out_h))
    return warp, Minv

def ratio_white(hsv):
    H, S, V = cv2.split(hsv)
    m = (S < 70) & (V > 140)
    return float(np.count_nonzero(m)) / m.size


def ratio_red(hsv):
    m1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 50, 40), (179, 255, 255))
    m = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(m)) / m.size


def ratio_blue(hsv):
    m = cv2.inRange(hsv, (75, 40, 35), (140, 255, 255))
    return float(np.count_nonzero(m)) / m.size

def ratio_yellow(hsv):
    H, S, V = cv2.split(hsv)

    m1 = cv2.inRange(hsv, (12, 35, 35), (60, 255, 255))

    m2 = ((H >= 12) & (H <= 60) & (V > 170) & (S > 10)).astype(np.uint8) * 255

    m = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(m)) / m.size

def classify_flag(warp_bgr, thr=0.30):
    warp_bgr = cv2.resize(warp_bgr, (360, 240), interpolation=cv2.INTER_AREA)
    warp_bgr = cv2.GaussianBlur(warp_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]

    top2 = hsv[:h//2, :, :]
    bot2 = hsv[h//2:, :, :]

    score_pl = min(ratio_white(top2), ratio_red(bot2))
    score_ua = min(ratio_blue(top2), ratio_yellow(bot2))

    a = hsv[:h//3, :, :]
    b = hsv[h//3:2*h//3, :, :]
    c = hsv[2*h//3:, :, :]

    score_ru = min(ratio_white(a), ratio_blue(b), ratio_red(c))

    scores = {"POLAND": score_pl, "UKRAINE": score_ua, "RUSSIA": score_ru}
    label = max(scores, key=scores.get)
    best = scores[label]

    if best < thr:
        return None, best, scores
    return label, best, scores

def find_quads(frame, max_quads=12):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 70, 170)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.01 * (W * H):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(approx)
        rw, rh = rect[1]
        if rw < 60 or rh < 60:
            continue
        ar = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ar < 1.2 or ar > 2.2:
            continue

        quads.append((area, approx))

    quads.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in quads[:max_quads]]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    label_hist = deque(maxlen=8)

    print("Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        quads = find_quads(frame)

        best = None

        for quad in quads:
            warp, Minv = warp_quad(frame, quad, out_w=360, out_h=240)
            label, score, scores = classify_flag(warp, thr=0.42)
            if label is None:
                continue

            corners = np.array([[0,0],[359,0],[359,239],[0,239]], dtype=np.float32).reshape(-1,1,2)
            mapped = cv2.perspectiveTransform(corners, Minv)
            mapped_i = np.int32(mapped)

            if best is None or score > best[0]:
                best = (score, label, quad, mapped_i)

        if best is not None:
            score, label, quad, poly = best
            label_hist.append(label)
            stable = Counter(label_hist).most_common(1)[0][0]

            x, y, w, h = cv2.boundingRect(poly)
            cx, cy = x + w//2, y + h//2

            cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"{stable} score={score:.2f}", (x, max(25, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"pos: x={x} y={y} w={w} h={h}", (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            label_hist.append("NONE")

        cv2.imshow("FLAG DETECTOR", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
