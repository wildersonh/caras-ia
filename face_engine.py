import os
import cv2
import numpy as np
import datetime
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

def warp_local(map_x, map_y, px, py, dx, dy, radius):
    """Aplica una deformación radial suave centrada en (px, py) con desplazamiento (dx, dy)"""
    h, w = map_x.shape
    x1 = max(0, int(px - radius))
    x2 = min(w, int(px + radius + 1))
    y1 = max(0, int(py - radius))
    y2 = min(h, int(py + radius + 1))
    
    if x1 >= x2 or y1 >= y2:
        return
        
    sub_x = map_x[y1:y2, x1:x2]
    sub_y = map_y[y1:y2, x1:x2]
    
    dx_grid = sub_x - px
    dy_grid = sub_y - py
    d2 = dx_grid * dx_grid + dy_grid * dy_grid
    r2 = radius * radius
    
    mask = d2 < r2
    factor = np.zeros_like(d2)
    factor[mask] = (1.0 - d2[mask] / r2) ** 2
    
    map_x[y1:y2, x1:x2] -= dx * factor
    map_y[y1:y2, x1:x2] -= dy * factor

class FaceAnalyzer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.landmarker_path = os.path.join(models_dir, "face_landmarker.task")
        self.gender_path = os.path.join(models_dir, "genderage.onnx")
        
        if not os.path.exists(self.landmarker_path):
            raise FileNotFoundError(f"No se encontró el modelo de landmarks en: {self.landmarker_path}")
        if not os.path.exists(self.gender_path):
            raise FileNotFoundError(f"No se encontró el modelo de género en: {self.gender_path}")
            
        base_options = python.BaseOptions(model_asset_path=self.landmarker_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.gender_session = ort.InferenceSession(
            self.gender_path, 
            providers=['CPUExecutionProvider']
        )
        
        self.tesselation = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    def analyze_image(self, img_bgr, filter_mode="HUD"):
        """
        Analiza la imagen y aplica el filtro visual seleccionado:
        - "NONE": Sin ningún filtro (Foto original limpia al 100%)
        - "MEWING": Filtro viral Mewing / Chad (ceja levantada estilo The Rock, mandíbula afilada, mejillas hundidas)
        - "RGB": Desfase cromático RGB Glitch / Split anaglifo 3D
        - "SNIPER": Mira de Rifle de Francotirador Roja (mil-dot reticle)
        - "NIGHT_VISION": Visión Nocturna Militar NVG (Green Phosphor)
        - "THERMAL": Termografía FLIR / Predator Heatmap
        - "TERMINATOR": Visor Ciborg Terminator T-800
        - "FACE_MESH": Malla Holográfica 3D Wireframe (478 vértices)
        - "CCTV": Cámara de Seguridad Vintage 90s con Timestamp
        - "HUD": HUD Cyberpunk Táctico estándar
        """
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.landmarker.detect(mp_img)
        
        annotated_img = img_bgr.copy()
        faces_results = []
        
        # 1. Si no hay rostros detectados
        if not detection_result.face_landmarks:
            if filter_mode == "NONE":
                return annotated_img, []
            elif filter_mode == "RGB":
                return self._apply_rgb_split_effect(img_bgr), []
            elif filter_mode == "SNIPER":
                return self._apply_sniper_scope_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "NIGHT_VISION":
                return self._apply_night_vision_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "THERMAL":
                return self._apply_thermal_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "TERMINATOR":
                return self._apply_terminator_effect(img_bgr, None, w, h), []
            elif filter_mode == "CCTV":
                return self._apply_cctv_effect(img_bgr, None, w, h), []
            else:
                self._draw_hud_outer_frame(annotated_img, w, h)
                cv2.putText(annotated_img, "STATUS: SEARCHING TARGET...", (30, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57, 255, 20), 2, cv2.LINE_AA)
                return annotated_img, []

        # 2. Procesar cada rostro detectado (Extracción de atributos)
        for face_idx, lms in enumerate(detection_result.face_landmarks):
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
            y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))
            
            margin_x = int((x_max - x_min) * 0.15)
            margin_y = int((y_max - y_min) * 0.15)
            crop_x1 = max(0, x_min - margin_x)
            crop_x2 = min(w, x_max + margin_x)
            crop_y1 = max(0, y_min - margin_y)
            crop_y2 = min(h, y_max + margin_y)
            
            face_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
            gender_info = self._predict_gender_and_age(face_crop)
            skin_info = self._analyze_skin_tone(img_rgb, lms, w, h)
            eye_info = self._analyze_eye_color(img_rgb, lms, w, h)
            
            face_data = {
                "face_id": face_idx + 1,
                "bbox": (x_min, y_min, x_max, y_max),
                "gender": gender_info["gender"],
                "gender_confidence": gender_info["confidence"],
                "estimated_age": gender_info["age"],
                "skin_tone": skin_info["name"],
                "skin_rgb": skin_info["rgb"],
                "skin_hex": skin_info["hex"],
                "eye_color": eye_info["name"],
                "eye_rgb": eye_info["rgb"],
                "eye_hex": eye_info["hex"],
                "eye_confidence": eye_info["confidence"]
            }
            faces_results.append(face_data)

        # 3. Aplicar el filtro visual seleccionado
        main_face = faces_results[0]
        x1, y1, x2, y2 = main_face["bbox"]
        target_center = ((x1 + x2) // 2, int(y1 + (y2 - y1) * 0.42))
        main_lms = detection_result.face_landmarks[0]

        if filter_mode == "NONE":
            # Foto original limpia sin añadidos
            annotated_img = img_bgr.copy()
        elif filter_mode == "MEWING":
            # Filtro viral idéntico a la referencia (Ceja levantada The Rock + Mandíbula Gigachad)
            annotated_img = self._apply_mewing_effect(img_bgr, main_lms, main_face, w, h)
        elif filter_mode == "RGB":
            # Filtro RGB Chromatic Glitch Split
            annotated_img = self._apply_rgb_split_effect(img_bgr)
        elif filter_mode == "SNIPER":
            annotated_img = self._apply_sniper_scope_effect(img_bgr, center=target_center)
        elif filter_mode == "NIGHT_VISION":
            annotated_img = self._apply_night_vision_effect(img_bgr, center=target_center)
        elif filter_mode == "THERMAL":
            annotated_img = self._apply_thermal_effect(img_bgr, center=target_center)
        elif filter_mode == "TERMINATOR":
            annotated_img = self._apply_terminator_effect(img_bgr, main_face, w, h)
        elif filter_mode == "FACE_MESH":
            annotated_img = self._apply_face_mesh_effect(img_bgr, main_lms, main_face, w, h)
        elif filter_mode == "CCTV":
            annotated_img = self._apply_cctv_effect(img_bgr, main_face, w, h)
        else: # "HUD" por defecto
            self._draw_hud_outer_frame(annotated_img, w, h)
            for idx, f in enumerate(faces_results):
                lms = detection_result.face_landmarks[idx]
                fx1, fy1, fx2, fy2 = f["bbox"]
                crop = img_bgr[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                self._draw_hud_face_target(
                    annotated_img, lms, w, h, 
                    f["bbox"], 
                    {"gender": f["gender"], "confidence": f["gender_confidence"], "age": f["estimated_age"]}, 
                    {"name": f["skin_tone"]}, 
                    {"name": f["eye_color"]}, 
                    crop, idx
                )

        return annotated_img, faces_results

    # =========================================================================
    # FILTRO VIRAL MEWING / CHAD (CEJA THE ROCK + MANDÍBULA GIGACHAD)
    # =========================================================================
    def _apply_mewing_effect(self, img_bgr, lms, face_data, w, h):
        """
        Deformación facial idéntica al filtro viral de TikTok / Instagram:
        1. Ceja izquierda arqueada e hiper-elevada hacia arriba estilo The Rock / Chad.
        2. Ceja y ojo derechos entrecerrados (squint / hunter eyes).
        3. Hundimiento profundo de mejillas (Hollow Cheeks / Buccal pinch).
        4. Ángulos mandibulares anchos hacia afuera (Gonial Jaw Flare).
        5. Mentón largo, ancho y esculpido (Gigachad Chin).
        6. Sombreado de contorno en mejillas para crear profundidad 3D y arruga en la frente.
        7. Fruncimiento de labios estilo Mewing Smirk.
        """
        if lms is None:
            return img_bgr.copy()

        face_w = abs(lms[454].x - lms[234].x) * w
        face_h = abs(lms[152].y - lms[10].y) * h
        if face_w < 10 or face_h < 10:
            return img_bgr.copy()

        src_pts = []
        dst_pts = []

        # 1. ANCLAJES PERIMETRALES (Fondo y bordes fijos sin distorsión)
        for bx in np.linspace(0, w - 1, 8):
            for by in np.linspace(0, h - 1, 8):
                dist = np.hypot(bx - lms[6].x * w, by - lms[6].y * h)
                if dist > max(face_w, face_h) * 0.85:
                    src_pts.append([bx, by])
                    dst_pts.append([bx, by])

        # 2. ANCLAJES SUPERIORES DEL CRÁNEO / NACIMIENTO DEL CABELLO
        top_y = lms[10].y * h
        for idx in [10, 338, 297, 332, 109, 67, 103]:
            px, py = lms[idx].x * w, lms[idx].y * h
            src_pts.append([px, py])
            dst_pts.append([px, py])

        # 3. ANCLAJES DEL PUENTE NASAL
        for idx in [168, 6, 197, 195, 5, 4]:
            px, py = lms[idx].x * w, lms[idx].y * h
            src_pts.append([px, py])
            dst_pts.append([px, py])

        # Espacio disponible en la frente para elevar la ceja
        eyebrow_y = lms[105].y * h
        headroom = max(15, eyebrow_y - top_y)
        brow_lift = min(face_h * 0.30, headroom * 0.80)

        # 4. LA MÍTICA CEJA LEVANTADA ESTILO THE ROCK (Espectador Izquierda)
        brow_moves = {
            105: (0, -brow_lift),                          # Pico central arqueado arriba
            66: (0, -brow_lift * 0.90),                   # Parte media-interna
            63: (0, -brow_lift * 0.80),                   # Parte media-externa
            107: (face_w * 0.02, -brow_lift * 0.40),       # Extremo interno fruncido
            70: (-face_w * 0.02, -brow_lift * 0.25),      # Cola externa inclinada
            52: (0, -brow_lift * 0.85),                   # Borde inferior de ceja
            53: (0, -brow_lift * 0.75),
            65: (0, -brow_lift * 0.85),
            159: (0, -brow_lift * 0.20),                  # Párpado abierto de sorpresa
        }

        # 5. OJO Y CEJA ENTRECERRADOS / HUNTER EYE (Espectador Derecha)
        hunter_moves = {
            334: (0, face_h * 0.04),                      # Ceja aplanada y baja
            296: (0, face_h * 0.05),
            293: (0, face_h * 0.03),
            386: (0, face_h * 0.05),                      # Párpado superior cerrado
            374: (0, -face_h * 0.03),                     # Párpado inferior tenso
        }

        # 6. MEJILLAS PROFUNDAMENTE HUNDIDAS (HOLLOW CHEEKS)
        pinch = face_w * 0.18
        cheek_moves = {
            116: (pinch, -face_h * 0.02),                 # Mejilla izquierda hacia el centro
            205: (pinch * 1.1, 0),
            147: (pinch * 0.8, face_h * 0.02),
            345: (-pinch, -face_h * 0.02),                # Mejilla derecha hacia el centro
            425: (-pinch * 1.1, 0),
            376: (-pinch * 0.8, face_h * 0.02),
        }

        # 7. PÓMULOS ALTOS Y ESCULPIDOS
        bone_moves = {
            123: (-face_w * 0.06, -face_h * 0.02),
            352: (face_w * 0.06, -face_h * 0.02),
        }

        # 8. MANDÍBULA AFILADA Y MENTÓN GIGACHAD
        chin_drop = min(face_h * 0.22, max(10, (h - lms[152].y * h) * 0.6))
        jaw_moves = {
            172: (-face_w * 0.09, -face_h * 0.02),        # Ángulo gonial izquierdo afuera
            397: (face_w * 0.09, -face_h * 0.02),         # Ángulo gonial derecho afuera
            152: (0, chin_drop),                          # Mentón hacia abajo
            175: (0, chin_drop * 0.90),
            199: (0, chin_drop * 0.80),
            148: (-face_w * 0.04, chin_drop * 0.80),      # Mentón ancho / cuadrado
            377: (face_w * 0.04, chin_drop * 0.80),
        }

        # 9. LABIOS FRUNCIDOS ESTILO MEWING POUT / SMIRK
        mouth_moves = {
            61: (face_w * 0.02, -face_h * 0.02),
            291: (-face_w * 0.02, -face_h * 0.02),
            0: (0, -face_h * 0.02),
            17: (0, -face_h * 0.03),
        }

        all_moves = {**brow_moves, **hunter_moves, **cheek_moves, **bone_moves, **jaw_moves, **mouth_moves}
        for idx, (dx, dy) in all_moves.items():
            px, py = lms[idx].x * w, lms[idx].y * h
            src_pts.append([px, py])
            dst_pts.append([px + dx, py + dy])

        src_arr = np.array([src_pts], dtype=np.float32)
        dst_arr = np.array([dst_pts], dtype=np.float32)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]

        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(src_arr, dst_arr, matches)
        warped = tps.warpImage(img_bgr)

        # 10. SOMBREADO DE CONTORNO EN LAS MEJILLAS Y ARRUGA EN LA FRENTE
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        c_pt1_l = (int(lms[123].x * w), int(lms[123].y * h + face_h * 0.05))
        c_pt2_l = (int(lms[61].x * w - face_w * 0.03), int(lms[61].y * h))
        cv2.line(shadow_mask, c_pt1_l, c_pt2_l, 0.70, int(face_w * 0.16))

        c_pt1_r = (int(lms[352].x * w), int(lms[352].y * h + face_h * 0.05))
        c_pt2_r = (int(lms[291].x * w + face_w * 0.03), int(lms[291].y * h))
        cv2.line(shadow_mask, c_pt1_r, c_pt2_r, 0.70, int(face_w * 0.16))

        # Arruga vertical de la frente por levantar la ceja
        f1 = (int(lms[107].x * w + face_w * 0.02), int(lms[107].y * h - brow_lift * 0.2))
        f2 = (int(lms[107].x * w + face_w * 0.02), int(lms[107].y * h - brow_lift * 0.8))
        cv2.line(shadow_mask, f1, f2, 0.55, max(2, int(face_w * 0.025)))

        blur_k = max(21, (int(face_w * 0.16) // 2) * 2 + 1)
        shadow_blurred = cv2.GaussianBlur(shadow_mask, (blur_k, blur_k), 0)
        shadow_factor = 1.0 - (shadow_blurred[:, :, np.newaxis] * 0.65)

        # Iluminaciones en pómulos y mentón esculpido
        highlight_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(highlight_mask, (int(lms[123].x * w), int(lms[123].y * h)), int(face_w * 0.07), 0.35, -1)
        cv2.circle(highlight_mask, (int(lms[352].x * w), int(lms[352].y * h)), int(face_w * 0.07), 0.35, -1)
        cv2.circle(highlight_mask, (int(lms[152].x * w), int(lms[152].y * h + chin_drop * 0.5)), int(face_w * 0.09), 0.35, -1)
        hl_blur = cv2.GaussianBlur(highlight_mask, (blur_k, blur_k), 0)

        result = warped.astype(np.float32) * shadow_factor + (hl_blur[:, :, np.newaxis] * 40.0)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Banner discreto de telemetría del filtro
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(result, "MEWING FILTER // THE ROCK EYEBROW // GIGACHAD JAW", (15, h - 15), font, scale, (245, 114, 182), 1, cv2.LINE_AA)
        return result

    # =========================================================================
    # FILTRO RGB CHROMA GLITCH SPLIT
    # =========================================================================
    def _apply_rgb_split_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        offset = max(5, int(w * 0.016))
        
        b, g, r = cv2.split(img_bgr)
        
        r_shifted = np.zeros_like(r)
        r_shifted[:, offset:] = r[:, :-offset]
        r_shifted[:, :offset] = r[:, 0:1]
        
        b_shifted = np.zeros_like(b)
        b_shifted[:, :-offset] = b[:, offset:]
        b_shifted[:, -offset:] = b[:, -1:]
        
        rgb_merged = cv2.merge([b_shifted, g, r_shifted])
        rgb_merged[::4, :] = (rgb_merged[::4, :].astype(np.float32) * 0.88).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.38, min(w, h) / 1600.0)
        
        cv2.putText(rgb_merged, "RGB CHROMA SPLIT // 3D ANAGLYPH", (16, 28), font, scale, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb_merged, "RGB CHROMA SPLIT // 3D ANAGLYPH", (14, 26), font, scale, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(rgb_merged, "RGB CHROMA SPLIT // 3D ANAGLYPH", (15, 27), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb_merged, "CHANNEL DISPLACEMENT: ACTIVE", (15, h - 15), font, scale * 0.85, (255, 200, 100), 1, cv2.LINE_AA)
        return rgb_merged

    # =========================================================================
    # FILTRO MIRA DE RIFLE (SNIPER RED SCOPE)
    # =========================================================================
    def _apply_sniper_scope_effect(self, img_bgr, center=None):
        h, w, _ = img_bgr.shape
        cx, cy = (w // 2, h // 2) if center is None else center
        radius = int(min(w, h) * 0.47)
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        r_ch = np.clip(gray.astype(np.float32) * 1.35 + 30, 0, 255).astype(np.uint8)
        g_ch = np.clip(gray.astype(np.float32) * 0.20 - 5, 0, 255).astype(np.uint8)
        b_ch = np.clip(gray.astype(np.float32) * 0.10 - 10, 0, 255).astype(np.uint8)
        red_tinted = cv2.merge([b_ch, g_ch, r_ch])
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
        blur_k = max(15, (radius // 18) * 2 + 1)
        mask_blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        output = (red_tinted.astype(np.float32) * mask_blurred[:, :, np.newaxis]).astype(np.uint8)
        
        cv2.circle(output, (cx, cy), radius, (40, 20, 100), 2, cv2.LINE_AA)
        
        c_reticle = (245, 205, 155)
        arm_len = int(radius * 0.90)
        gap = max(4, int(radius * 0.04))
        
        cv2.line(output, (cx - arm_len, cy), (cx - gap, cy), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx + gap, cy), (cx + arm_len, cy), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy - arm_len), (cx, cy - gap), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy + gap), (cx, cy + arm_len), c_reticle, 1, cv2.LINE_AA)
        
        cv2.line(output, (cx - 2, cy), (cx + 2, cy), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy - 2), (cx, cy + 2), c_reticle, 1, cv2.LINE_AA)
        
        tick_step = max(8, int(radius * 0.08))
        tick_len = max(5, int(radius * 0.045))
        for i in range(1, 6):
            dist = gap + i * tick_step
            if dist < arm_len - 5:
                t_h = tick_len if i % 2 == 1 else int(tick_len * 0.65)
                cv2.line(output, (cx - dist, cy - t_h), (cx - dist, cy + t_h), c_reticle, 1, cv2.LINE_AA)
                cv2.line(output, (cx + dist, cy - t_h), (cx + dist, cy + t_h), c_reticle, 1, cv2.LINE_AA)
                t_w = tick_len if i % 2 == 1 else int(tick_len * 0.65)
                cv2.line(output, (cx - t_w, cy - dist), (cx + t_w, cy - dist), c_reticle, 1, cv2.LINE_AA)
                cv2.line(output, (cx - t_w, cy + dist), (cx + t_w, cy + dist), c_reticle, 1, cv2.LINE_AA)
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(w, h) / 1800.0)
        cv2.putText(output, "OPTIC: 12x THERMAL // RETICLE: MIL-DOT", (15, h - 15), font, scale, (140, 160, 180), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # FILTRO VISIÓN NOCTURNA (NVG)
    # =========================================================================
    def _apply_night_vision_effect(self, img_bgr, center=None):
        h, w, _ = img_bgr.shape
        cx, cy = (w // 2, h // 2) if center is None else center
        radius = int(min(w, h) * 0.47)
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g_ch = np.clip(gray.astype(np.float32) * 1.45 + 40, 0, 255).astype(np.uint8)
        b_ch = np.clip(gray.astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
        r_ch = np.clip(gray.astype(np.float32) * 0.10, 0, 255).astype(np.uint8)
        nvg_tinted = cv2.merge([b_ch, g_ch, r_ch])
        
        nvg_tinted[::3, :] = (nvg_tinted[::3, :].astype(np.float32) * 0.82).astype(np.uint8)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
        blur_k = max(15, (radius // 16) * 2 + 1)
        mask_blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        output = (nvg_tinted.astype(np.float32) * mask_blurred[:, :, np.newaxis]).astype(np.uint8)
        
        c_nvg = (80, 255, 100)
        cv2.circle(output, (cx, cy), radius, (20, 80, 30), 2, cv2.LINE_AA)
        cv2.circle(output, (cx, cy), int(radius * 0.65), (20, 100, 35), 1, cv2.LINE_AA)
        
        arm = int(radius * 0.85)
        cv2.line(output, (cx - arm, cy), (cx + arm, cy), c_nvg, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy - arm), (cx, cy + arm), c_nvg, 1, cv2.LINE_AA)
        cv2.circle(output, (cx, cy), 4, c_nvg, 1, cv2.LINE_AA)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(w, h) / 1800.0)
        cv2.putText(output, "NVG GEN-3 // PHOSPHOR P43 // GAIN: +18dB", (15, h - 15), font, scale, c_nvg, 1, cv2.LINE_AA)
        cv2.putText(output, "IR BEACON: ON", (15, 25), font, scale, c_nvg, 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # FILTRO TERMOGRAFÍA (FLIR)
    # =========================================================================
    def _apply_thermal_effect(self, img_bgr, center=None):
        h, w, _ = img_bgr.shape
        cx, cy = (w // 2, h // 2) if center is None else center
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        thermal = cv2.applyColorMap(gray_eq, cv2.COLORMAP_INFERNO)
        
        bar_w = max(14, int(w * 0.025))
        bar_h = int(h * 0.6)
        bar_x = w - bar_w - 20
        bar_y = (h - bar_h) // 2
        
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, np.newaxis]
        grad_bar = cv2.applyColorMap(np.repeat(grad, bar_w, axis=1), cv2.COLORMAP_INFERNO)
        thermal[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = grad_bar
        cv2.rectangle(thermal, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(w, h) / 1800.0)
        cv2.putText(thermal, "38.5 C", (bar_x - 55, bar_y + 12), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(thermal, "36.6 C", (bar_x - 55, bar_y + bar_h // 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(thermal, "22.0 C", (bar_x - 55, bar_y + bar_h), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        c_cross = (0, 255, 255)
        cr = max(12, int(min(w, h) * 0.05))
        cv2.line(thermal, (cx - cr, cy), (cx + cr, cy), c_cross, 1, cv2.LINE_AA)
        cv2.line(thermal, (cx, cy - cr), (cx, cy + cr), c_cross, 1, cv2.LINE_AA)
        cv2.putText(thermal, "SPOT: 36.8 C", (cx + 8, cy - 8), font, scale, c_cross, 1, cv2.LINE_AA)
        
        cv2.putText(thermal, "FLIR THERMAL IMAGING // CALIBRATED", (15, h - 15), font, scale, (200, 220, 255), 1, cv2.LINE_AA)
        return thermal

    # =========================================================================
    # FILTRO TERMINATOR T-800
    # =========================================================================
    def _apply_terminator_effect(self, img_bgr, face_data, w, h):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        term_r = cv2.convertScaleAbs(gray, alpha=1.35, beta=25)
        term_g = np.zeros_like(gray)
        term_b = np.zeros_like(gray)
        output = cv2.merge([b_ch := term_b, g_ch := term_g, term_r])
        
        output[::3, :] = (output[::3, :].astype(np.float32) * 0.65).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(w, h) / 1600.0)
        c_red = (50, 50, 255)
        c_white = (240, 240, 240)
        
        if face_data:
            x1, y1, x2, y2 = face_data["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), c_red, 2)
            cv2.line(output, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), c_red, 1)
            
            cv2.putText(output, "ANALYSIS: HUMAN SUBJECT", (x1, max(20, y1 - 25)), font, scale, c_red, 1, cv2.LINE_AA)
            cv2.putText(output, f"GENDER: {face_data['gender'].upper()}", (x1, max(35, y1 - 10)), font, scale, c_white, 1, cv2.LINE_AA)
            cv2.putText(output, "ASSESSMENT: ZERO THREAT", (x1, min(h - 10, y2 + 18)), font, scale, c_red, 1, cv2.LINE_AA)
            
        cv2.putText(output, "CYBERDYNE SYSTEMS // T-800 MODEL 101", (15, 25), font, scale, c_red, 1, cv2.LINE_AA)
        cv2.putText(output, "ROM 0x7FA4 // MATCH PROBABILITY: 99.8%", (15, h - 15), font, scale, c_red, 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # FILTRO MALLA 3D (FACE MESH)
    # =========================================================================
    def _apply_face_mesh_effect(self, img_bgr, lms, face_data, w, h):
        output = (img_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        c_wire = (255, 230, 0)
        
        for conn in self.tesselation:
            pt1 = (int(lms[conn.start].x * w), int(lms[conn.start].y * h))
            pt2 = (int(lms[conn.end].x * w), int(lms[conn.end].y * h))
            cv2.line(output, pt1, pt2, c_wire, 1, cv2.LINE_AA)
            
        for idx in [468, 473, 1, 4, 168, 61, 291, 0, 17, 152]:
            nx, ny = int(lms[idx].x * w), int(lms[idx].y * h)
            cv2.circle(output, (nx, ny), 4, (0, 255, 100), -1, cv2.LINE_AA)
            cv2.circle(output, (nx, ny), 7, c_wire, 1, cv2.LINE_AA)
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(w, h) / 1600.0)
        cv2.putText(output, "[ 3D BIOMETRIC TOPOLOGY // 478 VERTICES ]", (15, 25), font, scale, (255, 240, 0), 1, cv2.LINE_AA)
        if face_data:
            cv2.putText(output, f"TARGET: {face_data['gender'].upper()} // MESH LOCKED", (15, h - 15), font, scale, (200, 255, 100), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # FILTRO CÁMARA CCTV 90s
    # =========================================================================
    def _apply_cctv_effect(self, img_bgr, face_data, w, h):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cctv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cctv = cv2.convertScaleAbs(cctv, alpha=0.92, beta=18)
        cctv[:, :, 0] = np.clip(cctv[:, :, 0].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
        cctv[::3, :] = (cctv[::3, :].astype(np.float32) * 0.86).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.38, min(w, h) / 1500.0)
        
        cv2.circle(cctv, (25, 25), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(cctv, "REC [LIVE]", (40, 30), font, scale, (240, 240, 240), 1, cv2.LINE_AA)
        
        now_str = datetime.datetime.now().strftime("%d-%b-%Y  %H:%M:%S  EST").upper()
        cv2.putText(cctv, now_str, (15, h - 15), font, scale, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(cctv, "CAM 04 [SEC-CORRIDOR A]", (w - 240, 30), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        if face_data:
            x1, y1, x2, y2 = face_data["bbox"]
            cv2.rectangle(cctv, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(cctv, "OBJ_DETECT", (x1, max(15, y1 - 8)), font, scale * 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            
        return cctv

    # =========================================================================
    # HUD TÁCTICO ESTÁNDAR
    # =========================================================================
    def _draw_hud_outer_frame(self, img, w, h):
        c_green = (57, 255, 20)
        c_faint = (40, 70, 45)
        c_white = (240, 245, 245)
        
        pad = max(10, int(min(w, h) * 0.025))
        cut = max(14, int(min(w, h) * 0.04))
        
        step_x = max(60, w // 4)
        step_y = max(60, h // 4)
        for x in range(step_x, w, step_x):
            cv2.line(img, (x, pad), (x, h - pad), c_faint, 1, cv2.LINE_AA)
        for y in range(step_y, h, step_y):
            cv2.line(img, (pad, y), (w - pad, y), c_faint, 1, cv2.LINE_AA)
            
        pts = np.array([
            [pad + cut, pad], [w - pad - cut, pad],
            [w - pad, pad + cut], [w - pad, h - pad - cut],
            [w - pad - cut, h - pad], [pad + cut, h - pad],
            [pad, h - pad - cut], [pad, pad + cut]
        ], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=c_green, thickness=2, lineType=cv2.LINE_AA)
        
        d = cut + 10
        cv2.line(img, (pad, pad + cut), (pad, pad + d), c_green, 3)
        cv2.line(img, (pad + cut, pad), (pad + d, pad), c_green, 3)
        cv2.line(img, (w - pad, pad + cut), (w - pad, pad + d), c_green, 3)
        cv2.line(img, (w - pad - cut, pad), (w - pad - d, pad), c_green, 3)
        
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        status_txt = f"SYSTEM STATUS: SCANNING | TIME: {now_str}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(w, h) / 1600.0)
        cv2.putText(img, status_txt, (pad + cut + 15, pad + 18), font, scale, c_white, 1, cv2.LINE_AA)
        
        sig_x = w - pad - cut - 55
        sig_y = pad + 18
        for i, h_bar in enumerate([4, 7, 10, 13]):
            bx = sig_x + i * 6
            cv2.line(img, (bx, sig_y), (bx, sig_y - h_bar), c_green, 2)
            
        sub_txt = "BIO-IDENT PROTOCOL: v2.7 // TARGET ACQUIRED"
        cv2.putText(img, sub_txt, (pad + cut + 15, h - pad - 10), font, scale, (180, 220, 190), 1, cv2.LINE_AA)

    def _draw_hud_face_target(self, img, lms, w, h, bbox, gender_info, skin_info, eye_info, face_crop, idx):
        x1, y1, x2, y2 = bbox
        c_neon = (57, 255, 20)
        c_white = (255, 255, 255)
        c_dark = (15, 25, 18)
        
        box_w = x2 - x1
        box_h = y2 - y1
        corner_len = max(14, int(min(box_w, box_h) * 0.28))
        thick = max(2, int(box_w / 120.0))
        
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), c_neon, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), c_neon, thick, cv2.LINE_AA)
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cross_r = max(8, int(box_w * 0.12))
        cv2.line(img, (cx - cross_r, cy), (cx + cross_r, cy), c_neon, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - cross_r), (cx, cy + cross_r), c_neon, 1, cv2.LINE_AA)
        
        key_nodes = [
            468, 473,       # Iris
            1, 4, 168,      # Nariz
            61, 291, 0, 17, # Boca
            152,            # Barbilla
            234, 454        # Contorno
        ]
        for node_idx in key_nodes:
            nx = int(lms[node_idx].x * w)
            ny = int(lms[node_idx].y * h)
            cv2.circle(img, (nx, ny), 5, c_neon, 1, cv2.LINE_AA)
            cv2.circle(img, (nx, ny), 2, (200, 255, 200), -1, cv2.LINE_AA)
            
        card_w = max(190, int(w * 0.32))
        card_h = max(170, int(h * 0.30))
        pad = 25
        card_x1 = pad if (x1 > w * 0.45 or w < 600) else (w - card_w - pad)
        card_y1 = max(pad + 30, min(y1 - 20, h - card_h - pad))
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        overlay = img.copy()
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), c_dark, -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        
        cv2.rectangle(img, (card_x1, card_y1), (card_x2, card_y2), c_neon, 1, cv2.LINE_AA)
        cv2.line(img, (card_x1, card_y1), (card_x1 + 10, card_y1), c_neon, 2)
        cv2.line(img, (card_x1, card_y1), (card_x1, card_y1 + 10), c_neon, 2)
        cv2.line(img, (card_x2, card_y2), (card_x2 - 10, card_y2), c_neon, 2)
        cv2.line(img, (card_x2, card_y2), (card_x2, card_y2 - 10), c_neon, 2)
        
        thumb_size = max(45, int(card_h * 0.38))
        thumb_x = card_x1 + 10
        thumb_y = card_y1 + 12
        
        if face_crop is not None and face_crop.size > 0:
            thumb = cv2.resize(face_crop, (thumb_size, thumb_size))
            img[thumb_y:thumb_y + thumb_size, thumb_x:thumb_x + thumb_size] = thumb
            cv2.rectangle(img, (thumb_x, thumb_y), (thumb_x + thumb_size, thumb_y + thumb_size), c_neon, 1)
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_hdr = max(0.35, min(w, h) / 1800.0)
        scale_txt = max(0.32, min(w, h) / 2000.0)
        
        tx_hdr = thumb_x + thumb_size + 10
        cv2.putText(img, "TARGET DETECTED", (tx_hdr, thumb_y + 12), font, scale_hdr, c_neon, 1, cv2.LINE_AA)
        cv2.putText(img, f"ID: SUBJ-0{idx+1}", (tx_hdr, thumb_y + 28), font, scale_hdr, c_white, 1, cv2.LINE_AA)
        
        line_y = thumb_y + thumb_size + 18
        spacing = max(16, int(card_h * 0.12))
        
        g_txt = f"GENDER: {gender_info['gender'].upper()} ({gender_info['confidence']}%)"
        cv2.putText(img, g_txt, (card_x1 + 10, line_y), font, scale_txt, (57, 255, 20), 1, cv2.LINE_AA)
        
        line_y += spacing
        s_txt = f"SKIN: {skin_info['name'].upper()}"
        cv2.putText(img, s_txt, (card_x1 + 10, line_y), font, scale_txt, (200, 240, 210), 1, cv2.LINE_AA)
        
        line_y += spacing
        e_txt = f"EYES: {eye_info['name'].upper()}"
        cv2.putText(img, e_txt, (card_x1 + 10, line_y), font, scale_txt, (200, 240, 210), 1, cv2.LINE_AA)
        
        line_y += spacing
        a_txt = f"AGE EST: ~{gender_info['age']} YRS"
        cv2.putText(img, a_txt, (card_x1 + 10, line_y), font, scale_txt, (140, 180, 160), 1, cv2.LINE_AA)

    # =========================================================================
    # EXTRACCIÓN DE ATRIBUTOS
    # =========================================================================
    def _predict_gender_and_age(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return {"gender": "Desconocido", "confidence": 0.0, "age": 0}
        
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (96, 96)).astype(np.float32)
        blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0)
        
        preds = self.gender_session.run(None, {'data': blob})[0][0]
        g_logits = preds[:2]
        exp_g = np.exp(g_logits - np.max(g_logits))
        probs = exp_g / np.sum(exp_g)
        
        gender_idx = int(np.argmax(g_logits))
        if gender_idx == 1:
            gender_str = "Hombre"
            conf = float(probs[1])
        else:
            gender_str = "Mujer"
            conf = float(probs[0])
            
        age_est = int(np.clip(np.round(preds[2] * 100), 1, 100))
        
        return {
            "gender": gender_str,
            "confidence": round(conf * 100, 1),
            "age": age_est
        }

    def _analyze_skin_tone(self, img_rgb, lms, w, h):
        sample_indices = [10, 151, 9, 205, 50, 425, 280, 117, 346]
        colors = []
        
        for idx in sample_indices:
            cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
            patch = img_rgb[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
            if patch.size > 0:
                colors.append(np.median(patch.reshape(-1, 3), axis=0))
                
        if not colors:
            return {"name": "No determinado", "rgb": [200, 180, 160], "hex": "#C8B4A0"}
            
        avg_rgb = np.median(colors, axis=0).astype(int)
        r, g, b = avg_rgb
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        
        rgb_norm = np.array([[[r, g, b]]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2LAB)[0][0]
        L, a, b_star = float(lab[0]), float(lab[1]), float(lab[2])
        L_std = L * (100.0 / 255.0)
        b_std = (b_star - 128.0)
        
        if abs(b_std) < 0.1:
            b_std = 0.1
        ita = np.arctan2(L_std - 50.0, b_std) * (180.0 / np.pi)
        
        if ita > 52 or L_std > 78:
            tone_name = "Muy Claro / Pálido"
        elif ita > 38 or L_std > 66:
            tone_name = "Claro / Beige"
        elif ita > 25 or L_std > 55:
            tone_name = "Intermedio / Trigueño"
        elif ita > 10 or L_std > 42:
            tone_name = "Moreno / Bronceado"
        else:
            tone_name = "Oscuro / Ébano"
            
        return {
            "name": tone_name,
            "rgb": [int(r), int(g), int(b)],
            "hex": hex_code,
            "ita": round(ita, 1)
        }

    def _analyze_eye_color(self, img_rgb, lms, w, h):
        iris_indices = [468, 473]
        iris_pixels = []
        
        face_w = abs(lms[454].x - lms[234].x) * w
        r = max(2, int(face_w * 0.016))
        
        for iris_idx in iris_indices:
            cx, cy = int(lms[iris_idx].x * w), int(lms[iris_idx].y * h)
            patch = img_rgb[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
            if patch.size > 0:
                flat = patch.reshape(-1, 3)
                lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
                valid = flat[(lum > 28) & (lum < 225)]
                if len(valid) > 0:
                    iris_pixels.extend(valid)
                else:
                    iris_pixels.extend(flat)
                    
        if len(iris_pixels) == 0:
            return {"name": "No visible", "rgb": [50, 40, 30], "hex": "#32281E", "confidence": 0.0}
            
        median_rgb = np.median(iris_pixels, axis=0).astype(int)
        r, g, b = median_rgb
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        
        rgb_pixel = np.uint8([[[r, g, b]]])
        hsv = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
        hue = int(hsv[0]) * 2
        sat = int(hsv[1])
        val = int(hsv[2])
        
        if val < 40:
            color_name = "Negro / Ojo muy oscuro"
            conf = 90
        elif sat < 30 and val >= 50:
            color_name = "Gris"
            conf = 85
        elif (180 <= hue <= 260) or (b > r + 15 and b > g + 10):
            color_name = "Azul"
            conf = 92
        elif (65 <= hue <= 175) or (g > r + 10 and g > b):
            color_name = "Verde"
            conf = 90
        elif (35 <= hue <= 64):
            color_name = "Miel / Ámbar / Hazel"
            conf = 88
        elif hue < 35 or hue > 340:
            if val < 80:
                color_name = "Marrón Oscuro"
                conf = 95
            else:
                color_name = "Marrón / Castaño"
                conf = 92
        else:
            color_name = "Marrón / Castaño"
            conf = 85
            
        return {
            "name": color_name,
            "rgb": [int(r), int(g), int(b)],
            "hex": hex_code,
            "confidence": conf
        }
