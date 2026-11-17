import os
import cv2
import numpy as np
import datetime
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

    def analyze_image(self, img_bgr, enable_sniper_filter=False):
        """
        Analiza la imagen. Si enable_sniper_filter=True, aplica el filtro óptico
        de mira de rifle (Sniper Scope) en lugar del HUD táctico estándar.
        """
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.landmarker.detect(mp_img)
        
        annotated_img = img_bgr.copy()
        faces_results = []
        
        # 1. Si no hay rostros detectados
        if not detection_result.face_landmarks:
            if enable_sniper_filter:
                annotated_img = self._apply_sniper_scope_effect(img_bgr, center=(w // 2, h // 2))
            else:
                self._draw_hud_outer_frame(annotated_img, w, h)
                cv2.putText(annotated_img, "STATUS: SEARCHING TARGET...", (30, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57, 255, 20), 2, cv2.LINE_AA)
            return annotated_img, []

        # 2. Procesar cada rostro detectado
        for face_idx, lms in enumerate(detection_result.face_landmarks):
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
            y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))
            
            # Margen para el recorte
            margin_x = int((x_max - x_min) * 0.15)
            margin_y = int((y_max - y_min) * 0.15)
            crop_x1 = max(0, x_min - margin_x)
            crop_x2 = min(w, x_max + margin_x)
            crop_y1 = max(0, y_min - margin_y)
            crop_y2 = min(h, y_max + margin_y)
            
            face_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Análisis biométrico
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

        # 3. Renderizado visual según el modo de filtro seleccionado
        if enable_sniper_filter:
            # Centrar la mira del rifle en el rostro principal detectado
            main_face = faces_results[0]
            x1, y1, x2, y2 = main_face["bbox"]
            center_x = (x1 + x2) // 2
            center_y = int(y1 + (y2 - y1) * 0.42) # Centrado a la altura de los ojos/nariz
            annotated_img = self._apply_sniper_scope_effect(img_bgr, center=(center_x, center_y))
        else:
            # Modo HUD Táctico estándar
            self._draw_hud_outer_frame(annotated_img, w, h)
            for idx, f in enumerate(faces_results):
                lms = detection_result.face_landmarks[idx]
                x1, y1, x2, y2 = f["bbox"]
                crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                self._draw_hud_face_target(
                    annotated_img, lms, w, h, 
                    f["bbox"], 
                    {"gender": f["gender"], "confidence": f["gender_confidence"], "age": f["estimated_age"]}, 
                    {"name": f["skin_tone"]}, 
                    {"name": f["eye_color"]}, 
                    crop, idx
                )

        return annotated_img, faces_results

    def _apply_sniper_scope_effect(self, img_bgr, center=None):
        """
        Aplica el filtro visual de Mira de Rifle (Sniper Scope):
        - Viñeta exterior negra con lente circular
        - Tinte térmico / infrarrojo rojo intenso cinematográfico
        - Retícula balística con marcas de graduación (mil-dot hash marks) en azul acero
        """
        h, w, _ = img_bgr.shape
        if center is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = center
            
        radius = int(min(w, h) * 0.47)
        
        # 1. Tinte rojo cinematográfico intenso (Thermal / Infrarrojo de mira)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        r_ch = np.clip(gray.astype(np.float32) * 1.35 + 30, 0, 255).astype(np.uint8)
        g_ch = np.clip(gray.astype(np.float32) * 0.20 - 5, 0, 255).astype(np.uint8)
        b_ch = np.clip(gray.astype(np.float32) * 0.10 - 10, 0, 255).astype(np.uint8)
        red_tinted = cv2.merge([b_ch, g_ch, r_ch])
        
        # 2. Máscara circular con borde suave (Feathered Scope Edge)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
        
        # Suavizado de bordes para imitar el bisel óptico del visor de rifle
        blur_k = max(15, (radius // 18) * 2 + 1)
        mask_blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        
        # Fondo completamente negro fuera del círculo de la mira
        output = (red_tinted.astype(np.float32) * mask_blurred[:, :, np.newaxis]).astype(np.uint8)
        
        # Anillo perimetral tenue de la lente
        cv2.circle(output, (cx, cy), radius, (40, 20, 100), 2, cv2.LINE_AA)
        
        # 3. Retícula de Mira de Francotirador (Sniper Crosshair)
        # Color azul acero / cian tenue idéntico a la imagen de referencia
        c_reticle = (245, 205, 155) # BGR
        
        arm_len = int(radius * 0.90)
        gap = max(4, int(radius * 0.04)) # Pequeña apertura central
        
        # Líneas de la cruz (Horizontal y Vertical)
        # Brazo izquierdo y derecho
        cv2.line(output, (cx - arm_len, cy), (cx - gap, cy), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx + gap, cy), (cx + arm_len, cy), c_reticle, 1, cv2.LINE_AA)
        # Brazo superior e inferior
        cv2.line(output, (cx, cy - arm_len), (cx, cy - gap), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy + gap), (cx, cy + arm_len), c_reticle, 1, cv2.LINE_AA)
        
        # Puntos de la micro-cruz central
        cv2.line(output, (cx - 2, cy), (cx + 2, cy), c_reticle, 1, cv2.LINE_AA)
        cv2.line(output, (cx, cy - 2), (cx, cy + 2), c_reticle, 1, cv2.LINE_AA)
        
        # 4. Graduaciones balísticas / Mil-dot hash marks (las líneas perpendiculares)
        tick_step = max(8, int(radius * 0.08))
        num_ticks = 5
        tick_len = max(5, int(radius * 0.045))
        
        for i in range(1, num_ticks + 1):
            dist = gap + i * tick_step
            # Graduaciones en eje horizontal (líneas verticales)
            if dist < arm_len - 5:
                t_h = tick_len if i % 2 == 1 else int(tick_len * 0.65)
                cv2.line(output, (cx - dist, cy - t_h), (cx - dist, cy + t_h), c_reticle, 1, cv2.LINE_AA)
                cv2.line(output, (cx + dist, cy - t_h), (cx + dist, cy + t_h), c_reticle, 1, cv2.LINE_AA)
                
            # Graduaciones en eje vertical (líneas horizontales)
            if dist < arm_len - 5:
                t_w = tick_len if i % 2 == 1 else int(tick_len * 0.65)
                cv2.line(output, (cx - t_w, cy - dist), (cx + t_w, cy - dist), c_reticle, 1, cv2.LINE_AA)
                cv2.line(output, (cx - t_w, cy + dist), (cx + t_w, cy + dist), c_reticle, 1, cv2.LINE_AA)
                
        # 5. Marcador discreto de telemetría de francotirador en la esquina inferior
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(w, h) / 1800.0)
        cv2.putText(output, "OPTIC: 12x THERMAL // RETICLE: MIL-DOT", (15, h - 15), font, scale, (140, 160, 180), 1, cv2.LINE_AA)
        
        return output

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
            
        sub_txt = "BIO-IDENT PROTOCOL: v2.4 // TARGET ACQUIRED"
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
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), c_neon, thick, cv2.LINE_AA)
        
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
