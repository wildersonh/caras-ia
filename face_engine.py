import os
import cv2
import numpy as np
import datetime
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

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
        - "MEXICAN_CINEMA": Cine Mexicano / Árido (tono cálido sepia-ámbar desértico)
        - "DUOTONE_POP": Duotono Pop (#8A2BE2 sombras x #FF62CB luces)
        - "DARK_MOODY": Dark & Moody (luma -50%, sombras densas, melancólico)
        - "HIGH_CONTRAST_BW": Monocromo de Alto Contraste (analógico en B&W)
        - "CYBERPUNK_NEON": Cyberpunk / Neón (cianes y magentas con resplandor)
        - "GOLDEN_HOUR": Golden Hour (gradiente dorado cálido de atardecer)
        - "SKETCH": Boceto a Lápiz (Fine Graphite Pencil Sketch)
        - "COLOR_SPLASH": Aislamiento de Color (rojos aislados, resto B&W)
        - "PIXEL_ART": Píxel Art 8-Bit (retro arcade cuantizado)
        - "VHS": Cinta VHS Retro (aberración, scanlines y tracking analógico)
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
            elif filter_mode == "MEXICAN_CINEMA":
                return self._apply_mexican_cinema_effect(img_bgr), []
            elif filter_mode == "DUOTONE_POP":
                return self._apply_duotone_pop_effect(img_bgr), []
            elif filter_mode == "DARK_MOODY":
                return self._apply_dark_moody_effect(img_bgr), []
            elif filter_mode == "HIGH_CONTRAST_BW":
                return self._apply_high_contrast_bw_effect(img_bgr), []
            elif filter_mode == "CYBERPUNK_NEON":
                return self._apply_cyberpunk_neon_effect(img_bgr), []
            elif filter_mode == "GOLDEN_HOUR":
                return self._apply_golden_hour_effect(img_bgr), []
            elif filter_mode == "SKETCH":
                return self._apply_sketch_effect(img_bgr), []
            elif filter_mode == "COLOR_SPLASH":
                return self._apply_color_splash_effect(img_bgr), []
            elif filter_mode == "PIXEL_ART":
                return self._apply_pixel_art_effect(img_bgr), []
            elif filter_mode == "VHS":
                return self._apply_vhs_effect(img_bgr), []
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
        elif filter_mode == "MEXICAN_CINEMA":
            annotated_img = self._apply_mexican_cinema_effect(img_bgr)
        elif filter_mode == "DUOTONE_POP":
            annotated_img = self._apply_duotone_pop_effect(img_bgr)
        elif filter_mode == "DARK_MOODY":
            annotated_img = self._apply_dark_moody_effect(img_bgr)
        elif filter_mode == "HIGH_CONTRAST_BW":
            annotated_img = self._apply_high_contrast_bw_effect(img_bgr)
        elif filter_mode == "CYBERPUNK_NEON":
            annotated_img = self._apply_cyberpunk_neon_effect(img_bgr)
        elif filter_mode == "GOLDEN_HOUR":
            annotated_img = self._apply_golden_hour_effect(img_bgr)
        elif filter_mode == "SKETCH":
            annotated_img = self._apply_sketch_effect(img_bgr)
        elif filter_mode == "COLOR_SPLASH":
            annotated_img = self._apply_color_splash_effect(img_bgr)
        elif filter_mode == "PIXEL_ART":
            annotated_img = self._apply_pixel_art_effect(img_bgr)
        elif filter_mode == "VHS":
            annotated_img = self._apply_vhs_effect(img_bgr)
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
    # 1. FILTRO CINE MEXICANO / ÁRIDO (WARM DESERT GRADE)
    # =========================================================================
    def _apply_mexican_cinema_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(img_bgr, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        blended = cv2.addWeighted(sepia, 0.40, img_bgr, 0.60, 0)
        hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.20, 0, 255)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.94 + 3, 0, 179)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        res = cv2.convertScaleAbs(res, alpha=1.10, beta=4)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "CINE ARIDO // WARM DESERT GRADE (SEPIA +40% / SAT +120%)", (15, h - 15), font, scale, (40, 190, 240), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # 2. FILTRO DUOTONO POP (#8A2BE2 SOMBRAS x #FF62CB LUCES)
    # =========================================================================
    def _apply_duotone_pop_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        c_shadow = np.array([226, 43, 138], dtype=np.float32)
        c_high = np.array([203, 98, 255], dtype=np.float32)
        
        t = gray[:, :, np.newaxis]
        duo = (1.0 - t) * c_shadow + t * c_high
        output = np.clip(duo, 0, 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(output, "DUOTONE POP // #8A2BE2 DEEP PURPLE x #FF62CB VIBRANT ACCENT", (15, h - 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # 3. FILTRO DARK & MOODY (BAJO BRILLO & SOMBRAS DENSAS)
    # =========================================================================
    def _apply_dark_moody_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        darkened = cv2.convertScaleAbs(img_bgr, alpha=0.55, beta=-12)
        hsv = cv2.cvtColor(darkened, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.82, 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        X = cv2.getGaussianKernel(w, w * 0.52)
        Y = cv2.getGaussianKernel(h, h * 0.52)
        kernel = Y * X.T
        mask = kernel / kernel.max()
        res = (res.astype(np.float32) * (0.35 + 0.65 * mask[:, :, np.newaxis])).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "DARK & MOODY // CINEMATIC LOW-KEY (LUMA -50% / CRUSHED SHADOWS)", (15, h - 15), font, scale, (160, 180, 200), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # 4. FILTRO MONOCROMO DE ALTO CONTRASTE (ANALOG B&W)
    # =========================================================================
    def _apply_high_contrast_bw_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            v = i / 255.0
            v_s = v * v * (3 - 2 * v)
            lut[i] = np.clip(v_s * 280.0 - 15.0, 0, 255)
        stretched = cv2.LUT(gray, lut)
        res = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "HIGH CONTRAST B&W // ANALOG SILVER GELATIN (PURE BLACKS & HIGHLIGHTS)", (15, h - 15), font, scale, (230, 230, 230), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # 5. FILTRO CYBERPUNK / NEÓN (CIANES, MAGENTAS Y RESPLANDOR)
    # =========================================================================
    def _apply_cyberpunk_neon_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_ch = hsv[:, :, 0]
        h_new = np.where(h_ch < 90, 85.0 + (h_ch / 90.0) * 15.0, 145.0 + ((h_ch - 90.0) / 90.0) * 20.0)
        hsv[:, :, 0] = h_new
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.50 + 25, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        blur = cv2.GaussianBlur(res, (21, 21), 0)
        output = cv2.addWeighted(res, 0.72, blur, 0.38, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(output, "CYBERPUNK NEON // CYAN-MAGENTA HUE ROTATE & HIGHLIGHT BLOOM", (15, h - 15), font, scale, (255, 240, 0), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # 6. FILTRO GOLDEN HOUR (GRADIENTE CÁLIDO DE ATARDECER)
    # =========================================================================
    def _apply_golden_hour_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        y_coords = np.linspace(1.0, 0.15, h, dtype=np.float32)[:, np.newaxis]
        grad_golden = np.zeros((h, w, 3), dtype=np.float32)
        grad_golden[:, :, 0] = 25.0 * y_coords
        grad_golden[:, :, 1] = 140.0 * y_coords
        grad_golden[:, :, 2] = 255.0 * y_coords
        
        blended = cv2.addWeighted(img_bgr.astype(np.float32), 0.76, grad_golden, 0.36, 4)
        output = np.clip(blended, 0, 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(output, "GOLDEN HOUR // SUNSET AMBER GRADIENT & WARM DIFFUSION", (15, h - 15), font, scale, (30, 210, 255), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # 7. FILTRO BOCETO A LÁPIZ (FINE GRAPHITE SKETCH)
    # =========================================================================
    def _apply_sketch_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur_k = max(15, (int(min(w, h) * 0.04) // 2) * 2 + 1)
        blur = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        output = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(output, "PENCIL SKETCH // FINE GRAPHITE & GAUSSIAN DODGE", (15, h - 15), font, scale, (60, 60, 60), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # 8. FILTRO AISLAMIENTO DE COLOR (COLOR SPLASH ROJO / CÁLIDO)
    # =========================================================================
    def _apply_color_splash_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 55, 45])
        upper1 = np.array([12, 255, 255])
        lower2 = np.array([168, 55, 45])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
        m = (mask_blur.astype(np.float32) / 255.0)[:, :, np.newaxis]
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        boosted = cv2.convertScaleAbs(img_bgr, alpha=1.25, beta=10)
        splash = (boosted.astype(np.float32) * m + gray_bgr.astype(np.float32) * (1.0 - m))
        output = np.clip(splash, 0, 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(output, "COLOR SPLASH // ISOLATED RED-WARM CHROMA (HSV MASK)", (15, h - 15), font, scale, (50, 50, 240), 1, cv2.LINE_AA)
        return output

    # =========================================================================
    # 9. FILTRO PÍXEL ART / 8-BIT RETRO
    # =========================================================================
    def _apply_pixel_art_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        scale_w = max(48, min(120, w // 7))
        scale_h = max(48, min(120, h // 7))
        small = cv2.resize(img_bgr, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        quantized = (small // 32) * 32 + 16
        pixelated = cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)
        pixelated[::4, :] = (pixelated[::4, :].astype(np.float32) * 0.85).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(pixelated, "PIXEL ART // 8-BIT RETRO ARCADE (NEAREST-NEIGHBOR QUANTIZED)", (15, h - 15), font, scale, (0, 255, 180), 1, cv2.LINE_AA)
        return pixelated

    # =========================================================================
    # 10. FILTRO CINTA VHS RETRO (NTSC GLITCH & TRACKING)
    # =========================================================================
    def _apply_vhs_effect(self, img_bgr):
        h, w, _ = img_bgr.shape
        offset = max(4, int(w * 0.012))
        b, g, r = cv2.split(img_bgr)
        r_sh = np.roll(r, offset, axis=1)
        b_sh = np.roll(b, -offset, axis=1)
        vhs = cv2.merge([b_sh, g, r_sh])
        vhs[::3, :] = (vhs[::3, :].astype(np.float32) * 0.78).astype(np.uint8)
        noise = np.random.normal(0, 10, (h, w, 3)).astype(np.float32)
        vhs = np.clip(vhs.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        track_y = int(h * 0.84)
        track_h = max(10, int(h * 0.07))
        if track_y + track_h < h:
            vhs[track_y:track_y+track_h, :] = np.roll(vhs[track_y:track_y+track_h, :], 24, axis=1)
            vhs[track_y:track_y+track_h, :] = cv2.GaussianBlur(vhs[track_y:track_y+track_h, :], (15, 1), 0)
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.38, min(w, h) / 1500.0)
        cv2.putText(vhs, "PLAY ▶  SP  0:24:18", (20, 35), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vhs, "VHS NTSC // TRACKING ACTIVE", (15, h - 15), font, scale * 0.85, (220, 220, 220), 1, cv2.LINE_AA)
        return vhs

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
