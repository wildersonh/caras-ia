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
        - "FACE_MESH": Malla Holográfica 3D Wireframe (478 vértices)
        - "EMBOSS": Relieve Esculpido 3D (convolución asimétrica direccional)
        - "GLOWING_EDGES": Bordes de Neón luminiscentes (Canny + resplandor cian)
        - "UNSHARP_MASK": Nitidez Extrema (máscara de desenfoque y microcontraste)
        - "GAMEBOY_DITHER": Dithering Retro Game Boy clásico (paleta DMG 4 tonos)
        - "LED_MATRIX": Pantalla LED de estadio (matriz de diodos luminosos)
        - "POSTERIZE_HOPE": Posterización Estricta pop art (póster HOPE propaganda)
        - "MOTION_BLUR": Desenfoque de Movimiento 1D (alta velocidad horizontal)
        - "ZOOM_BURST": Desenfoque Radial / hipervelocidad cinética sobre el rostro
        - "GHOSTING": Estelas Espectrales y rastro de movimiento multi-frame
        - "GREEN_SCREEN": Segmentación de Fondo verde croma de estudio (#00FF00)
        - "PORTRAIT_ISOLATE": Aislamiento de Retrato (sujeto a color, fondo desenfocado B&W)
        - "PIXEL_SORTING": Pixel Sorting / Derretimiento Digital (glitch cascade)
        - "MANGA_JAPANESE": Manga Japonés Clásico (plumilla G-Pen y trama screentone)
        - "BULGE_MIRROR": Espejo de Feria / Lupa (distorsión esférica)
        - "SWIRL": Remolino / Vórtice de rotación angular
        - "KALEIDOSCOPE": Caleidoscopio (patrón simétrico radial 8 sectores)
        - "WATER_RIPPLE": Ondulación de Agua (ondas senoidales líquidas)
        - "TILT_SHIFT": Efecto Miniatura Tilt-Shift con desenfoque progresivo
        - "ANAMORPHIC_FLARE": Destello Anamórfico Azul de lentes de cine
        - "LIGHT_LEAKS": Fugas de Luz Analógica cálidas
        - "FISHEYE": Aberración Esférica Ojo de Pez con viñeteado
        - "ASCII_ART": Arte ASCII matricial monoespaciado
        - "HALFTONE_COMIC": Medio Tono / Cómic Pop Art Roy Lichtenstein
        - "OIL_PAINTING": Pintura al Óleo con trazos y bordes definidos
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
            elif filter_mode == "FACE_MESH":
                self._draw_hud_outer_frame(annotated_img, w, h)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(0.40, min(w, h) / 1400.0)
                cv2.putText(annotated_img, "[ 3D BIOMETRIC TOPOLOGY // SCANNING TARGET... ]", (25, h // 2), font, scale, (0, 255, 230), 2, cv2.LINE_AA)
                return annotated_img, []
            elif filter_mode == "EMBOSS":
                return self._apply_emboss_effect(img_bgr), []
            elif filter_mode == "GLOWING_EDGES":
                return self._apply_glowing_edges_effect(img_bgr), []
            elif filter_mode == "UNSHARP_MASK":
                return self._apply_unsharp_mask_effect(img_bgr), []
            elif filter_mode == "GAMEBOY_DITHER":
                return self._apply_gameboy_dither_effect(img_bgr), []
            elif filter_mode == "LED_MATRIX":
                return self._apply_led_matrix_effect(img_bgr), []
            elif filter_mode == "POSTERIZE_HOPE":
                return self._apply_posterize_hope_effect(img_bgr), []
            elif filter_mode == "MOTION_BLUR":
                return self._apply_motion_blur_effect(img_bgr), []
            elif filter_mode == "ZOOM_BURST":
                return self._apply_zoom_burst_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "GHOSTING":
                return self._apply_ghosting_effect(img_bgr), []
            elif filter_mode == "GREEN_SCREEN":
                return self._apply_green_screen_effect(img_bgr, landmarks=None), []
            elif filter_mode == "PORTRAIT_ISOLATE":
                return self._apply_portrait_isolate_effect(img_bgr, landmarks=None), []
            elif filter_mode == "PIXEL_SORTING":
                return self._apply_pixel_sorting_effect(img_bgr), []
            elif filter_mode == "MANGA_JAPANESE":
                return self._apply_japanese_manga_effect(img_bgr), []
            elif filter_mode == "BULGE_MIRROR":
                return self._apply_bulge_mirror_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "SWIRL":
                return self._apply_swirl_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "KALEIDOSCOPE":
                return self._apply_kaleidoscope_effect(img_bgr), []
            elif filter_mode == "WATER_RIPPLE":
                return self._apply_water_ripple_effect(img_bgr, center=(w // 2, h // 2)), []
            elif filter_mode == "TILT_SHIFT":
                return self._apply_tilt_shift_effect(img_bgr), []
            elif filter_mode == "ANAMORPHIC_FLARE":
                return self._apply_anamorphic_flare_effect(img_bgr), []
            elif filter_mode == "LIGHT_LEAKS":
                return self._apply_light_leaks_effect(img_bgr), []
            elif filter_mode == "FISHEYE":
                return self._apply_fisheye_effect(img_bgr), []
            elif filter_mode == "ASCII_ART":
                return self._apply_ascii_art_effect(img_bgr), []
            elif filter_mode == "HALFTONE_COMIC":
                return self._apply_halftone_comic_effect(img_bgr), []
            elif filter_mode == "OIL_PAINTING":
                return self._apply_oil_painting_effect(img_bgr), []
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
        elif filter_mode == "FACE_MESH":
            annotated_img = self._apply_face_mesh_effect(img_bgr, main_lms, main_face, w, h)
        elif filter_mode == "EMBOSS":
            annotated_img = self._apply_emboss_effect(img_bgr)
        elif filter_mode == "GLOWING_EDGES":
            annotated_img = self._apply_glowing_edges_effect(img_bgr)
        elif filter_mode == "UNSHARP_MASK":
            annotated_img = self._apply_unsharp_mask_effect(img_bgr)
        elif filter_mode == "GAMEBOY_DITHER":
            annotated_img = self._apply_gameboy_dither_effect(img_bgr)
        elif filter_mode == "LED_MATRIX":
            annotated_img = self._apply_led_matrix_effect(img_bgr)
        elif filter_mode == "POSTERIZE_HOPE":
            annotated_img = self._apply_posterize_hope_effect(img_bgr)
        elif filter_mode == "MOTION_BLUR":
            annotated_img = self._apply_motion_blur_effect(img_bgr)
        elif filter_mode == "ZOOM_BURST":
            annotated_img = self._apply_zoom_burst_effect(img_bgr, center=target_center)
        elif filter_mode == "GHOSTING":
            annotated_img = self._apply_ghosting_effect(img_bgr)
        elif filter_mode == "GREEN_SCREEN":
            annotated_img = self._apply_green_screen_effect(img_bgr, landmarks=main_lms)
        elif filter_mode == "PORTRAIT_ISOLATE":
            annotated_img = self._apply_portrait_isolate_effect(img_bgr, landmarks=main_lms)
        elif filter_mode == "PIXEL_SORTING":
            annotated_img = self._apply_pixel_sorting_effect(img_bgr)
        elif filter_mode == "MANGA_JAPANESE":
            annotated_img = self._apply_japanese_manga_effect(img_bgr)
        elif filter_mode == "BULGE_MIRROR":
            annotated_img = self._apply_bulge_mirror_effect(img_bgr, center=target_center)
        elif filter_mode == "SWIRL":
            annotated_img = self._apply_swirl_effect(img_bgr, center=target_center)
        elif filter_mode == "KALEIDOSCOPE":
            annotated_img = self._apply_kaleidoscope_effect(img_bgr)
        elif filter_mode == "WATER_RIPPLE":
            annotated_img = self._apply_water_ripple_effect(img_bgr, center=target_center)
        elif filter_mode == "TILT_SHIFT":
            annotated_img = self._apply_tilt_shift_effect(img_bgr)
        elif filter_mode == "ANAMORPHIC_FLARE":
            annotated_img = self._apply_anamorphic_flare_effect(img_bgr)
        elif filter_mode == "LIGHT_LEAKS":
            annotated_img = self._apply_light_leaks_effect(img_bgr)
        elif filter_mode == "FISHEYE":
            annotated_img = self._apply_fisheye_effect(img_bgr)
        elif filter_mode == "ASCII_ART":
            annotated_img = self._apply_ascii_art_effect(img_bgr)
        elif filter_mode == "HALFTONE_COMIC":
            annotated_img = self._apply_halftone_comic_effect(img_bgr)
        elif filter_mode == "OIL_PAINTING":
            annotated_img = self._apply_oil_painting_effect(img_bgr)
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
    # A. FILTROS GEOMÉTRICOS Y DE DEFORMACIÓN ESPACIAL
    # =========================================================================
    def _apply_bulge_mirror_effect(self, img_bgr, center=None):
        """Espejo de Feria (Pinch & Bulge): Distorsión esférica / lupa centrada en el rostro"""
        h, w = img_bgr.shape[:2]
        cx, cy = center if center is not None else (w // 2, h // 2)
        radius = min(w, h) * 0.45
        
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = x_indices - cx
        dy = y_indices - cy
        r = np.sqrt(dx * dx + dy * dy)
        
        mask = r < radius
        r_norm = r[mask] / radius
        distortion = np.sin(r_norm * np.pi * 0.5) ** 0.55
        denom = np.maximum(r_norm, 1e-6)
        factor = np.where(r_norm > 0, distortion / denom, 1.0)
        
        map_x = x_indices.copy()
        map_y = y_indices.copy()
        map_x[mask] = cx + dx[mask] * factor
        map_y[mask] = cy + dy[mask] * factor
        
        res = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "BULGE MIRROR // DISTORSION ESFERICA DE LUPA", (15, h - 15), font, scale, (255, 230, 80), 1, cv2.LINE_AA)
        return res

    def _apply_swirl_effect(self, img_bgr, center=None):
        """Remolino (Swirl Vortex): Vórtice de rotación angular centrada en el rostro"""
        h, w = img_bgr.shape[:2]
        cx, cy = center if center is not None else (w // 2, h // 2)
        radius = min(w, h) * 0.50
        strength = 4.5  # Rotación máxima en el núcleo
        
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = x_indices - cx
        dy = y_indices - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        
        mask = r < radius
        angle = theta[mask] + strength * ((radius - r[mask]) / radius) ** 2
        
        map_x = x_indices.copy()
        map_y = y_indices.copy()
        map_x[mask] = cx + r[mask] * np.cos(angle)
        map_y[mask] = cy + r[mask] * np.sin(angle)
        
        res = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "SWIRL VORTEX // REMOLINO DE ROTACION RADIAL", (15, h - 15), font, scale, (120, 240, 255), 1, cv2.LINE_AA)
        return res

    def _apply_kaleidoscope_effect(self, img_bgr):
        """Caleidoscopio: Simetría radial pop de 8 sectores angulares"""
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = x_indices - cx
        dy = y_indices - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        
        num_sectors = 8
        sector_angle = 2.0 * np.pi / num_sectors
        theta_folded = np.abs(np.mod(theta + np.pi, sector_angle) - sector_angle * 0.5)
        
        map_x = (cx + r * np.cos(theta_folded)).astype(np.float32)
        map_y = (cy + r * np.sin(theta_folded)).astype(np.float32)
        
        res = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "KALEIDOSCOPE // SIMETRIA RADIAL POP (8 SECTORES)", (15, h - 15), font, scale, (240, 180, 255), 1, cv2.LINE_AA)
        return res

    def _apply_water_ripple_effect(self, img_bgr, center=None):
        """Ondulación de Agua (Water Ripple): Ondas senoidales líquidas concéntricas"""
        h, w = img_bgr.shape[:2]
        cx, cy = center if center is not None else (w // 2, h // 2)
        
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = x_indices - cx
        dy = y_indices - cy
        r = np.sqrt(dx * dx + dy * dy)
        
        wavelength = 35.0
        amplitude = 14.0
        decay = np.exp(-r / (max(w, h) * 0.40))
        wave = np.sin(2.0 * np.pi * r / wavelength) * amplitude * decay
        
        r_safe = np.maximum(r, 1e-6)
        map_x = (x_indices + wave * (dx / r_safe)).astype(np.float32)
        map_y = (y_indices + wave * (dy / r_safe)).astype(np.float32)
        
        res = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "WATER RIPPLE // ONDULACION LIQUIDA CON REFLEJO", (15, h - 15), font, scale, (100, 220, 255), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # B. FILTROS CINEMATOGRÁFICOS Y ÓPTICOS
    # =========================================================================
    def _apply_tilt_shift_effect(self, img_bgr):
        """Tilt-Shift: Enfoque selectivo horizontal con saturación miniatura"""
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)
        saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        saturated = cv2.convertScaleAbs(saturated, alpha=1.12, beta=3)
        
        blur_ksize = max(15, (min(w, h) // 30) | 1)
        blurred = cv2.GaussianBlur(saturated, (blur_ksize, blur_ksize), 0)
        
        y_norm = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        blur_weight = np.clip((np.abs(y_norm) - 0.20) / 0.40, 0.0, 1.0)
        blur_weight = (blur_weight * blur_weight * (3 - 2 * blur_weight))[:, :, np.newaxis]
        
        res = ((1.0 - blur_weight) * saturated.astype(np.float32) + blur_weight * blurred.astype(np.float32)).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "TILT-SHIFT // EFECTO MAQUETA MINIATURA (PROFUNDIDAD SELECTIVA)", (15, h - 15), font, scale, (255, 255, 180), 1, cv2.LINE_AA)
        return res

    def _apply_anamorphic_flare_effect(self, img_bgr):
        """Destello Anamórfico: Resplandor horizontal azul/cian característico de lentes anamórficos"""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_TOZERO)
        
        streak_w = max(31, (w // 8) | 1)
        streak_h = max(3, (h // 120) | 1)
        streak = cv2.GaussianBlur(thresh, (streak_w, streak_h), 0)
        
        streak_bgr = np.zeros((h, w, 3), dtype=np.float32)
        s_norm = streak.astype(np.float32) / 255.0
        streak_bgr[:, :, 0] = s_norm * 255.0
        streak_bgr[:, :, 1] = s_norm * 170.0
        streak_bgr[:, :, 2] = s_norm * 35.0
        
        base = img_bgr.astype(np.float32)
        res = 255.0 - ((255.0 - base) * (255.0 - streak_bgr * 0.85) / 255.0)
        res = np.clip(res, 0, 255).astype(np.uint8)
        
        cv2.line(res, (0, 3), (w, 3), (255, 180, 50), 1)
        cv2.line(res, (0, h - 4), (w, h - 4), (255, 180, 50), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "ANAMORPHIC FLARE // LENTES DE CINE 2.39:1 (CELESTE ANAMORFICO)", (15, h - 15), font, scale, (255, 210, 80), 1, cv2.LINE_AA)
        return res

    def _apply_light_leaks_effect(self, img_bgr):
        """Fugas de Luz (Light Leaks): Manchas cálidas de película analógica 35mm"""
        h, w = img_bgr.shape[:2]
        leak = np.zeros((h, w, 3), dtype=np.float32)
        
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        r_tl = np.sqrt(x_idx**2 + y_idx**2)
        leak_tl = np.clip(1.0 - r_tl / (max(w, h) * 0.65), 0.0, 1.0) ** 1.8
        
        r_br = np.sqrt((w - 1 - x_idx)**2 + (h - 1 - y_idx)**2)
        leak_br = np.clip(1.0 - r_br / (max(w, h) * 0.55), 0.0, 1.0) ** 2.0
        
        leak[:, :, 0] += leak_tl * 30.0 + leak_br * 40.0
        leak[:, :, 1] += leak_tl * 120.0 + leak_br * 190.0
        leak[:, :, 2] += leak_tl * 255.0 + leak_br * 255.0
        
        base = img_bgr.astype(np.float32)
        res = 255.0 - ((255.0 - base) * (255.0 - np.clip(leak, 0, 255)) / 255.0)
        res = np.clip(res, 0, 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "LIGHT LEAKS // FUGA DE LUZ ANALOGICA 35MM (WARM VINTAGE)", (15, h - 15), font, scale, (80, 180, 255), 1, cv2.LINE_AA)
        return res

    def _apply_fisheye_effect(self, img_bgr):
        """Ojo de Pez (Fisheye): Aberración esférica extrema y viñeteado circular"""
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = (x_indices - cx) / cx
        dy = (y_indices - cy) / cy
        r = np.sqrt(dx * dx + dy * dy)
        
        r_dist = r * (1.0 + 0.35 * r * r)
        factor = np.where(r > 0, r_dist / np.maximum(r, 1e-6), 1.0)
        
        map_x = (cx + dx * factor * cx).astype(np.float32)
        map_y = (cy + dy * factor * cy).astype(np.float32)
        
        distorted = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        r_corner = np.sqrt(((x_indices - cx) / (w * 0.55))**2 + ((y_indices - cy) / (h * 0.55))**2)
        vignette = np.clip(1.0 - np.clip((r_corner - 0.75) / 0.35, 0.0, 1.0), 0.0, 1.0)
        res = (distorted.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "FISHEYE LENS // ABERRACION ESFERICA & GRAN ANGULAR", (15, h - 15), font, scale, (220, 220, 220), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # C. FILTROS ARTÍSTICOS Y COMPUTACIONALES AVANZADOS
    # =========================================================================
    def _apply_ascii_art_effect(self, img_bgr):
        """Arte ASCII: Renderizado de matriz de texto monoespaciado según luminancia"""
        h, w = img_bgr.shape[:2]
        chars = " .:-=+*#%@"
        num_chars = len(chars)
        
        block_size = max(7, min(w, h) // 70)
        small_w = w // block_size
        small_h = h // block_size
        
        small_bgr = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        
        res = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.24, block_size / 24.0)
        
        for r in range(small_h):
            y_pos = r * block_size + block_size - 2
            for c in range(small_w):
                val = small_gray[r, c]
                char_idx = int(val / 256.0 * num_chars)
                char = chars[min(char_idx, num_chars - 1)]
                color = tuple(int(x) for x in small_bgr[r, c])
                x_pos = c * block_size + 1
                cv2.putText(res, char, (x_pos, y_pos), font, font_scale, color, 1, cv2.LINE_AA)
                
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "ASCII ART // MATRIZ MONOESPACIADA & LUMINANCIA", (15, h - 15), font, scale, (0, 255, 120), 1, cv2.LINE_AA)
        return res

    def _apply_halftone_comic_effect(self, img_bgr):
        """Medio Tono (Halftone / Cómic Pop Art): Retícula de puntos Roy Lichtenstein"""
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.55, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        quant = (boosted // 64) * 64 + 32
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        step = max(7, min(w, h) // 80)
        max_radius = step * 0.70
        canvas = quant.copy()
        
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                val = float(gray[y, x])
                darkness = 1.0 - (val / 255.0)
                rad = int(darkness * max_radius)
                if rad > 1:
                    cv2.circle(canvas, (x, y), rad, (20, 20, 30), -1, cv2.LINE_AA)
                    
        edges = cv2.Canny(gray, 60, 160)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        canvas[edges_dilated > 0] = (15, 15, 15)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(canvas, "HALFTONE COMIC // RETICULA POP ART & TINTA NEGRA", (15, h - 15), font, scale, (30, 30, 240), 1, cv2.LINE_AA)
        return canvas

    def _apply_oil_painting_effect(self, img_bgr):
        """Pintura al Óleo: Suavizado bilateral múltiple y realce de trazos pastosos"""
        h, w = img_bgr.shape[:2]
        scale_factor = 0.65 if max(w, h) > 700 else 1.0
        small = cv2.resize(img_bgr, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA) if scale_factor < 1.0 else img_bgr
        
        smooth = small
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=65, sigmaSpace=65)
            
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        painterly = cv2.bitwise_and(smooth, edges_bgr)
        
        if scale_factor < 1.0:
            res = cv2.resize(painterly, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            res = painterly
            
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "OIL PAINTING // PINTURA AL OLEO & TRAZOS PASTOSOS", (15, h - 15), font, scale, (240, 240, 240), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # D. EFECTOS MATRICIALES (CONVOLUCIONES ESPACIALES)
    # =========================================================================
    def _apply_emboss_effect(self, img_bgr):
        """Relieve (Emboss): Convolución asimétrica direccional simulando piedra o moneda tallada en 3D"""
        h, w = img_bgr.shape[:2]
        kernel = np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ], dtype=np.float32)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        embossed = cv2.filter2D(gray, -1, kernel) + 128
        res = cv2.cvtColor(np.clip(embossed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "EMBOSS // RELIEVE ASIMETRICO 3D (ESCULTURA MONOLITICA)", (15, h - 15), font, scale, (230, 230, 230), 1, cv2.LINE_AA)
        return res

    def _apply_glowing_edges_effect(self, img_bgr):
        """Bordes de Neón (Glowing Edges): Canny + dilatación + luminiscencia cian/magenta sobre negro"""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 140)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        glow = cv2.GaussianBlur(dilated, (15, 15), 0)
        
        res = np.zeros((h, w, 3), dtype=np.uint8)
        res[:, :, 0] = np.clip(dilated.astype(np.float32) * 0.95 + glow.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        res[:, :, 1] = np.clip(dilated.astype(np.float32) * 0.85 + glow.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        res[:, :, 2] = np.clip(glow.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "GLOWING EDGES // BORDES DE NEON LUMINISCENTES", (15, h - 15), font, scale, (255, 255, 100), 1, cv2.LINE_AA)
        return res

    def _apply_unsharp_mask_effect(self, img_bgr):
        """Nitidez Extrema (Unsharp Mask): Realce agresivo de frecuencias altas y microcontraste"""
        h, w = img_bgr.shape[:2]
        gaussian = cv2.GaussianBlur(img_bgr, (9, 9), 10.0)
        res = cv2.addWeighted(img_bgr, 1.85, gaussian, -0.85, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "UNSHARP MASK // NITIDEZ EXTREMA & MICROCONTRASTE OPTICO", (15, h - 15), font, scale, (255, 230, 100), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # E. ESTILOS RETRO ALGORÍTMICOS
    # =========================================================================
    def _apply_gameboy_dither_effect(self, img_bgr):
        """Dithering Game Boy Clásico: 4 tonos de verde original con matriz de difusión Bayer"""
        h, w = img_bgr.shape[:2]
        target_w = 160
        target_h = max(1, int(h * (target_w / float(w))))
        small = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        bayer4 = np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ], dtype=np.float32) / 16.0 - 0.5
        
        tiled_bayer = np.tile(bayer4, (target_h // 4 + 1, target_w // 4 + 1))[:target_h, :target_w]
        dithered = np.clip(gray + tiled_bayer * 0.35, 0.0, 1.0)
        indices = np.clip((dithered * 3.999).astype(np.int32), 0, 3)
        
        gb_palette = np.array([
            [15, 56, 15],     # Verde muy oscuro (#0f380f)
            [48, 98, 48],     # Verde oliva oscuro (#306230)
            [15, 172, 139],   # Verde lima medio (#8bac0f)
            [15, 218, 155]    # Verde pistacho claro (#9bbc0f)
        ], dtype=np.uint8)
        
        res_small = gb_palette[indices]
        res = cv2.resize(res_small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "GAME BOY DMG-01 // 4-SHADE RETRO DITHERING (160x144)", (15, h - 15), font, scale, (15, 218, 155), 1, cv2.LINE_AA)
        return res

    def _apply_led_matrix_effect(self, img_bgr):
        """Pantalla LED de Estadio: Malla de diodos circulares luminosos rodeados de bisel oscuro"""
        h, w = img_bgr.shape[:2]
        step = max(7, min(w, h) // 55)
        radius = max(2, step // 2 - 1)
        grid_w = w // step
        grid_h = h // step
        
        small = cv2.resize(img_bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        res = np.zeros((h, w, 3), dtype=np.uint8)
        
        for r in range(grid_h):
            cy = r * step + step // 2
            for c in range(grid_w):
                cx = c * step + step // 2
                col = tuple(int(x) for x in small[r, c])
                cv2.circle(res, (cx, cy), radius, col, -1, cv2.LINE_AA)
                spec = tuple(min(255, int(x * 1.35 + 25)) for x in small[r, c])
                if radius > 2:
                    cv2.circle(res, (cx - 1, cy - 1), 1, spec, -1)
                    
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "LED MATRIX DISPLAY // PANEL DE DIODOS DE ESTADIO", (15, h - 15), font, scale, (255, 200, 50), 1, cv2.LINE_AA)
        return res

    def _apply_posterize_hope_effect(self, img_bgr):
        """Posterización Estricta (Póster HOPE): Cuantización tonal a paleta icónica Shepard Fairey"""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        hope_palette = np.array([
            [73, 43, 0],       # Azul Marino (#002b49)
            [33, 26, 215],     # Rojo Carmesí (#d71a21)
            [176, 155, 112],   # Celeste Pastel (#709bb0)
            [196, 235, 252]    # Crema Cálido (#fcebc4)
        ], dtype=np.uint8)
        
        idx = np.clip(gray // 64, 0, 3)
        res = hope_palette[idx]
        
        edges = cv2.Canny(gray, 60, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        res[edges > 0] = [73, 43, 0]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "POSTERIZE HOPE // PROPAGANDA POP ART 4-COLOR PALETTE", (15, h - 15), font, scale, (196, 235, 252), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # F. FILTROS ÓPTICOS Y TEMPORALES DE MOVIMIENTO
    # =========================================================================
    def _apply_motion_blur_effect(self, img_bgr):
        """Desenfoque de Movimiento (Motion Blur): Kernel unidireccional horizontal de alta velocidad"""
        h, w = img_bgr.shape[:2]
        ksize = max(15, min(w, h) // 25)
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[int((ksize - 1) / 2), :] = 1.0 / float(ksize)
        res = cv2.filter2D(img_bgr, -1, kernel)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "MOTION BLUR // DESENFOQUE HORIZONTAL DE ALTA VELOCIDAD", (15, h - 15), font, scale, (200, 240, 255), 1, cv2.LINE_AA)
        return res

    def _apply_zoom_burst_effect(self, img_bgr, center=None):
        """Desenfoque Radial (Zoom Burst): Aceleración cinética e hipervelocidad centrada en el rostro"""
        h, w = img_bgr.shape[:2]
        cx, cy = center if center is not None else (w // 2, h // 2)
        
        scales = np.linspace(1.0, 1.28, 7)
        acc = np.zeros_like(img_bgr, dtype=np.float32)
        for s in scales:
            M = cv2.getRotationMatrix2D((cx, cy), 0, s)
            acc += cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        res = (acc / len(scales)).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "ZOOM BURST // HIPERVELOCIDAD RADIAL KINETICA", (15, h - 15), font, scale, (255, 230, 80), 1, cv2.LINE_AA)
        return res

    def _apply_ghosting_effect(self, img_bgr):
        """Estelas Temporales (Ghosting / Frame Echo): Desfase secuencial y rastro espectral de movimiento"""
        h, w = img_bgr.shape[:2]
        shift = max(10, w // 35)
        g1 = np.roll(img_bgr, shift, axis=1)
        g2 = np.roll(img_bgr, shift * 2, axis=1)
        g3 = np.roll(img_bgr, shift * 3, axis=1)
        
        res = (img_bgr.astype(np.float32) * 0.52 + 
               g1.astype(np.float32) * 0.26 + 
               g2.astype(np.float32) * 0.14 + 
               g3.astype(np.float32) * 0.08).astype(np.uint8)
               
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "GHOSTING ECHO // ESTELAS ESPECTRALES & RASTRO MULTI-FRAME", (15, h - 15), font, scale, (200, 150, 255), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # G. FILTROS DE SEGMENTACIÓN Y AISLAMIENTO DE SUJETO
    # =========================================================================
    def _create_subject_mask(self, img_bgr, landmarks=None, w=None, h=None):
        """Genera una máscara suave del sujeto a partir de los puntos faciales o el centro de la imagen"""
        if w is None or h is None:
            h, w = img_bgr.shape[:2]
            
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks:
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks], dtype=np.int32)
            hull = cv2.convexHull(pts)
            M = cv2.moments(hull)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            x, y, bw, bh = cv2.boundingRect(hull)
            
            center_head = (cx, int(cy - bh * 0.05))
            axes_head = (int(bw * 0.68), int(bh * 0.72))
            cv2.ellipse(mask, center_head, axes_head, 0, 0, 360, 255, -1)
            
            torso_pts = np.array([
                [max(0, cx - int(bw * 0.95)), h],
                [min(w, cx + int(bw * 0.95)), h],
                [min(w, cx + int(bw * 0.70)), min(h, cy + int(bh * 0.50))],
                [max(0, cx - int(bw * 0.70)), min(h, cy + int(bh * 0.50))]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [torso_pts], 255)
            cv2.fillPoly(mask, [hull], 255)
        else:
            cv2.ellipse(mask, (w // 2, int(h * 0.45)), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, 255, -1)
            torso_pts = np.array([
                [max(0, int(w * 0.15)), h],
                [min(w, int(w * 0.85)), h],
                [min(w, int(w * 0.70)), int(h * 0.65)],
                [max(0, int(w * 0.30)), int(h * 0.65)]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [torso_pts], 255)
            
        feather_ksize = max(21, (min(w, h) // 25) | 1)
        feathered = cv2.GaussianBlur(mask, (feather_ksize, feather_ksize), 0).astype(np.float32) / 255.0
        return feathered[:, :, np.newaxis]

    def _apply_green_screen_effect(self, img_bgr, landmarks=None):
        """Segmentación de Fondo (Green Screen): Fondo verde croma #00FF00 de estudio con sujeto intacto"""
        h, w = img_bgr.shape[:2]
        alpha = self._create_subject_mask(img_bgr, landmarks, w, h)
        
        green_bg = np.zeros_like(img_bgr)
        green_bg[:] = (0, 235, 0)
        
        for y in range(0, h, 40):
            cv2.line(green_bg, (0, y), (w, y), (0, 205, 0), 1)
        for x in range(0, w, 40):
            cv2.line(green_bg, (x, 0), (x, h), (0, 205, 0), 1)
            
        res = (img_bgr.astype(np.float32) * alpha + green_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "GREEN SCREEN // CROMA DE ESTUDIO (#00FF00) & SUJETO AISLADO", (15, h - 15), font, scale, (0, 70, 0), 1, cv2.LINE_AA)
        return res

    def _apply_portrait_isolate_effect(self, img_bgr, landmarks=None):
        """Aislamiento de Retrato: Sujeto nítido a todo color con fondo desenfocado en blanco y negro"""
        h, w = img_bgr.shape[:2]
        alpha = self._create_subject_mask(img_bgr, landmarks, w, h)
        
        blur_k = max(25, (min(w, h) // 20) | 1)
        bg_blurred = cv2.GaussianBlur(img_bgr, (blur_k, blur_k), 0)
        bg_gray = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2GRAY)
        bg_bgr = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        bg_bgr = cv2.convertScaleAbs(bg_bgr, alpha=0.85, beta=-10)
        
        res = (img_bgr.astype(np.float32) * alpha + bg_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "PORTRAIT BOKEH // SUJETO A COLOR & FONDO DESENFOCADO B&W", (15, h - 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return res

    # =========================================================================
    # H. FILTROS AVANZADOS: GLITCH DIGITAL & MANGA JAPONÉS
    # =========================================================================
    def _apply_pixel_sorting_effect(self, img_bgr):
        """
        Pixel Sorting (Derretimiento Digital / Cascading Glitch):
        Algoritmo:
        1. Segmentación por luminiscencia: Identifica intervalos continuos de píxeles
           cuyo brillo se encuentra dentro del rango de fusión [65, 220].
        2. Ordenamiento de vecindad unidireccional: En cada columna seleccionada,
           los píxeles se reordenan según su intensidad luminosa Y = 0.299R + 0.587G + 0.114B
           creando un efecto de cascada líquida / derretimiento digital.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        thresh_low = 65
        thresh_high = 220
        mask = (gray >= thresh_low) & (gray <= thresh_high)
        
        sorted_img = img_bgr.copy()
        
        for c in range(0, w, 2):
            col_mask = mask[:, c]
            diff = np.diff(np.concatenate(([False], col_mask, [False])).astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if e - s > 8:
                    slice_lum = gray[s:e, c]
                    sort_idx = np.argsort(slice_lum)
                    sorted_img[s:e, c] = sorted_img[s:e, c][sort_idx]
                    if c + 1 < w:
                        sorted_img[s:e, c + 1] = sorted_img[s:e, c][sort_idx]
                        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(sorted_img, "PIXEL SORTING // DERRETIMIENTO DIGITAL & GLITCH CASCADE", (15, h - 15), font, scale, (100, 255, 240), 1, cv2.LINE_AA)
        return sorted_img

    def _apply_japanese_manga_effect(self, img_bgr):
        """
        Filtro Manga Japonés Clásico (Authentic Screentone Lineart):
        Matemática y pipeline estricto:
        1. Escala de grises: Y = 0.114*B + 0.587*G + 0.299*R
        2. Filtro bilateral para preservar aristas reduciendo ruido de textura (d=7, sigmaColor=50, sigmaSpace=50).
        3. Umbralización adaptativa con media local C=4, block=9 para extraer el lineart de plumilla G-pen.
        4. Morfología de erosión para consolidar el entintado tradicional con tinta china pura.
        5. Aislamiento de sombras profundas (negro sólido < 55) y zonas de sombras medias [55, 165).
        6. Trama de semitonos (screentones): matriz periódica modular (x % step, y % step)
           con distancia euclidiana al centro del punto r(x,y) = sqrt(dx^2 + dy^2). El radio de cada punto
           varía de manera proporcional a la densidad de la sombra I_norm = (165 - Y) / 110.
        7. Blancos puros: Todos los píxeles con Y >= 165 se asignan estrictamente a 255 (papel blanco).
        Resultado final: Rigurosamente binario y monocromático (solo negro 0 y blanco 255).
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1 y 2. Suavizado bilateral preservando bordes duros
        smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        
        # 2. Extracción de lineart nítido emulando entintado con plumilla G-pen
        lineart = cv2.adaptiveThreshold(
            smooth, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            blockSize=9, C=4
        )
        lineart = cv2.erode(lineart, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        
        # 3, 4 y 5. Cuantización de sombras y tramado Screentone
        step = max(5, min(w, h) // 110)
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        center_offset = step / 2.0
        dot_dist = np.sqrt((x_coords % step - center_offset)**2 + (y_coords % step - center_offset)**2)
        
        # Inicializar lienzo en blanco puro (255)
        manga = np.full((h, w), 255, dtype=np.uint8)
        
        # Sombras profundas: negro absoluto (tinta china sólida)
        manga[smooth < 55] = 0
        
        # Sombras medias: trama de puntos (screentone)
        mid_mask = (smooth >= 55) & (smooth < 165)
        shadow_intensity = (165.0 - smooth.astype(np.float32)) / 110.0
        max_dot_radius = center_offset * 0.95
        dot_radius = shadow_intensity * max_dot_radius
        
        screentone_dots = (dot_dist <= dot_radius) & mid_mask
        manga[screentone_dots] = 0
        
        # Superposición de lineart entintado (trazos en negro 0)
        manga[lineart == 0] = 0
        
        # Convertir a 3 canales BGR para compatibilidad total con la UI
        res = cv2.cvtColor(manga, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.34, min(w, h) / 1600.0)
        cv2.putText(res, "JAPANESE MANGA // CLASSIC SCREENTONE & G-PEN LINEART", (15, h - 15), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        return res

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
