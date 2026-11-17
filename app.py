import os
import re
import sys
import csv
import glob
import cv2
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Importar el motor de análisis facial con suite de filtros (RGB, Mira de Rifle, Quitar Filtro, etc.)
from face_engine import FaceAnalyzer

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class AppReconocimientoFacial:
    def __init__(self, root):
        self.root = root
        self.root.title("BIO-TACTICAL RECONNAISSANCE SYSTEM // HUD v2.7")
        self.root.geometry("1280x900")
        self.root.minsize(1160, 800)
        
        # Archivo para guardar las evaluaciones del usuario
        self.csv_path = "evaluaciones_usuario.csv"
        self._init_csv()
        
        # Cargar motor de IA
        try:
            self.analyzer = FaceAnalyzer(models_dir="models")
        except Exception as e:
            messagebox.showerror("Error al inicializar modelos", f"Error: {e}")
            sys.exit(1)
            
        # Variables de estado
        self.current_img_bgr = None
        self.current_annotated_bgr = None
        self.current_file_path = ""
        self.last_results = []
        self.tk_img_ref = None
        
        # Modo de filtro óptico activo:
        # "NONE", "RGB", "SNIPER", "NIGHT_VISION", "THERMAL", "TERMINATOR", "FACE_MESH", "CCTV", "HUD"
        self.current_filter_mode = "HUD"
        
        # Escaneo de imágenes del dataset del usuario
        self.dataset_images = self._scan_dataset_images()
        self.current_dataset_idx = 0
        
        # Configurar estilos visuales cibernéticos / HUD táctico militar
        self._setup_hud_theme()
        
        # Construir Interfaz Gráfica
        self._build_ui()
        
        # Cargar la primera imagen del dataset al iniciar
        self.root.after(250, self._load_initial_example)

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Fecha_Hora", "Archivo", "Rostro_ID",
                    "Pred_Genero", "Pred_Confianza", "Pred_Color_Ojos", "Pred_Color_Piel",
                    "Val_Genero", "Val_Color_Ojos", "Val_Color_Piel", "Es_Correcto", "Comentarios"
                ])

    def _scan_dataset_images(self):
        files = []
        if os.path.exists("dataset"):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                files.extend(glob.glob(os.path.join("dataset", ext)))
        if not files and os.path.exists("ejemplos_prueba"):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                files.extend(glob.glob(os.path.join("ejemplos_prueba", ext)))
                
        files = sorted(files, key=natural_sort_key)
        return files

    def _setup_hud_theme(self):
        self.c_bg_root = "#080C10"      # Fondo principal ultra oscuro
        self.c_bg_card = "#0D131A"      # Tarjetas tácticas
        self.c_bg_inner = "#121A24"     # Contenedores interiores
        self.c_neon_green = "#39FF14"   # Verde Neón HUD
        self.c_neon_dark = "#173820"    # Verde oscuro de acento
        self.c_border = "#1E2C3A"       # Borde de paneles
        self.c_border_neon = "#22C55E"  # Borde iluminado
        self.c_text_bright = "#F0FDF4"  # Blanco ligeramente verdoso
        self.c_text_dim = "#86EFAC"     # Verde claro secundario
        self.c_text_muted = "#64748B"   # Gris técnico
        self.c_amber = "#F59E0B"        # Naranja de advertencia
        self.c_crimson = "#EF4444"      # Rojo
        
        self.root.configure(bg=self.c_bg_root)
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.style.configure(".", background=self.c_bg_root, foreground=self.c_text_bright, font=("Consolas", 10))
        self.style.configure("TFrame", background=self.c_bg_root)
        self.style.configure("Card.TFrame", background=self.c_bg_card)
        self.style.configure("Inner.TFrame", background=self.c_bg_inner)
        
        self.style.map("TCombobox", 
            fieldbackground=[("readonly", self.c_bg_inner)],
            selectbackground=[("readonly", self.c_neon_dark)],
            selectforeground=[("readonly", self.c_neon_green)],
            foreground=[("readonly", self.c_text_bright)]
        )

    def _build_ui(self):
        # ========================================================
        # 1. ENCABEZADO TÁCTICO HUD (Superior)
        # ========================================================
        header_frame = tk.Frame(self.root, bg=self.c_bg_card, bd=1, relief="solid", highlightthickness=1)
        header_frame.configure(highlightbackground=self.c_border_neon)
        header_frame.pack(fill="x", padx=14, pady=(8, 4), ipady=3, ipadx=10)
        
        lbl_top_brand = tk.Label(
            header_frame, 
            text="[ SYSTEM STATUS: ONLINE // ADVANCED OPTICS & FACIAL WARPING v2.7 ]", 
            bg=self.c_bg_card, fg=self.c_neon_green, font=("Consolas", 11, "bold")
        )
        lbl_top_brand.pack(side="left", padx=5)
        
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.lbl_hud_time = tk.Label(
            header_frame, 
            text=f"SYS TIME: {now_time} | FILTERS: ACTIVE | BIOMETRICS: READY", 
            bg=self.c_bg_card, fg=self.c_text_dim, font=("Consolas", 10)
        )
        self.lbl_hud_time.pack(side="right", padx=5)

        # ========================================================
        # 2. CONTENEDOR PRINCIPAL (2 Columnas Tácticas)
        # ========================================================
        main_container = tk.Frame(self.root, bg=self.c_bg_root)
        main_container.pack(fill="both", expand=True, padx=14, pady=4)
        main_container.columnconfigure(0, weight=6) # Visor HUD
        main_container.columnconfigure(1, weight=5) # Telemetría
        main_container.rowconfigure(0, weight=1)

        # --------------------------------------------------------
        # COLUMNA IZQUIERDA: Visor HUD con Botones 100% FIJOS
        # --------------------------------------------------------
        left_card = tk.Frame(main_container, bg=self.c_bg_card, bd=1, relief="solid", highlightthickness=1)
        left_card.configure(highlightbackground=self.c_border_neon)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        
        # A) Barra de acciones generales (Fijada abajo del todo)
        action_bar = tk.Frame(left_card, bg=self.c_bg_card)
        action_bar.pack(side="bottom", fill="x", padx=10, pady=(4, 8))
        
        btn_load = tk.Button(
            action_bar, text="[ 📂 CARGAR OBJETIVO ]", bg="#122417", fg=self.c_neon_green,
            activebackground=self.c_neon_green, activeforeground="#000000",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=8, pady=3, cursor="hand2",
            command=self.load_image_dialog
        )
        btn_load.pack(side="left", padx=(0, 6))
        
        btn_cam = tk.Button(
            action_bar, text="[ 📷 SENSOR ÓPTICO ]", bg="#16202C", fg="#38BDF8",
            activebackground="#38BDF8", activeforeground="#000000",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=8, pady=3, cursor="hand2",
            command=self.capture_from_webcam
        )
        btn_cam.pack(side="left", padx=(0, 6))
        
        tk.Label(action_bar, text="LOTES:", bg=self.c_bg_card, fg=self.c_text_muted, font=("Consolas", 9)).pack(side="left", padx=(6, 2))
        
        btn_filter_man = tk.Button(
            action_bar, text="HOMBRES (1-12)", bg="#0E1A29", fg="#60A5FA",
            font=("Consolas", 9), relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            command=lambda: self._select_by_keyword("hombre")
        )
        btn_filter_man.pack(side="left", padx=2)
        
        btn_filter_woman = tk.Button(
            action_bar, text="MUJERES (1-18)", bg="#261224", fg="#F472B6",
            font=("Consolas", 9), relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            command=lambda: self._select_by_keyword("mujer")
        )
        btn_filter_woman.pack(side="left", padx=2)

        # B) Barra de navegación del dataset (Fijada encima de las acciones)
        dataset_bar = tk.Frame(left_card, bg=self.c_bg_inner, bd=1, relief="solid", highlightthickness=1)
        dataset_bar.configure(highlightbackground=self.c_border)
        dataset_bar.pack(side="bottom", fill="x", padx=10, pady=(3, 3), ipady=2, ipadx=4)
        
        tk.Label(dataset_bar, text="TARGET FILE:", bg=self.c_bg_inner, fg=self.c_neon_green, font=("Consolas", 9, "bold")).pack(side="left", padx=(6, 4))
        
        img_display_names = [os.path.basename(p) for p in self.dataset_images]
        self.cmb_dataset = ttk.Combobox(dataset_bar, values=img_display_names, state="readonly", width=18)
        if img_display_names:
            self.cmb_dataset.current(0)
        self.cmb_dataset.pack(side="left", padx=4)
        self.cmb_dataset.bind("<<ComboboxSelected>>", self._on_combobox_image_selected)
        
        btn_prev = tk.Button(
            dataset_bar, text="◀ PREV", bg="#152B1E", fg=self.c_neon_green,
            activebackground=self.c_neon_green, activeforeground="#000000",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=6, pady=1, cursor="hand2",
            command=self.prev_dataset_image
        )
        btn_prev.pack(side="left", padx=2)
        
        btn_next = tk.Button(
            dataset_bar, text="NEXT ▶", bg="#152B1E", fg=self.c_neon_green,
            activebackground=self.c_neon_green, activeforeground="#000000",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=6, pady=1, cursor="hand2",
            command=self.next_dataset_image
        )
        btn_next.pack(side="left", padx=2)
        
        btn_rand = tk.Button(
            dataset_bar, text="🎲 RANDOM", bg="#1E1E38", fg="#C084FC",
            font=("Consolas", 9), relief="solid", bd=1, padx=5, pady=1, cursor="hand2",
            command=self.random_dataset_image
        )
        btn_rand.pack(side="left", padx=3)

        # C) APARTADO DE FILTROS (RGB, Quitar Filtro, Mira de Rifle, etc.)
        filter_section = tk.Frame(left_card, bg="#101722", bd=1, relief="solid", highlightthickness=1)
        filter_section.configure(highlightbackground=self.c_border_neon)
        filter_section.pack(side="bottom", fill="x", padx=10, pady=(4, 3), ipady=4, ipadx=4)
        
        # Fila 1: Título y Lista Desplegable + Botón Destacado para Quitar Filtro
        top_filter_bar = tk.Frame(filter_section, bg="#101722")
        top_filter_bar.pack(fill="x", padx=4, pady=(1, 3))
        
        tk.Label(
            top_filter_bar, text="🎯 FILTRO:", 
            bg="#101722", fg=self.c_neon_green, font=("Consolas", 9, "bold")
        ).pack(side="left", padx=(4, 6))
        
        self.filter_options = [
            ("🚫 [QUITAR FILTRO] Foto Original Limpia", "NONE"),
            ("🏜️ Cine Mexicano / Árido (Warm Desert Grade)", "MEXICAN_CINEMA"),
            ("🟣 Duotono Pop (#8A2BE2 x #FF62CB)", "DUOTONE_POP"),
            ("🌑 Dark & Moody (Cinematic Low-Key)", "DARK_MOODY"),
            ("⚪ Monocromo Alto Contraste (Analog B&W)", "HIGH_CONTRAST_BW"),
            ("🌆 Cyberpunk / Neón (Cyan & Magenta)", "CYBERPUNK_NEON"),
            ("🌅 Golden Hour (Puesta de Sol Dorada)", "GOLDEN_HOUR"),
            ("✏️ Boceto a Lápiz (Graphite Sketch)", "SKETCH"),
            ("🎯 Aislamiento de Color (Color Splash)", "COLOR_SPLASH"),
            ("👾 Píxel Art / 8-Bit (Retro Arcade)", "PIXEL_ART"),
            ("📼 Cinta VHS / Glitch Retro (NTSC)", "VHS"),
            ("🌈 Desfase RGB (3D Chroma Glitch)", "RGB"),
            ("🎯 Mira de Rifle (Sniper Red Scope)", "SNIPER"),
            ("🟢 Visión Nocturna NVG (Green Phosphor)", "NIGHT_VISION"),
            ("🔥 Termografía FLIR (Predator Heatmap)", "THERMAL"),
            ("🤖 Terminator T-800 (Cyber Red HUD)", "TERMINATOR"),
            ("🕸️ Malla Holográfica 3D (Wireframe)", "FACE_MESH"),
            ("📹 Cámara CCTV 90s (Security Feed)", "CCTV"),
            ("⚡ HUD Táctico (Cyberpunk Verde)", "HUD")
        ]
        
        self.cmb_filters = ttk.Combobox(
            top_filter_bar, 
            values=[opt[0] for opt in self.filter_options], 
            state="readonly", width=38
        )
        # Iniciar con HUD (índice 18)
        self.cmb_filters.current(18)
        self.cmb_filters.pack(side="left", padx=4)
        self.cmb_filters.bind("<<ComboboxSelected>>", self._on_filter_combobox_selected)
        
        # Botón rápido para QUITAR FILTRO con 1 solo clic
        btn_clear_filter = tk.Button(
            top_filter_bar, text="[ 🚫 QUITAR FILTRO ]", bg="#371717", fg="#FCA5A5",
            activebackground="#EF4444", activeforeground="#FFFFFF",
            font=("Consolas", 8, "bold"), relief="solid", bd=1, padx=6, pady=1, cursor="hand2",
            command=lambda: self.set_filter_mode("NONE")
        )
        btn_clear_filter.pack(side="left", padx=(6, 2))
        
        # Fila 2: Botonera directa - Grupo Color, Tono y Estéticos
        quick_filters_bar1 = tk.Frame(filter_section, bg="#101722")
        quick_filters_bar1.pack(fill="x", padx=4, pady=(2, 1))
        
        buttons_spec1 = [
            ("🚫 ORIGINAL", "NONE", "#27272A", "#E4E4E7"),
            ("🏜️ ÁRIDO", "MEXICAN_CINEMA", "#3D2714", "#FBBF24"),
            ("🟣 DUOTONO", "DUOTONE_POP", "#2E1065", "#F472B6"),
            ("🌑 MOODY", "DARK_MOODY", "#111827", "#9CA3AF"),
            ("⚪ B&W ALTO", "HIGH_CONTRAST_BW", "#1F2937", "#F3F4F6"),
            ("🌆 CYBER", "CYBERPUNK_NEON", "#1E1B4B", "#38BDF8"),
            ("🌅 GOLDEN", "GOLDEN_HOUR", "#451A03", "#F59E0B"),
            ("✏️ SKETCH", "SKETCH", "#374151", "#E5E7EB")
        ]
        
        # Fila 3: Botonera directa - Grupo Computacionales y Tácticos
        quick_filters_bar2 = tk.Frame(filter_section, bg="#101722")
        quick_filters_bar2.pack(fill="x", padx=4, pady=(1, 1))
        
        buttons_spec2 = [
            ("🎯 SPLASH", "COLOR_SPLASH", "#450A0A", "#F87171"),
            ("👾 8-BIT", "PIXEL_ART", "#064E3B", "#34D399"),
            ("📼 VHS", "VHS", "#312E81", "#A5B4FC"),
            ("🌈 RGB", "RGB", "#1A2542", "#67E8F9"),
            ("🎯 RIFLE", "SNIPER", "#450A0A", "#F87171"),
            ("🟢 NVG", "NIGHT_VISION", "#0F2E1B", "#4ADE80"),
            ("🔥 TÉRMICO", "THERMAL", "#3B1804", "#FB923C"),
            ("🤖 T-800", "TERMINATOR", "#3A0D0D", "#FCA5A5"),
            ("⚡ HUD", "HUD", "#1E293B", "#94A3B8")
        ]
        
        self.filter_buttons = {}
        for label, mode_key, bg_col, fg_col in buttons_spec1:
            b = tk.Button(
                quick_filters_bar1, text=label, bg=bg_col, fg=fg_col,
                font=("Consolas", 8, "bold"), relief="solid", bd=1, padx=3, pady=1, cursor="hand2",
                command=lambda m=mode_key: self.set_filter_mode(m)
            )
            b.pack(side="left", padx=2)
            self.filter_buttons[mode_key] = b

        for label, mode_key, bg_col, fg_col in buttons_spec2:
            b = tk.Button(
                quick_filters_bar2, text=label, bg=bg_col, fg=fg_col,
                font=("Consolas", 8, "bold"), relief="solid", bd=1, padx=3, pady=1, cursor="hand2",
                command=lambda m=mode_key: self.set_filter_mode(m)
            )
            b.pack(side="left", padx=2)
            self.filter_buttons[mode_key] = b

        # D) Barra superior del visor táctico
        top_visor_bar = tk.Frame(left_card, bg=self.c_bg_card)
        top_visor_bar.pack(side="top", fill="x", padx=10, pady=(6, 4))
        
        self.lbl_file_info = tk.Label(top_visor_bar, text=">> SCANNING TARGET...", bg=self.c_bg_card, fg=self.c_neon_green, font=("Consolas", 10, "bold"))
        self.lbl_file_info.pack(side="left")
        
        self.lbl_file_counter = tk.Label(top_visor_bar, text="", bg=self.c_bg_card, fg=self.c_text_muted, font=("Consolas", 9))
        self.lbl_file_counter.pack(side="right")
        
        # E) Canvas central para el visor HUD táctico y los filtros
        self.image_canvas = tk.Canvas(
            left_card, 
            bg="#05070A", 
            highlightthickness=1, 
            highlightbackground="#16251C"
        )
        self.image_canvas.pack(side="top", fill="both", expand=True, padx=10, pady=4)
        self.image_canvas.bind("<Configure>", self._on_canvas_resize)

        # --------------------------------------------------------
        # COLUMNA DERECHA: Telemetría & Módulo de Evaluación
        # --------------------------------------------------------
        right_card = tk.Frame(main_container, bg=self.c_bg_card, bd=1, relief="solid", highlightthickness=1)
        right_card.configure(highlightbackground=self.c_border_neon)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        
        # CARD 3: TELEMETRÍA Y LOGS DEL SISTEMA (Anclado al fondo derecho)
        card_stats = tk.Frame(right_card, bg=self.c_bg_inner, bd=1, relief="solid", highlightthickness=1)
        card_stats.configure(highlightbackground=self.c_border)
        card_stats.pack(side="bottom", fill="x", padx=10, pady=(4, 8), ipady=3, ipadx=8)
        
        tk.Label(card_stats, text="[ RECON DATABASE // LOGS ]", bg=self.c_bg_inner, fg=self.c_neon_green, font=("Consolas", 9, "bold")).pack(anchor="w", padx=6, pady=(1, 1))
        
        self.lbl_stats = tk.Label(card_stats, text="Cargando telemetría...", bg=self.c_bg_inner, fg=self.c_text_dim, font=("Consolas", 9))
        self.lbl_stats.pack(anchor="w", padx=6)
        
        btn_hist_bar = tk.Frame(card_stats, bg=self.c_bg_inner)
        btn_hist_bar.pack(fill="x", padx=6, pady=(3, 2))
        
        btn_open_csv = tk.Button(
            btn_hist_bar, text="[ ABRIR LOG CSV ]", bg="#132319", fg=self.c_neon_green,
            font=("Consolas", 9), relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            command=self.open_evaluations_csv
        )
        btn_open_csv.pack(side="left", padx=(0, 6))
        
        btn_retrain = tk.Button(
            btn_hist_bar, text="[ 🧠 ENTRENAR CLASIFICADOR ML ]", bg="#201A38", fg="#A78BFA",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=8, pady=2, cursor="hand2",
            command=self.run_ml_training
        )
        btn_retrain.pack(side="left")

        # CARD 1: PREDICCIÓN BIOMÉTRICA DE LA IA
        card_pred = tk.Frame(right_card, bg=self.c_bg_inner, bd=1, relief="solid", highlightthickness=1)
        card_pred.configure(highlightbackground=self.c_border)
        card_pred.pack(side="top", fill="x", padx=10, pady=(8, 4), ipady=4, ipadx=8)
        
        tk.Label(card_pred, text="[ BIOMETRIC SCAN TELEMETRY ]", bg=self.c_bg_inner, fg=self.c_neon_green, font=("Consolas", 11, "bold")).pack(anchor="w", padx=8, pady=(2, 4))
        
        grid_res = tk.Frame(card_pred, bg=self.c_bg_inner)
        grid_res.pack(fill="x", padx=8)
        grid_res.columnconfigure(1, weight=1)
        
        tk.Label(grid_res, text="GENDER:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_gender = tk.Label(grid_res, text="WAITING FOR SCAN...", bg=self.c_bg_inner, fg=self.c_neon_green, font=("Consolas", 11, "bold"))
        self.lbl_gender.grid(row=0, column=1, sticky="w", padx=8, pady=2)
        
        tk.Label(grid_res, text="OCULAR TONE:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 10, "bold")).grid(row=1, column=0, sticky="w", pady=2)
        eye_frame = tk.Frame(grid_res, bg=self.c_bg_inner)
        eye_frame.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        self.swatch_eye = tk.Canvas(eye_frame, width=18, height=18, bg="#2D3748", highlightthickness=1, highlightbackground=self.c_neon_green)
        self.swatch_eye.pack(side="left", padx=(0, 6))
        self.lbl_eye = tk.Label(eye_frame, text="-", bg=self.c_bg_inner, fg=self.c_text_bright, font=("Consolas", 10))
        self.lbl_eye.pack(side="left")
        
        tk.Label(grid_res, text="SKIN CHROMATICS:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 10, "bold")).grid(row=2, column=0, sticky="w", pady=2)
        skin_frame = tk.Frame(grid_res, bg=self.c_bg_inner)
        skin_frame.grid(row=2, column=1, sticky="w", padx=8, pady=2)
        self.swatch_skin = tk.Canvas(skin_frame, width=18, height=18, bg="#2D3748", highlightthickness=1, highlightbackground=self.c_neon_green)
        self.swatch_skin.pack(side="left", padx=(0, 6))
        self.lbl_skin = tk.Label(skin_frame, text="-", bg=self.c_bg_inner, fg=self.c_text_bright, font=("Consolas", 10))
        self.lbl_skin.pack(side="left")
        
        tk.Label(grid_res, text="AGE ESTIMATE:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 10, "bold")).grid(row=3, column=0, sticky="w", pady=2)
        self.lbl_age = tk.Label(grid_res, text="-", bg=self.c_bg_inner, fg=self.c_text_bright, font=("Consolas", 10))
        self.lbl_age.grid(row=3, column=1, sticky="w", padx=8, pady=2)

        # CARD 2: EVALUACIÓN Y VALIDACIÓN DEL USUARIO ("YO VALÚO LA RESPUESTA")
        card_eval = tk.Frame(right_card, bg=self.c_bg_card, bd=1, relief="solid", highlightthickness=1)
        card_eval.configure(highlightbackground=self.c_border)
        card_eval.pack(side="top", fill="both", expand=True, padx=10, pady=(4, 4), ipady=3, ipadx=8)
        
        tk.Label(card_eval, text="[ HUMAN VERIFICATION // OPERATOR VALUATION ]", bg=self.c_bg_card, fg=self.c_neon_green, font=("Consolas", 11, "bold")).pack(anchor="w", padx=8, pady=(2, 1))
        tk.Label(
            card_eval, 
            text="Valida si la telemetría es exacta o anula con valores manuales para retroalimentar el sistema:", 
            bg=self.c_bg_card, fg=self.c_text_muted, font=("Segoe UI", 9), wraplength=420, justify="left"
        ).pack(anchor="w", padx=8, pady=(0, 4))
        
        quick_btn_frame = tk.Frame(card_eval, bg=self.c_bg_card)
        quick_btn_frame.pack(fill="x", padx=8, pady=(0, 4))
        
        self.btn_correct = tk.Button(
            quick_btn_frame, text="[ ✔ APROBAR PREDICCIÓN ]", bg="#0F3820", fg=self.c_neon_green,
            activebackground=self.c_neon_green, activeforeground="#000000",
            font=("Consolas", 10, "bold"), relief="solid", bd=1, padx=10, pady=4, cursor="hand2",
            command=self.on_approve_prediction
        )
        self.btn_correct.pack(side="left", padx=(0, 8))
        
        self.btn_wrong = tk.Button(
            quick_btn_frame, text="[ ✖ MANUAL OVERRIDE ]", bg="#3B2610", fg=self.c_amber,
            activebackground=self.c_amber, activeforeground="#000000",
            font=("Consolas", 10, "bold"), relief="solid", bd=1, padx=10, pady=4, cursor="hand2",
            command=self.on_enable_correction
        )
        self.btn_wrong.pack(side="left")
        
        self.form_frame = tk.Frame(card_eval, bg=self.c_bg_inner, bd=1, relief="solid", highlightthickness=1)
        self.form_frame.configure(highlightbackground=self.c_border)
        self.form_frame.pack(fill="x", padx=8, pady=2, ipady=3, ipadx=6)
        
        tk.Label(self.form_frame, text="VALORES REALES VALIDADOS:", bg=self.c_bg_inner, fg=self.c_neon_green, font=("Consolas", 9, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(1, 3))
        
        tk.Label(self.form_frame, text="GÉNERO REAL:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 9)).grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.cmb_real_gender = ttk.Combobox(self.form_frame, values=["Hombre", "Mujer"], state="readonly", width=20)
        self.cmb_real_gender.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        
        tk.Label(self.form_frame, text="OJOS REALES:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 9)).grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.cmb_real_eyes = ttk.Combobox(
            self.form_frame, 
            values=["Marrón / Castaño", "Marrón Oscuro", "Negro / Ojo muy oscuro", "Azul", "Verde", "Miel / Ámbar / Hazel", "Gris"], 
            state="readonly", width=20
        )
        self.cmb_real_eyes.grid(row=2, column=1, sticky="w", padx=6, pady=2)
        
        tk.Label(self.form_frame, text="PIEL REAL:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 9)).grid(row=3, column=0, sticky="w", padx=6, pady=2)
        self.cmb_real_skin = ttk.Combobox(
            self.form_frame, 
            values=["Muy Claro / Pálido", "Claro / Beige", "Intermedio / Trigueño", "Moreno / Bronceado", "Oscuro / Ébano"], 
            state="readonly", width=20
        )
        self.cmb_real_skin.grid(row=3, column=1, sticky="w", padx=6, pady=2)
        
        tk.Label(self.form_frame, text="NOTAS:", bg=self.c_bg_inner, fg=self.c_text_muted, font=("Consolas", 9)).grid(row=4, column=0, sticky="w", padx=6, pady=2)
        self.txt_notes = tk.Entry(self.form_frame, width=23, font=("Consolas", 9), bg="#0B1017", fg=self.c_text_bright, insertbackground=self.c_neon_green)
        self.txt_notes.grid(row=4, column=1, sticky="w", padx=6, pady=2)
        
        self.btn_save_eval = tk.Button(
            card_eval, text="[ 💾 GUARDAR EVALUACIÓN ]", bg="#153B25", fg=self.c_neon_green,
            activebackground=self.c_neon_green, activeforeground="#000000",
            font=("Consolas", 9, "bold"), relief="solid", bd=1, padx=10, pady=3, cursor="hand2",
            command=self.save_user_evaluation
        )
        self.btn_save_eval.pack(anchor="w", padx=8, pady=(3, 2))
        
        self.lbl_eval_status = tk.Label(card_eval, text="", bg=self.c_bg_card, fg=self.c_neon_green, font=("Consolas", 9))
        self.lbl_eval_status.pack(anchor="w", padx=8)

        self._update_stats_display()

    # =========================================================================
    # CONTROL DE FILTROS ÓPTICOS Y DEFORMACIONES
    # =========================================================================
    def _on_filter_combobox_selected(self, event=None):
        idx = self.cmb_filters.current()
        if 0 <= idx < len(self.filter_options):
            mode_key = self.filter_options[idx][1]
            self.set_filter_mode(mode_key)

    def set_filter_mode(self, mode_key):
        """Cambia el modo de filtro activo y actualiza la vista inmediatamente"""
        self.current_filter_mode = mode_key
        
        # Sincronizar combobox
        for i, opt in enumerate(self.filter_options):
            if opt[1] == mode_key:
                self.cmb_filters.current(i)
                break
                
        # Reprocesar la imagen actual con el nuevo filtro
        if self.current_file_path:
            self.process_image_path(self.current_file_path)

    def _load_initial_example(self):
        if self.dataset_images:
            self._load_dataset_index(0)

    def _on_canvas_resize(self, event=None):
        if self.current_annotated_bgr is not None:
            self._display_image(self.current_annotated_bgr)

    def _on_combobox_image_selected(self, event=None):
        idx = self.cmb_dataset.current()
        if 0 <= idx < len(self.dataset_images):
            self._load_dataset_index(idx)

    def prev_dataset_image(self):
        if not self.dataset_images:
            return
        new_idx = (self.current_dataset_idx - 1) % len(self.dataset_images)
        self._load_dataset_index(new_idx)

    def next_dataset_image(self):
        if not self.dataset_images:
            return
        new_idx = (self.current_dataset_idx + 1) % len(self.dataset_images)
        self._load_dataset_index(new_idx)

    def random_dataset_image(self):
        import random
        if not self.dataset_images:
            return
        new_idx = random.randint(0, len(self.dataset_images) - 1)
        self._load_dataset_index(new_idx)

    def _select_by_keyword(self, keyword):
        for idx, p in enumerate(self.dataset_images):
            if keyword.lower() in os.path.basename(p).lower():
                self._load_dataset_index(idx)
                break

    def _load_dataset_index(self, idx):
        self.current_dataset_idx = idx
        path = self.dataset_images[idx]
        self.cmb_dataset.current(idx)
        fname = os.path.basename(path)
        self.lbl_file_info.config(text=f">> TARGET: {fname.upper()}")
        self.lbl_file_counter.config(text=f"[{idx + 1} / {len(self.dataset_images)}]")
        self.process_image_path(path)

    def load_image_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen de Rostro",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            self.lbl_file_info.config(text=f">> EXTERNAL: {os.path.basename(file_path).upper()}")
            self.lbl_file_counter.config(text="[EXT]")
            self.process_image_path(file_path)

    def capture_from_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showwarning("Sensor no disponible", "No se detectó un dispositivo de video.")
            return
            
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            temp_path = "captura_webcam.jpg"
            cv2.imwrite(temp_path, frame)
            self.lbl_file_info.config(text=">> LIVE OPTICAL SENSOR")
            self.lbl_file_counter.config(text="[WEBCAM]")
            self.process_image_path(temp_path)
        else:
            messagebox.showerror("Error", "Fallo al capturar fotograma del sensor.")

    def process_image_path(self, path):
        if not os.path.exists(path):
            return
            
        self.current_file_path = path
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {path}")
            return
            
        self.current_img_bgr = img_bgr
        
        # Procesar con el motor de IA aplicando el filtro activo
        annotated_bgr, results = self.analyzer.analyze_image(
            img_bgr, 
            filter_mode=self.current_filter_mode
        )
        self.current_annotated_bgr = annotated_bgr
        self.last_results = results
        
        # Mostrar imagen en el Canvas
        self._display_image(annotated_bgr)
        
        # Actualizar datos de telemetría
        self._update_prediction_panel(results)
        self.lbl_eval_status.config(text="")

    def _display_image(self, img_bgr):
        self.root.update_idletasks()
        
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()
        
        if canvas_w < 50:
            canvas_w = 540
        if canvas_h < 50:
            canvas_h = 440
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        orig_w, orig_h = pil_img.size
        
        ratio = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))
        
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_w // 2, canvas_h // 2, image=tk_img, anchor="center")
        self.tk_img_ref = tk_img

    def _update_prediction_panel(self, results):
        if not results:
            self.lbl_gender.config(text="[ NO TARGET ACQUIRED ]", fg="#EF4444")
            self.lbl_eye.config(text="-")
            self.lbl_skin.config(text="-")
            self.lbl_age.config(text="-")
            self.swatch_eye.config(bg="#1E293B")
            self.swatch_skin.config(bg="#1E293B")
            return
            
        res = results[0]
        
        g = res["gender"]
        conf = res["gender_confidence"]
        g_color = "#38BDF8" if g == "Hombre" else "#F472B6"
        icon = "👨" if g == "Hombre" else "👩"
        self.lbl_gender.config(text=f"{icon} {g.upper()} [{conf}% CONFIDENCE]", fg=g_color)
        
        self.lbl_eye.config(text=f"{res['eye_color']} ({res['eye_hex']})")
        self.swatch_eye.config(bg=res["eye_hex"])
        
        self.lbl_skin.config(text=f"{res['skin_tone']} ({res['skin_hex']})")
        self.swatch_skin.config(bg=res["skin_hex"])
        
        self.lbl_age.config(text=f"~{res['estimated_age']} YEARS")
        
        self.cmb_real_gender.set(g)
        self.cmb_real_eyes.set(res["eye_color"])
        self.cmb_real_skin.set(res["skin_tone"])
        self.txt_notes.delete(0, tk.END)

    def on_approve_prediction(self):
        if not self.last_results:
            messagebox.showinfo("Aviso", "No hay objetivo analizado.")
            return
            
        res = self.last_results[0]
        self._record_evaluation(
            is_correct="Sí",
            val_gender=res["gender"],
            val_eyes=res["eye_color"],
            val_skin=res["skin_tone"],
            comment="Aprobado por el usuario (100% de acuerdo)"
        )

    def on_enable_correction(self):
        self.lbl_eval_status.config(
            text=">> INGRESA VALORES MANUALES Y PRESIONA GUARDAR.",
            fg=self.c_amber
        )

    def save_user_evaluation(self):
        if not self.last_results:
            messagebox.showinfo("Aviso", "No hay objetivo analizado.")
            return
            
        res = self.last_results[0]
        val_g = self.cmb_real_gender.get().strip()
        val_e = self.cmb_real_eyes.get().strip()
        val_s = self.cmb_real_skin.get().strip()
        notes = self.txt_notes.get().strip()
        
        if not val_g or not val_e or not val_s:
            messagebox.showwarning("Incompleto", "Por favor completa género, ojos y piel reales.")
            return
            
        is_exact = (val_g == res["gender"] and val_e == res["eye_color"] and val_s == res["skin_tone"])
        is_correct_str = "Sí" if is_exact else "Parcial / Corregido"
        
        self._record_evaluation(
            is_correct=is_correct_str,
            val_gender=val_g,
            val_eyes=val_e,
            val_skin=val_s,
            comment=notes if notes else ("Corregido manualmente" if not is_exact else "Confirmado")
        )

    def _record_evaluation(self, is_correct, val_gender, val_eyes, val_skin, comment):
        res = self.last_results[0]
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname = os.path.basename(self.current_file_path) if self.current_file_path else "captura.jpg"
        
        row = [
            now_str, fname, res["face_id"],
            res["gender"], res["gender_confidence"], res["eye_color"], res["skin_tone"],
            val_gender, val_eyes, val_skin, is_correct, comment
        ]
        
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        self.lbl_eval_status.config(
            text=f">> TELEMETRY LOGGED TO DATABASE ({now_str})",
            fg=self.c_neon_green
        )
        self._update_stats_display()

    def _update_stats_display(self):
        if not os.path.exists(self.csv_path):
            self.lbl_stats.config(text="LOGGED TARGETS: 0")
            return
            
        total = 0
        correctos = 0
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                total += 1
                if r.get("Es_Correcto") == "Sí":
                    correctos += 1
                    
        if total == 0:
            self.lbl_stats.config(text="Aún no hay registros en la base de datos.")
        else:
            acc = round((correctos / total) * 100, 1)
            self.lbl_stats.config(text=f"TOTAL SCANS: {total} | VERIFIED MATCHES: {correctos} | ACCURACY: {acc}%")

    def open_evaluations_csv(self):
        if not os.path.exists(self.csv_path):
            messagebox.showinfo("Log", "No hay registros generados.")
            return
        try:
            os.startfile(self.csv_path)
        except Exception:
            messagebox.showinfo("Log", f"Archivo en:\n{os.path.abspath(self.csv_path)}")

    def run_ml_training(self):
        import train_supervised_model
        msg = train_supervised_model.train_all_models()
        messagebox.showinfo("Machine Learning Engine", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppReconocimientoFacial(root)
    root.mainloop()
