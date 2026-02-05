"""
Biblioteca de Materiales y Secciones Típicas para Ingeniería Mecánica y Civil.
Unidades: SI (Metros, Kilogramos, Segundos, Pascales)

Referencias:
- ASTM Standards
- Eurocode 3 (En 10025) para perfiles
- Mechanics of Materials standard values
"""

STANDARD_MATERIALS = {
    # Aceros
    "ASTM A36 Steel": {
        "E": 200e9,      # 200 GPa
        "nu": 0.26,
        "rho": 7850,     # kg/m3
        "G": 79.3e9,     # ~79 GPa
        "yield_strength": 250e6
    },
    "ASTM A992 Steel (Wide Flange)": {
        "E": 200e9,
        "nu": 0.3,
        "rho": 7850,
        "G": 77e9,
        "yield_strength": 345e6 # 50 ksi
    },
    "Stainless Steel 304": {
        "E": 193e9,
        "nu": 0.29,
        "rho": 8000,
        "G": 75e9,
        "yield_strength": 215e6
    },
    
    # Aluminios
    "Aluminum 6061-T6": {
        "E": 68.9e9,
        "nu": 0.33,
        "rho": 2700,
        "G": 26e9,
        "yield_strength": 276e6
    },
    "Aluminum 2024-T4": {
        "E": 73.1e9,
        "nu": 0.33,
        "rho": 2780,
        "G": 28e9,
        "yield_strength": 324e6
    },

    # Otros Metales
    "Titanium Ti-6Al-4V": {
        "E": 113.8e9,
        "nu": 0.342,
        "rho": 4430,
        "G": 42e9,
        "yield_strength": 880e6
    },
    "Gray Cast Iron ASTM 30": {
        "E": 103e9,
        "nu": 0.27,
        "rho": 7200,
        "G": 41e9,
        "yield_strength": 200e6 # Compressive is higher
    },
    "Copper C11000": {
        "E": 110e9,
        "nu": 0.34,
        "rho": 8890,
        "G": 44e9,
        "yield_strength": 69e6
    }
}

# Propiedades aproximadas para perfiles IPE (European I-Beams)
# Eje Fuerte = Iz (Alrededor de Z local), Eje Débil = Iy (Alrededor de Y local)
# Dimensiones en metros.
STANDARD_SECTIONS = {
    # Perfiles IPE (Vigas I Europeas)
    "IPE 80": {
        "area": 7.64e-4,
        "height": 0.080, "width": 0.046,
        "Iz": 80.1e-8, "Iy": 8.49e-8, "J": 0.70e-8 # J approx Torsion constant
    },
    "IPE 100": {
        "area": 10.3e-4,
        "height": 0.100, "width": 0.055,
        "Iz": 171e-8, "Iy": 15.9e-8, "J": 1.2e-8
    },
    "IPE 120": {
        "area": 13.2e-4,
        "height": 0.120, "width": 0.064,
        "Iz": 318e-8, "Iy": 27.7e-8, "J": 1.74e-8
    },
    "IPE 140": {
        "area": 16.4e-4,
        "height": 0.140, "width": 0.073,
        "Iz": 541e-8, "Iy": 44.9e-8, "J": 2.45e-8
    },
    "IPE 160": {
        "area": 20.1e-4,
        "height": 0.160, "width": 0.082,
        "Iz": 869e-8, "Iy": 68.3e-8, "J": 3.60e-8
    },
    "IPE 200": {
        "area": 28.5e-4,
        "height": 0.200, "width": 0.100,
        "Iz": 1943e-8, "Iy": 142e-8, "J": 6.98e-8
    },
    "IPE 240": {
        "area": 39.1e-4,
        "height": 0.240, "width": 0.120,
        "Iz": 3892e-8, "Iy": 284e-8, "J": 12.9e-8
    },
    "IPE 300": {
        "area": 53.8e-4,
        "height": 0.300, "width": 0.150,
        "Iz": 8356e-8, "Iy": 604e-8, "J": 20.1e-8
    },

    # Perfiles HEB (Vigas H Europeas - Alas Anchas)
    "HEB 100": {
        "area": 26.0e-4,
        "height": 0.100, "width": 0.100,
        "Iz": 450e-8, "Iy": 167e-8, "J": 9.25e-8
    },
    "HEB 140": {
        "area": 43.0e-4,
        "height": 0.140, "width": 0.140,
        "Iz": 1509e-8, "Iy": 550e-8, "J": 20.1e-8
    },
    "HEB 200": {
        "area": 78.1e-4,
        "height": 0.200, "width": 0.200,
        "Iz": 5696e-8, "Iy": 2003e-8, "J": 59.3e-8
    },

    # Perfiles Tubulares Cuadrados (SHS - Square Hollow Sections) - (Lado x Espesor)
    "SHS 40x40x3": {
        "area": 4.41e-4,
        "height": 0.040, "width": 0.040,
        "Iz": 10.1e-8, "Iy": 10.1e-8, "J": 16.5e-8 # J for closed section is much higher
    },
    "SHS 50x50x4": {
        "area": 7.36e-4,
        "height": 0.050, "width": 0.050,
        "Iz": 26.1e-8, "Iy": 26.1e-8, "J": 41.2e-8
    },
    "SHS 100x100x5": {
        "area": 19.0e-4,
        "height": 0.100, "width": 0.100,
        "Iz": 279e-8, "Iy": 279e-8, "J": 435e-8
    },
    
    # Perfiles Tubulares Circulares (CHS - Circular Hollow Sections) - (Diametro x Espesor)
    "CHS 48.3x3.2": { # Tubo estandar de andamio
        "area": 4.53e-4,
        "height": 0.0483, "width": 0.0483, # Diametro
        "Iz": 11.6e-8, "Iy": 11.6e-8, "J": 23.2e-8 # J = 2*I for circular
    },
    "CHS 88.9x4.0": {
        "area": 10.7e-4,
        "height": 0.0889, "width": 0.0889,
        "Iz": 96.3e-8, "Iy": 96.3e-8, "J": 192.6e-8
    }
}

def get_library():
    return {
        "materials": STANDARD_MATERIALS,
        "sections": STANDARD_SECTIONS
    }
