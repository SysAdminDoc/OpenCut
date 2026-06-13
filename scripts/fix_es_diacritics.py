#!/usr/bin/env python3
"""Fix missing Spanish diacritics in UXP es.json locale file."""
import json
import re
import sys

LOCALE_PATH = "extension/com.opencut.uxp/locales/es.json"

# Pattern → replacement pairs for Spanish diacritics.
# Only unambiguous cases where the accent-less form is incorrect Spanish.
WORD_FIXES = [
    # -ción endings (virtually always need accent)
    ("conexion", "conexión"), ("accion", "acción"),
    ("configuracion", "configuración"), ("autorizacion", "autorización"),
    ("informacion", "información"), ("aplicacion", "aplicación"),
    ("seleccion", "selección"), ("deteccion", "detección"),
    ("ejecucion", "ejecución"), ("operacion", "operación"),
    ("exportacion", "exportación"), ("importacion", "importación"),
    ("eliminacion", "eliminación"), ("resolucion", "resolución"),
    ("navegacion", "navegación"), ("instalacion", "instalación"),
    ("normalizacion", "normalización"), ("estabilizacion", "estabilización"),
    ("interpolacion", "interpolación"), ("separacion", "separación"),
    ("sincronizacion", "sincronización"), ("restauracion", "restauración"),
    ("calibracion", "calibración"), ("verificacion", "verificación"),
    ("reproduccion", "reproducción"), ("generacion", "generación"),
    ("transcripcion", "transcripción"), ("localizacion", "localización"),
    ("grabacion", "grabación"), ("posicion", "posición"),
    ("animacion", "animación"), ("duracion", "duración"),
    ("saturacion", "saturación"), ("presentacion", "presentación"),
    ("funcion", "función"), ("seccion", "sección"),
    ("descripcion", "descripción"), ("relacion", "relación"),
    ("creacion", "creación"), ("modificacion", "modificación"),
    ("notificacion", "notificación"), ("cancelacion", "cancelación"),
    ("publicacion", "publicación"), ("organizacion", "organización"),
    ("optimizacion", "optimización"), ("personalizacion", "personalización"),
    ("visualizacion", "visualización"), ("inicializacion", "inicialización"),
    ("combinacion", "combinación"), ("recomendacion", "recomendación"),
    ("desinstalacion", "desinstalación"), ("rotacion", "rotación"),
    ("opcion", "opción"), ("solucion", "solución"),
    ("evaluacion", "evaluación"), ("definicion", "definición"),
    ("identificacion", "identificación"), ("documentacion", "documentación"),
    ("integracion", "integración"), ("migracion", "migración"),
    ("transicion", "transición"), ("produccion", "producción"),
    ("reduccion", "reducción"), ("correccion", "corrección"),
    ("direccion", "dirección"), ("proteccion", "protección"),
    ("proporcion", "proporción"), ("preparacion", "preparación"),
    ("grabacion", "grabación"), ("demostracion", "demostración"),

    # -sión endings
    ("compresion", "compresión"), ("expresion", "expresión"),
    ("extension", "extensión"), ("dimension", "dimensión"),
    ("version", "versión"), ("conversion", "conversión"),
    ("emision", "emisión"), ("transmision", "transmisión"),
    ("expansion", "expansión"), ("precision", "precisión"),
    ("conclusion", "conclusión"), ("inclusion", "inclusión"),
    ("exclusion", "exclusión"), ("revision", "revisión"),

    # Common accented words
    ("linea", "línea"), ("pagina", "página"),
    ("numero", "número"), ("numeros", "números"),
    ("capitulo", "capítulo"), ("capitulos", "capítulos"),
    ("ultimo", "último"), ("ultimos", "últimos"),
    ("ultima", "última"), ("ultimas", "últimas"),
    ("codigo", "código"), ("codigos", "códigos"),
    ("metodo", "método"), ("metodos", "métodos"),
    ("unico", "único"), ("unica", "única"),
    ("minimo", "mínimo"), ("minima", "mínima"),
    ("maximo", "máximo"), ("maxima", "máxima"),
    ("titulo", "título"), ("titulos", "títulos"),
    ("dialogo", "diálogo"),
    ("tambien", "también"), ("ademas", "además"),
    ("despues", "después"),
    ("energia", "energía"),
    ("tamano", "tamaño"),
    ("valido", "válido"), ("valida", "válida"),
    ("invalido", "inválido"), ("invalida", "inválida"),
    ("rapido", "rápido"), ("rapida", "rápida"),
    ("dinamico", "dinámico"), ("dinamica", "dinámica"),
    ("automatico", "automático"), ("automatica", "automática"),
    ("automaticamente", "automáticamente"),
    ("basico", "básico"), ("basica", "básica"),
    ("grafico", "gráfico"), ("grafica", "gráfica"),
    ("musica", "música"),
    ("publico", "público"), ("publica", "pública"),
    ("analisis", "análisis"), ("sintesis", "síntesis"),
    ("diagnostico", "diagnóstico"),
    ("exito", "éxito"),
    ("indice", "índice"),
    ("parametro", "parámetro"), ("parametros", "parámetros"),
    ("practica", "práctica"), ("practico", "práctico"),
    ("tecnica", "técnica"), ("tecnico", "técnico"),
    ("camara", "cámara"), ("camaras", "cámaras"),
    ("imagenes", "imágenes"),
    ("calculo", "cálculo"), ("modulo", "módulo"),
    ("movil", "móvil"),
    ("util", "útil"), ("facil", "fácil"), ("dificil", "difícil"),
    ("perdida", "pérdida"),
    ("credito", "crédito"), ("creditos", "créditos"),

    # ñ words
    ("pestanas", "pestañas"), ("pestana", "pestaña"),
    ("senales", "señales"), ("senal", "señal"),
    ("diseno", "diseño"), ("disenos", "diseños"),
    ("ano", "año"), ("anos", "años"),
    ("espanol", "español"), ("espanola", "española"),

    # Verb forms (only unambiguous contexts)
    ("no esta ", "no está "),
    ("esta disponible", "está disponible"),
    ("esta activo", "está activo"), ("esta activa", "está activa"),
    ("esta en ", "está en "),
    ("esta listo", "está listo"), ("esta lista", "está lista"),
    ("esta conectado", "está conectado"),
    ("esta instalado", "está instalado"),
    ("esta habilitado", "está habilitado"),
    ("esta deshabilitado", "está deshabilitado"),
    ("esta procesando", "está procesando"),
    ("esta ejecutando", "está ejecutando"),
    ("esta cargando", "está cargando"),
    ("esta vacio", "está vacío"),
    ("esta vacia", "está vacía"),
    ("envio ", "envió "),
]


def fix_case(original_word, replacement):
    """Preserve the original word's case pattern."""
    if original_word[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_fixes(text):
    """Apply all diacritics fixes to a string, preserving {placeholder} names."""
    # Extract placeholders, apply fixes to prose, then restore placeholders
    placeholders = {}
    def stash(m):
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = m.group()
        return key
    safe_text = PLACEHOLDER_RE.sub(stash, text)

    for wrong, right in WORD_FIXES:
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        safe_text = pattern.sub(lambda m: fix_case(m.group(), right), safe_text)

    for key, original in placeholders.items():
        safe_text = safe_text.replace(key, original)
    return safe_text


PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def main():
    with open(LOCALE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for key in data:
        if not isinstance(data[key], str):
            continue
        fixed = apply_fixes(data[key])
        if fixed != data[key]:
            data[key] = fixed
            changed += 1

    with open(LOCALE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Stats
    full_text = json.dumps(data, ensure_ascii=False)
    diacritic_chars = "áéíóúñüÁÉÍÓÚÑÜ"
    counts = {c: full_text.count(c) for c in diacritic_chars if full_text.count(c) > 0}

    print(f"Keys modified: {changed}")
    print(f"Diacritics introduced:")
    for char, count in sorted(counts.items()):
        print(f"  {char}: {count}")
    print(f"Total diacritic characters: {sum(counts.values())}")


if __name__ == "__main__":
    main()
