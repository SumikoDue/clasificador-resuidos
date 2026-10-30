import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Clasificador automático de residuos sólidos",
                   page_icon="♻️", layout="centered")

NAVY = "#0B1530"
# Mismo orden con el que se entrenó: cardboard, glass, metal, organic, paper, plastic
CLASES = ["Cartón", "Vidrio", "Metal", "Orgánico", "Papel", "Plástico"]


@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("tl_EfficientNetB0.keras", compile=False)


def predecir(archivo):
    img = Image.open(archivo).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.expand_dims(np.array(img, dtype="float32"), 0)  # valores 0-255
    probs = cargar_modelo().predict(x, verbose=0)[0]
    return {c: float(p) for c, p in zip(CLASES, probs)}


st.markdown(f"""
<style>
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ max-width: 820px; padding: 0 0 2rem 0; }}
.stApp {{ background: #ffffff; }}
.encabezado {{ position: relative; background: {NAVY}; height: 190px; overflow: hidden;
    text-align: center; border-radius: 0 0 50% 50% / 0 0 36px 36px; margin-bottom: 18px; }}
.encabezado .icono {{ position: absolute; font-size: 90px; line-height: 1; opacity: .95; }}
.encabezado .titulo {{ position: relative; z-index: 2; margin: 0; padding-top: 44px; color: #fff;
    font-size: 34px; font-weight: 800; letter-spacing: 1px; }}
.encabezado .subtitulo {{ position: relative; z-index: 2; margin: 6px 0 0 0; color: #fff;
    font-size: 15px; font-weight: 600; letter-spacing: 1px; }}
.instruccion {{ color: {NAVY}; font-weight: 700; font-size: 16px; padding: 0 28px; }}
div[data-testid="stVerticalBlock"] > div {{ padding-left: 0; }}
.stButton > button {{ background: {NAVY}; color: #fff; border: 1px solid #3a4a73;
    border-radius: 0; font-weight: 700; font-size: 16px; min-height: 52px; }}
.stButton > button:hover {{ background: #16264f; color: #fff; border-color: #3a4a73; }}
.pie {{ background: {NAVY}; color: #cfd6e6; text-align: center; padding: 34px 12px;
    font-size: 13px; margin-top: 28px; }}
</style>

<div class="encabezado">
  <span class="icono" style="left:1%;  top:-4px;  color:#F4A340;">♻</span>
  <span class="icono" style="left:-2%; top:62px;  color:#B8372B;">♻</span>
  <span class="icono" style="left:22%; top:10px;  color:#1F4E8C;">♻</span>
  <span class="icono" style="left:30%; top:-10px; color:#5C8A3A;">♻</span>
  <span class="icono" style="left:40%; top:-20px; color:#2E86AB;">♻</span>
  <h1 class="titulo">CLASIFICADOR AUTOMÁTICO</h1>
  <p class="subtitulo">RESIDUOS SÓLIDOS</p>
</div>
<p class="instruccion">Introduzca imágenes de RSU de categoría: cartón, vidrio, metal,
orgánico, papel y plástico</p>
""", unsafe_allow_html=True)

if "k" not in st.session_state:
    st.session_state.k = 0
if "res" not in st.session_state:
    st.session_state.res = None

pad1, centro, pad2 = st.columns([0.04, 0.92, 0.04])
with centro:
    tab_subir, tab_camara = st.tabs(["Subir imagen", "Tomar foto"])
    with tab_subir:
        archivo = st.file_uploader("Sube una foto del residuo",
                                   type=["jpg", "jpeg", "png", "webp"],
                                   key=f"up{st.session_state.k}")
    with tab_camara:
        foto = st.camera_input("Toma una foto del residuo",
                               key=f"cam{st.session_state.k}")

    imagen = foto or archivo
    if archivo and not foto:
        st.image(archivo, use_container_width=True)

    c1, c2 = st.columns(2)
    limpiar = c1.button("Limpiar", use_container_width=True)
    enviar = c2.button("Enviar", use_container_width=True)

    if limpiar:
        st.session_state.k += 1
        st.session_state.res = None
        st.rerun()

    if enviar:
        if imagen is None:
            st.warning("Primero sube o toma una foto.")
        else:
            with st.spinner("Clasificando..."):
                st.session_state.res = predecir(imagen)

    if st.session_state.res:
        orden = sorted(st.session_state.res.items(), key=lambda kv: kv[1], reverse=True)
        nombre, p = orden[0]
        if p >= 0.60:
            st.success(f"**Resultado: {nombre}** ({p*100:.1f}% de confianza)")
        else:
            st.warning(f"**Posible: {nombre}** ({p*100:.1f}%). La confianza es baja; "
                       "prueba con mejor luz y un solo objeto en la imagen.")
        for n, pr in orden[:3]:
            st.write(f"{n}: {pr*100:.1f}%")
            st.progress(float(pr))

st.markdown('<div class="pie">Proyecto de investigación · Clasificación automática de '
            'residuos sólidos mediante Deep Learning</div>', unsafe_allow_html=True)