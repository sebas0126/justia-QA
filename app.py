import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Ocultar advertencias
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(page_title="JustIA - QA", page_icon="⚖️", layout="centered")

st.title("⚖️ JustIA — QA")
st.markdown("""
DEMO web que implementa una arquitectura **RAG** para responder 
preguntas jurídicas basadas en la normativa colombiana vigente.
""")

# Base de datos juridica
@st.cache_resource
def inicializar_base_conocimiento():
    # Corpus optimizado con anclas semánticas para evitar colisiones entre ramas
    documentos = [
        # --- DERECHO LABORAL ---
        {
            "id": "LEY_1010_2006", 
            "tema": "Acoso Laboral", 
            "texto": "La Ley 1010 de 2006 define el acoso laboral como conducta persistente de maltrato, donde el jefe grita, intimida o insulta en la oficina para infundir miedo o terror. Las víctimas pueden denunciar ante el Comité de Convivencia y tienen fuero de protección contra despido."
        },
        {
            "id": "CST_ART_64", 
            "tema": "Despido Sin Justa Causa", 
            "texto": "El Código Sustantivo del Trabajo establece la indemnización por despido sin justa causa. Si la empresa echa al trabajador de forma injustificada, se calcula un pago de liquidación monetaria según el tiempo laborado y tipo de contrato fijo o indefinido."
        },
        # --- DERECHO CIVIL ---
        {
            "id": "LEY_820_2003", 
            "tema": "Arrendamiento de Vivienda", 
            "texto": "La Ley 820 de 2003 dicta el arriendo de vivienda urbana. El dueño o arrendador tiene derecho a pedir la restitución si el inquilino no paga. Queda prohibido que el dueño de casa corte los servicios públicos unilateralmente o retenga bienes por retraso."
        },
        {
            "id": "CODIGO_CIVIL_PROMESA", 
            "tema": "Contratos y Promesas de Compraventa", 
            "texto": "Para que una promesa de compraventa de un inmueble, casa o apartamento sea válida legalmente, debe constar en un contrato por escrito que contenga precio, datos del bien, y fije la fecha y hora exacta para firmar la escritura pública en la notaría."
        },
        # --- DERECHO DE FAMILIA ---
        {
            "id": "LEY_1098_2006", 
            "tema": "Cuota Alimentaria para Menores", 
            "texto": "El Código de Infancia y Adolescencia obliga a pagar la cuota alimentaria para hijos menores. Cubre de forma obligatoria los gastos de manutención, vivienda, vestido, salud y educación de los niños, fijada por un juez de familia o en conciliación."
        },
        {
            "id": "CODIGO_CIVIL_DIVORCIO", 
            "tema": "Divorcio y Separación", 
            "texto": "El divorcio y la separación de cuerpos disuelve el matrimonio civil. Puede ser de mutuo acuerdo ante notario (divorcio exprés) o contencioso ante juez por infidelidad o maltrato. Si hay hijos menores de edad, interviene obligatoriamente el Defensor de Familia."
        },
        # --- DERECHO PENAL ---
        {
            "id": "CODIGO_PENAL_ART_239", 
            "tema": "Delito de Hurto", 
            "texto": "El delito de hurto consiste en robar o apoderarse de una cosa mueble ajena. La pena de prisión se agrava a hurto calificado si se emplea violencia física, armas de fuego, atraco o si se rompen las cerraduras y puertas para entrar."
        },
        {
            "id": "CODIGO_PENAL_ART_32", 
            "tema": "Legítima Defensa", 
            "texto": "La legítima defensa exime de responsabilidad penal por lesiones o muerte. Requiere defender un derecho propio de una agresión injusta, actual o inminente, siempre que la reacción y la fuerza utilizada para el contraataque sean proporcionales."
        },
        # --- DERECHO CONSTITUCIONAL ---
        {
            "id": "DECRETO_2591_1991", 
            "tema": "Acción de Tutela", 
            "texto": "La acción de tutela protege derechos fundamentales vulnerados por una autoridad. No aplica para pensiones o contratos comunes, sino para amparar de urgencia la vida o la salud. El juez constitucional tiene un término de 10 días para resolver el fallo."
        },
        # --- DERECHO COMERCIAL / CONSUMIDOR ---
        {
            "id": "LEY_1480_2011", 
            "tema": "Derecho de Retracto (Consumidor)", 
            "texto": "El Estatuto del Consumidor otorga el derecho de retracto para compras por internet o catálogos online. El cliente tiene un plazo de 5 días hábiles para devolver el producto y solicitar el reembolso del 100% de su dinero si se arrepiente."
        },
        # --- DOCUMENTO ESPECÍFICO PARA EL REQUISITO DE PENSIONES ---
        {
            "id": "LEY_100_1993", 
            "tema": "Sistema de Pensiones y Jubilación", 
            "texto": "Para obtener la pensión de vejez y jubilación bajo el régimen de Colpensiones, los requisitos obligatorios de semanas y edad son: haber cumplido 57 años de edad si es mujer o 62 años de edad si es hombre, y registrar un mínimo de 1300 semanas cotizadas."
        }
    ]
    return documentos

documentos = inicializar_base_conocimiento()
textos_corpus = [doc["texto"] for doc in documentos]

# Cargar modelos de IA con caching para optimizar rendimiento
@st.cache_resource
def cargar_modelos_ia():
    # Modelo Retriever: Codificador semántico para búsqueda
    retriever = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings_corpus = retriever.encode(textos_corpus)
    
    # Modelo Reader: Comprensión lectora
    modelo_qa_id = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
    tokenizer = AutoTokenizer.from_pretrained(modelo_qa_id)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(modelo_qa_id)
    
    return retriever, embeddings_corpus, tokenizer, qa_model

with st.spinner("Inicializando modelos..."):
    retriever, embeddings_corpus, tokenizer, qa_model = cargar_modelos_ia()


# Modulo de inferencia
def procesar_consulta_rag(pregunta):
    # Retriever
    vector_pregunta = retriever.encode([pregunta.lower().strip()])
    similitudes = cosine_similarity(vector_pregunta, embeddings_corpus)[0]
    indice_mejor_doc = np.argmax(similitudes)
    puntaje_similitud = similitudes[indice_mejor_doc]
    
    # Guardar trazas del flujo para explicabilidad algorítmica
    trazas = {
        "similitud": puntaje_similitud,
        "doc_id": documentos[indice_mejor_doc]["id"],
        "tema": documentos[indice_mejor_doc]["tema"]
    }
    
    # Control ético para evitar alucinaciones
    if puntaje_similitud < 0.30:
        return "La consulta se encuentra fuera del dominio de JustIA.", trazas, None

    documento_contexto = documentos[indice_mejor_doc]
    
    # Reader
    inputs = tokenizer(pregunta, documento_contexto["texto"], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    tokens = inputs.input_ids[0][start_idx:end_idx + 1]
    texto_extraido = tokenizer.decode(tokens, skip_special_tokens=True)
    confianza = torch.max(torch.softmax(outputs.start_logits, dim=-1)).item()
    
    trazas["confianza_reader"] = confianza
    return texto_extraido.capitalize(), trazas, documento_contexto["texto"]

# Interfaz de consulta para el usuario

pregunta_usuario = st.text_input(
    "Ingrese su pregunta jurídica:", 
    placeholder="Ej: ¿Cuáles son los requisitos de semanas y edad para pensionarse?"
)

if st.button("Consultar a JustIA"):
    if pregunta_usuario.strip():
        respuesta, log_trazabilidad, texto_fuente = procesar_consulta_rag(pregunta_usuario)
        
        # Resultados de la consulta
        st.subheader("Respuesta extraída:")
        st.success(respuesta)
        
        # Panel explicabilidad algorítmica
        with st.expander("Trazabilidad algorítmica y fuente de datos"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Similitud del buscador semántico", value=f"{round(log_trazabilidad['similitud'] * 100, 2)}%")
            with col2:
                if "confianza_reader" in log_trazabilidad:
                    st.metric(label="Confianza de extracción ", value=f"{round(log_trazabilidad['confianza_reader'] * 100, 2)}%")
            
            st.markdown(f"**Fuente de datos normativa:** `{log_trazabilidad['doc_id']}` ({log_trazabilidad['tema']})")
            if texto_fuente:
                st.markdown(f"*Texto de respaldo:* {texto_fuente}")
    else:
        st.warning("Ingrese una consulta válida.")