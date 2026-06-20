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
    # Corpus normativo indexado: 10 documentos de alta densidad semántica
    documentos = [
        # --- DERECHO LABORAL ---
        {
            "id": "LEY_1010_2006", 
            "tema": "Acoso Laboral", 
            "texto": "La Ley 1010 de 2006 define el acoso laboral como conducta persistente y demostrable ejercida sobre un empleado por un jefe o compañero para infundir miedo o terror. Las víctimas pueden denunciar ante el Comité de Convivencia y tienen fuero de protección contra el despido."
        },
        {
            "id": "CST_ART_64", 
            "tema": "Despido Sin Justa Causa", 
            "texto": "El Código Sustantivo del Trabajo establece que si un empleador termina un contrato laboral sin una justa causa comprobada, deberá pagar al trabajador una indemnización monetaria, la cual se calcula dependiendo del tipo de contrato (fijo o indefinido) y el tiempo laborado."
        },
        # --- DERECHO CIVIL ---
        {
            "id": "LEY_820_2003", 
            "tema": "Arrendamiento de Vivienda", 
            "texto": "La Ley 820 de 2003 dicta que el arrendador puede exigir la restitución del inmueble si el inquilino incumple con el pago del arriendo o servicios. Es ilegal que el arrendador suspenda los servicios públicos por su cuenta o retenga los bienes del arrendatario."
        },
        {
            "id": "CODIGO_CIVIL_PROMESA", 
            "tema": "Contratos y Promesas de Compraventa", 
            "texto": "Para que una promesa de compraventa de un inmueble sea válida legalmente, debe constar por escrito, contener los datos exactos del bien, el precio, y fijar una fecha y hora exacta para la firma de la escritura pública en una notaría específica."
        },
        # --- DERECHO DE FAMILIA ---
        {
            "id": "LEY_1098_2006", 
            "tema": "Cuota Alimentaria para Menores", 
            "texto": "El Código de Infancia y Adolescencia obliga a los padres a proveer alimentos a sus hijos menores. La cuota alimentaria cubre vivienda, vestido, salud, educación y recreación. Puede fijarse mediante conciliación voluntaria o por orden de un juez de familia."
        },
        {
            "id": "CODIGO_CIVIL_DIVORCIO", 
            "tema": "Divorcio y Separación", 
            "texto": "El divorcio puede solicitarse por mutuo acuerdo ante un notario (divorcio exprés) o por causas contenciosas ante un juez, como infidelidad, maltrato o separación de cuerpos por más de dos años. En el mutuo acuerdo, si hay menores, interviene el Defensor de Familia."
        },
        # --- DERECHO PENAL ---
        {
            "id": "CODIGO_PENAL_ART_239", 
            "tema": "Delito de Hurto", 
            "texto": "El hurto consiste en apoderarse de un bien mueble ajeno para obtener provecho. Incurrirá en prisión quien lo cometa. La condena se agrava (hurto calificado) si se utiliza violencia contra las personas, se rompen cerraduras o se usa armas."
        },
        {
            "id": "CODIGO_PENAL_ART_32", 
            "tema": "Legítima Defensa", 
            "texto": "No habrá responsabilidad penal por lesiones o muerte si la persona actúa en legítima defensa. Para que sea válida, debe existir una necesidad de defender un derecho propio o ajeno contra una agresión injusta, actual o inminente, y la defensa debe ser proporcional al ataque."
        },
        # --- DERECHO CONSTITUCIONAL ---
        {
            "id": "DECRETO_2591_1991", 
            "tema": "Acción de Tutela", 
            "texto": "La acción de tutela es un mecanismo para proteger derechos fundamentales (como la vida, la salud o el debido proceso) cuando son vulnerados por autoridades o particulares. El juez tiene un plazo máximo e improrrogable de 10 días hábiles para resolverla."
        },
        # --- DERECHO COMERCIAL / CONSUMIDOR ---
        {
            "id": "LEY_1480_2011", 
            "tema": "Derecho de Retracto (Consumidor)", 
            "texto": "El Estatuto del Consumidor otorga el derecho de retracto en compras por internet, catálogo o financiadas. El cliente tiene 5 días hábiles desde que recibe el producto para devolverlo sin justificación, y el vendedor debe reembolsar el 100% del dinero."
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
    vector_pregunta = retriever.encode([pregunta])
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
    if puntaje_similitud < 0.35:
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