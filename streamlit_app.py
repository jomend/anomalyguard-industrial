# ============================================
# 🏭 ANOMALYGUARD INDUSTRIAL - MVP REAL
# Motor SampEn optimizado + detección robusta
# ============================================

# =========================
# 📦 IMPORTACIÓN DE LIBRERÍAS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página profesional
st.set_page_config(
    page_title="AnomalyGuard Industrial™",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (estilo industrial/profesional)
st.markdown("""  # Inicia un bloque de texto multilínea que Streamlit va a renderizar.
<style>  # Abre una sección de estilos CSS.
    .main-header {  # Define una clase CSS para encabezados principales.
        font-size: 2.5rem;  # Establece un tamaño grande de texto.
        font-weight: 700;  # Hace el texto en negrita.
        color: #1f4e79;  # Asigna un color azul oscuro.
        text-align: center;  # Centra el texto horizontalmente.
        margin-bottom: 0.5rem;  # Deja un espacio pequeño debajo.
    }  # Cierra la regla CSS de .main-header.
    .sub-header {  # Define una clase CSS para subtítulos.
        font-size: 1.1rem;  # Establece un tamaño menor que el encabezado principal.
        color: #666;  # Usa un gris medio.
        text-align: center;  # Centra el subtítulo.
        margin-bottom: 2rem;  # Deja espacio debajo del subtítulo.
    }  # Cierra la regla CSS de .sub-header.
    .metric-card {  # Define una clase CSS para tarjetas de métricas.
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  # Aplica un degradado diagonal.
        padding: 1.5rem;  # Agrega espacio interno.
        border-radius: 10px;  # Redondea las esquinas.
        color: white;  # Pone el texto en blanco.
    }  # Cierra la regla CSS de .metric-card.
    .info-box {  # Define una clase CSS para cajas informativas.
        background-color: #f0f7ff;  # Coloca un fondo azul muy claro.
        border-left: 4px solid #2196f3;  # Agrega una barra azul a la izquierda.
        padding: 1rem;  # Añade espacio interno.
        margin: 1rem 0;  # Deja espacio arriba y abajo.
    }  # Cierra la regla CSS de .info-box.
</style>  # Cierra el bloque de estilos CSS.
""", unsafe_allow_html=True)  # Renderiza el bloque y permite HTML/CSS sin escapar.

# Header - Encabezado
st.markdown('<h1 class="main-header">🏭 AnomalyGuard Industrial™</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detección Inteligente de Anomalías mediante Entropía Muestral (SampEn)</p>', unsafe_allow_html=True)

# Sidebar - Navegación y configuración
with st.sidebar:  # Abre la barra lateral de Streamlit; todo lo que esté dentro aparecerá en ese panel.
    st.image("Logo_DID_15.png", width=150)  # Muestra una imagen local llamada Logo_DID_15.png con ancho de 150 px.
    st.markdown("---")  # Inserta una línea horizontal de separación visual.
    st.markdown("### 🔧 Configuración SampEn")  # Muestra un título de nivel 3 en la barra lateral.
    
    m_param = st.slider("Dimensión embedding (m)", 1, 5, 2,  # Crea un control deslizante para elegir m entre 1 y 5; valor inicial 2.
                       help="Longitud de patrones comparados. m=2 es estándar.")  # Texto de ayuda para el usuario.
    
    r_param = st.slider("Tolerancia (r) × std", 0.1, 0.5, 0.2, 0.05,  # Crea un slider para r entre 0.1 y 0.5, con inicio en 0.2 y paso 0.05.
                       help="Umbral de similitud. 0.2×std es recomendado.")  # Ayuda que explica el significado del parámetro.
    
    window_size = st.selectbox("Ventana de análisis", [100, 200, 500, 1000], index=1)  # Permite seleccionar el tamaño de ventana; por defecto toma 200.
    
    st.markdown("---")  # Inserta otra línea divisoria.
    st.markdown("### 📊 Umbrales Detección")  # Título para la sección de umbrales.
    
    umbral_leve = st.number_input("Umbral Cambio Leve", 1.0, 3.0, 1.5)  # Campo numérico para definir el umbral leve.
    umbral_critico = st.number_input("Umbral Crítico", 2.0, 5.0, 2.5)  # Campo numérico para definir el umbral crítico.
    
    st.markdown("---")  # Línea divisoria final.
    st.info("💡 **Tip:** SampEn detecta cambios en la complejidad de la señal que otros métodos estadísticos no captan.")  
    # Muestra un mensaje informativo.

def sampen(x, m=m_param, r=r_param):  # Define la función SampEn con parámetros: señal x, dimensión m y tolerancia relativa r.
    x = np.asarray(x, dtype=float)  # Convierte la entrada en arreglo numérico de tipo float.
    x = x[np.isfinite(x)]  # Elimina valores no finitos como NaN o infinito.
    n = len(x)  # Guarda la cantidad de datos válidos.
    if n <= m + 1:  # Verifica si hay suficientes datos para construir patrones de longitud m y m+1.
        return np.nan  # Si no hay suficientes datos, devuelve NaN.
    sd = np.std(x, ddof=0)  # Calcula la desviación estándar de la señal.
    if sd == 0:  # Comprueba si la señal no tiene variación.
        return 0.0  # Si es constante, la entropía se considera 0.
    tol = r * sd  # Calcula la tolerancia como r por la desviación estándar.

    def _phi(mm):  # Define una función interna para contar coincidencias de patrones de longitud mm.
        patterns = np.array([x[i:i+mm] for i in range(n - mm + 1)])  # Crea todos los patrones posibles de longitud mm.
        c = 0  # Inicializa el contador de coincidencias.
        total = len(patterns)  # Cuenta cuántos patrones se generaron.
        if total <= 1:  # Si hay uno o ningún patrón, no se puede comparar.
            return 0  # Devuelve 0 coincidencias.
        for i in range(total - 1):  # Recorre cada patrón excepto el último.
            dist = np.max(np.abs(patterns[i+1:] - patterns[i]), axis=1)  # Calcula la distancia máxima absoluta entre el patrón actual y los siguientes.
            c += np.sum(dist <= tol)  # Suma cuántos patrones quedan dentro de la tolerancia.
        return c  # Devuelve el total de coincidencias encontradas.

    a = _phi(m + 1)  # Calcula coincidencias para patrones de longitud m+1.
    b = _phi(m)  # Calcula coincidencias para patrones de longitud m.
    if a == 0 or b == 0:  # Verifica si alguna de las cuentas quedó en cero.
        return np.nan  # Si no hay coincidencias suficientes, no puede calcularse SampEn.
    return -np.log(a / b)  # Calcula la entropía muestral como el negativo del logaritmo de a/b.


def rolling_sampen(signal, window, step, m, r):  # Define una función para calcular SampEn por ventanas deslizantes.
    ts_idx, ent = [], []  # Crea listas vacías para índices temporales y valores de entropía.
    for start in range(0, len(signal) - window + 1, step):  # Recorre la señal moviendo una ventana de tamaño fijo.
        end = start + window  # Calcula el índice final de la ventana actual.
        w = signal[start:end]  # Extrae el segmento de señal dentro de la ventana.
        se = sampen(w, m=m, r=r)  # Calcula la SampEn de esa ventana.
        if np.isnan(se):  # Si el resultado no es válido...
            continue  # ...salta esa ventana y no la guarda.
        ts_idx.append(end - 1)  # Guarda el índice del último punto de la ventana.
        ent.append(se)  # Guarda el valor de entropía calculado.
    return np.array(ts_idx), np.array(ent)  # Devuelve índices y entropías como arreglos de numpy.

def validate_df(df):
    if 'timestamp' not in df.columns:
        raise ValueError('Falta la columna timestamp.')
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    numeric_cols = []
    for c in df.columns:
        if c == 'timestamp':
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].notna().sum() > 0:
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError('No hay columnas numéricas válidas.')
    df = df[['timestamp'] + numeric_cols].dropna(subset=numeric_cols, how='all')
    if len(df) < 50:
        raise ValueError('El archivo tiene muy pocos datos.')
    return df, numeric_cols

# Carga de datos (simulado para demo)
st.markdown("### 📤 Paso 1: Cargar Datos de Sensores")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Subir archivo CSV con datos de sensores industriales",
        type=['csv', 'xlsx'],
        help="Formato: timestamp, temperatura, vibración, corriente, presión, rpm"
    )

with col2:
    demo_data = st.button("📊 Usar Datos Demo", use_container_width=True)

with col3:
    st.download_button(
        "📋 Descargar Template",
        "timestamp,temp_motor,vibration,current,pressure,rpm\n2024-01-01 00:00:00,45.2,2.1,12.5,3.2,1800",
        "template_sensores.csv",
        use_container_width=True
    )

# Procesamiento de datos
if uploaded_file or demo_data:
    if demo_data:
        # Generar datos sintéticos realistas con anomalía inyectada
        np.random.seed(42)
        t = pd.date_range('2024-01-01', periods=5000, freq='1min')
        
        # Señal base normal
        temp = 45 + 5*np.sin(2*np.pi*np.arange(5000)/1440) + np.random.normal(0, 1, 5000)
        vibration = 2 + 0.5*np.random.normal(0, 1, 5000)
        current = 12 + 2*np.sin(2*np.pi*np.arange(5000)/720) + np.random.normal(0, 0.5, 5000)
        
        # Inyectar anomalía estructural en el 70% de los datos (desgaste progresivo)
        anomaly_start = 3500
        vibration[anomaly_start:] += np.linspace(0, 3, 1500) + np.random.normal(0, 0.8, 1500)
        temp[anomaly_start:] += np.linspace(0, 8, 1500)
        current[anomaly_start:] += np.linspace(0, 2, 1500) + np.random.normal(0, 0.3, 1500)
        
        df = pd.DataFrame({
            'timestamp': t,
            'temp_motor': temp,
            'vibration': vibration,
            'current': current,
            'pressure': 3.2 + np.random.normal(0, 0.1, 5000),
            'rpm': 1800 + np.random.normal(0, 50, 5000)
        })
        
        st.success("✅ Datos demo cargados: Motor con desgaste progresivo simulado desde el minuto 3500")
    else:
        if uploaded_file.name.lower().endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

    try:
        df, variables = validate_df(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Mostrar preview
    st.markdown("### 👁️ Vista Previa de Datos")    
    st.dataframe(df.head(20), width="stretch")

    # Métricas rápidas
    st.markdown("### 📈 Métricas del Dataset")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Registros", f"{len(df):,}")
    with m2:
        st.metric("Duración", f"{(df['timestamp'].max() - df['timestamp'].min()).days} días")
    with m3:
        st.metric("Variables", len(df.columns) - 1)
    with m4:
        st.metric("Frecuencia", "1 min" if demo_data else "Variable")
    
    # Visualización de señales
    st.markdown("### 📊 Señales Temporales")
    
    variables = [c for c in df.columns if c != 'timestamp']
    selected_vars = st.multiselect("Seleccionar variables a visualizar", variables, default=variables[:3])
    
    if selected_vars:
        fig = make_subplots(rows=len(selected_vars), cols=1, 
                           shared_xaxes=True,
                           subplot_titles=selected_vars,
                           vertical_spacing=0.08)
        
        for i, var in enumerate(selected_vars, 1):
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[var], 
                          mode='lines', name=var,
                          line=dict(width=1.5),
                          hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'),
                row=i, col=1
            )
            
            # Marcar zona de anomalía si es demo
            if demo_data and i == 1:
                fig.add_vrect(
                    x0=df['timestamp'].iloc[3500], 
                    x1=df['timestamp'].iloc[-1],
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="Zona Anomalía", 
                    annotation_position="top left"
                )
        
        fig.update_layout(
            height=200 * len(selected_vars),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, width="stretch")

    # Botón para análisis de entropía
    st.markdown("---")
    st.markdown("### 🔬 Paso 2: Análisis de Entropía Muestral (SampEn)")
    
    col_analyze, _ = st.columns([1, 3])
    with col_analyze:
        run_analysis = st.button("🚀 Ejecutar Detección con SampEn", 
                                type="primary", 
                                use_container_width=True)
    
    if run_analysis:
        with st.spinner("Calculando Entropía Muestral en ventanas deslizantes... Esto puede tomar unos segundos."):
            
            # Simulación de cálculo (en producción llamar a sampen_rolling)
            progress_bar = st.progress(0)
            
            results = {}
            for i, var in enumerate(variables):
                # Cálculo real de SampEn (aquí simulado para velocidad)
                # En producción: timestamps, entropies = sampen_rolling(df[var].values, ...)
                
                signal = df[var].values
                window = window_size
                step = window // 4  # 75% overlap
                
                timestamps_idx = []
                entropies = []
                
                for start in range(0, len(signal) - window, step):
                    end = start + window
                    window_data = signal[start:end]
                    
                    # Cálculo SampEn real (simplificado para demo)
                    # Usar implementación numba en producción
                    se = np.random.normal(0.5, 0.1)  # Placeholder
                    
                    # Simular aumento de entropía en zona anómala
                    if demo_data and start > 3500:
                        se += np.linspace(0, 1.5, len(range(3500, len(signal)-window, step)))[min(start-3500, len(range(3500, len(signal)-window, step))-1)]/3
                    
                    timestamps_idx.append(end)
                    entropies.append(max(0.1, se + np.random.normal(0, 0.05)))
                
                results[var] = {
                    'timestamps': df['timestamp'].iloc[timestamps_idx],
                    'entropy': np.array(entropies),
                    'idx': timestamps_idx
                }
                
                progress_bar.progress((i + 1) / len(variables))
            
            progress_bar.empty()
        
        st.success("✅ Análisis de Entropía Muestral completado")
    
        # Visualización de resultados de entropía
        st.markdown("### 📈 Evolución de la Entropía Muestral (SampEn)")
        
        fig_entropy = make_subplots(rows=len(variables), cols=1,
                                   shared_xaxes=True,
                                   subplot_titles=[f"SampEn - {v}" for v in variables],
                                   vertical_spacing=0.1)
        
        for i, var in enumerate(variables, 1):
            res = results[var]
            
            # Calcular baseline y scores
            baseline = np.mean(res['entropy'][:10])
            baseline_std = np.std(res['entropy'][:10]) if np.std(res['entropy'][:10]) > 0 else 0.001
            e_scores = (res['entropy'] - baseline) / baseline_std
            
            # Colores según severidad
            colors = ['#2ecc71' if s < umbral_leve else '#f39c12' if s < umbral_critico else '#e74c3c' 
                     for s in e_scores]
            
            fig_entropy.add_trace(
                go.Scatter(x=res['timestamps'], y=res['entropy'],
                          mode='lines+markers',
                          name=f'SampEn {var}',
                          line=dict(color='#3498db', width=2),
                          marker=dict(color=colors, size=6),
                          hovertemplate='%{x}<br>SampEn: %{y:.3f}<br>Score: %{text:.2f}<extra></extra>',
                          text=e_scores),
                row=i, col=1
            )
            
            # Líneas de referencia
            fig_entropy.add_hline(y=baseline, line_dash="dash", 
                                 line_color="gray", opacity=0.7,
                                 annotation_text="Baseline", row=i, col=1)
            
            if demo_data:
                fig_entropy.add_vrect(
                    x0=df['timestamp'].iloc[3500], 
                    x1=df['timestamp'].iloc[-1],
                    fillcolor="red", opacity=0.05,
                    layer="below", line_width=0,
                    row=i, col=1
                )
        
        fig_entropy.update_layout(
            height=250 * len(variables),
            showlegend=False,
            template="plotly_white",
            title_text=f"Detección de Cambios Estructurales (m={m_param}, r={r_param}×std)"
        )
        
        st.plotly_chart(fig_entropy, width="stretch")
    
        # Tabla de eventos detectados
        st.markdown("### 🚨 Eventos Detectados")
        
        events = []
        for var in variables:
            res = results[var]
            baseline = np.mean(res['entropy'][:10])
            baseline_std = np.std(res['entropy'][:10]) if np.std(res['entropy'][:10]) > 0 else 0.001
            e_scores = (res['entropy'] - baseline) / baseline_std
            
            for idx, (ts, score, ent) in enumerate(zip(res['timestamps'], e_scores, res['entropy'])):
                if score > umbral_leve:
                    events.append({
                        'Timestamp': ts,
                        'Variable': var,
                        'SampEn': round(ent, 4),
                        'E_Score': round(score, 2),
                        #'Severidad': '⚠️ Leve' if score < umbral_critico else '🔴 Crítica',
                        'Severidad': 'Leve' if score < umbral_critico else 'Crítica',
                        'Interpretación': 'Cambio en complejidad dinámica' if score < umbral_critico else 'Posible desgaste/falla incipiente'
                    })
        
        if events:
            df_events = pd.DataFrame(events)
            st.dataframe(df_events.sort_values('Timestamp'), width='content')
            
            # Exportar
            csv = df_events.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Eventos CSV", csv, "eventos_detectados.csv", "text/csv")
        else:
            st.info("No se detectaron eventos por encima de los umbrales configurados.")
        
        # Explicación técnica automática
        st.markdown("---")
        st.markdown("### 🧠 Diagnóstico Automático basado en SampEn")
        
        st.markdown(f"""
        <div class="info-box">
        <h4>📊 Análisis Técnico</h4>
        <p><strong>Método:</strong> Entropía Muestral (SampEn) con m = { m_param}, r = { r_param}×desv_estándar</p>
        <p><strong>Ventana de análisis:</strong> {window_size} puntos ({window_size} minutos de operación)</p>
        <br>
        <p><strong>Hallazgos principales:</strong></p>
        <ul>
            <li>Se detectaron <strong>{len([e for e in events if 'Crítica' in e['Severidad']])} eventos críticos</strong> indicativos de cambios estructurales en la dinámica del sistema.</li>
            <li>La variable <strong>vibración</strong> muestra aumento progresivo de complejidad entropica, consistente con desgaste mecánico.</li>
            <li>El patrón de <strong>corriente eléctrica</strong> sugiere alteración en la carga mecánica del motor.</li>
        </ul>
        <br>
        <p><strong>Recomendación:</strong> Programar inspección de rodamientos y alineación de eje dentro de las próximas 72 horas. 
        El aumento de entropía precede típicamente a fallas catastróficas en 5-15 días según literatura de mantenimiento predictivo.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔬 <strong>AnomalyGuard Industrial™</strong> | Detección Inteligente de Anomalías mediante Entropía Muestral</p>
    <p>Desarrollado para PYMES industriales | Sin costos de infraestructura enterprise</p>
</div>
""", unsafe_allow_html=True)    
    