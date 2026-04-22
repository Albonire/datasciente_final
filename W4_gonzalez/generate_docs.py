import os
import re

def main():
    print("Generando documentos Kami...")

    # 1. INFORME W4 (long-doc.html)
    with open('W4_gonzalez/informe_W4.html', 'r', encoding='utf-8') as f:
        informe = f.read()

    # Extract head and style
    head_match = re.search(r'(<head>.*?</head>)', informe, flags=re.DOTALL)
    if head_match:
        head_content = head_match.group(1)
        # Fix title
        head_content = head_content.replace('{{文档标题}}', 'Trabajo 4 - Tipo de dato, calidad y representación')
        # Fix font path
        head_content = head_content.replace('../fonts/', 'assets/fonts/')
    else:
        head_content = "<head></head>"

    body_content = """
<body>

<!-- COVER -->
<section class="cover">
  <div>
    <div class="cover-eyebrow">Ciencia de Datos - Corte 1</div>
    <div class="cover-title">Trabajo 4<br>Tipo de Dato, Calidad y Representación</div>
    <div class="cover-sub">Análisis empírico de operaciones válidas y su impacto en modelos predictivos</div>
  </div>
  <div class="cover-meta">
    <strong>Estudiante: Anderson González</strong><br>
    Profesor: Omar Portilla<br>
    Universidad de Pamplona · Periodo: 2026-1<br>
  </div>
</section>

<!-- TOC -->
<section class="toc">
  <h2>Índice</h2>
  <div class="toc-item"><span class="toc-num">01</span><span class="toc-title">Resumen Ejecutivo</span><span class="toc-page">03</span></div>
  <div class="toc-item"><span class="toc-num">02</span><span class="toc-title">Pregunta Rectora</span><span class="toc-page">04</span></div>
  <div class="toc-item"><span class="toc-num">03</span><span class="toc-title">Variables Obligatorias</span><span class="toc-page">05</span></div>
  <div class="toc-item"><span class="toc-num">04</span><span class="toc-title">Protocolo Experimental</span><span class="toc-page">06</span></div>
  <div class="toc-item"><span class="toc-num">05</span><span class="toc-title">Resultados Obligatorios</span><span class="toc-page">07</span></div>
  <div class="toc-item"><span class="toc-num">06</span><span class="toc-title">Razonamientos Obligatorios</span><span class="toc-page">11</span></div>
  <div class="toc-item"><span class="toc-num">07</span><span class="toc-title">Decisión Final</span><span class="toc-page">13</span></div>
  <div class="toc-item"><span class="toc-num">08</span><span class="toc-title">Errores Evitados</span><span class="toc-page">14</span></div>
  <div class="toc-item"><span class="toc-num">09</span><span class="toc-title">Conclusión Didáctica</span><span class="toc-page">15</span></div>
  <div class="toc-item"><span class="toc-num">10</span><span class="toc-title">Referencias</span><span class="toc-page">16</span></div>
</section>

<!-- 1. RESUMEN EJECUTIVO -->
<section class="chapter">
  <div class="chapter-num">01 · Summary</div>
  <h1>1. Resumen Ejecutivo</h1>
  <p class="lead">Este trabajo demuestra empíricamente que tratar todos los números como escalares continuos y todos los valores nulos como ignorancia aleatoria destruye información predictiva valiosa.</p>
  <p>A través del análisis del dataset <code>eda_laboratorio_clientes_v1.csv</code>, aplicamos un protocolo riguroso de deduplicación y partición para comparar un <strong>Pipeline Ingenuo</strong> contra un <strong>Pipeline Coherente</strong>. Al respetar la semántica de los datos (composiciones con CLR, cíclicas con seno/coseno) y modelar los mecanismos de faltantes (estructurales y explícitos), logramos un mejor rendimiento predictivo, elevando el ROC-AUC en validación al aislar las decisiones de representación.</p>
</section>

<!-- 2. PREGUNTA RECTORA -->
<section>
  <div class="chapter-num">02 · Core Question</div>
  <h1>2. Pregunta Rectora</h1>
  <div class="callout">
    ¿Qué operaciones son válidas o inválidas según el tipo de dato de cada variable, y cómo cambia la calidad del modelo cuando se representa el dato de forma semánticamente coherente en lugar de hacerlo con una codificación ingenua?
  </div>
  <p>Esta pregunta nos obliga a abandonar la comodidad computacional de "imputar todo con la media" o "pasar todo a entero", para adoptar un enfoque de diseño donde la estructura matemática de la variable dicta su representación.</p>
</section>

<!-- 3. VARIABLES -->
<section class="chapter">
  <div class="chapter-num">03 · Features</div>
  <h1>3. Variables Obligatorias</h1>
  <p>El estudio se centra en dos bloques críticos:</p>
  <ul>
    <li><strong>Representación:</strong> <code>region</code>, <code>plan_type</code> (Nominales); <code>education_level</code>, <code>loyalty_tier</code>, <code>satisfaction_score</code> (Ordinales); <code>promo_open_rate_90d</code> (Proporción); <code>share_*</code> (Composiciones); <code>peak_hour_local</code> (Cíclica).</li>
    <li><strong>Calidad (Faltantes):</strong> <code>avg_order_value_90d</code> (Estructural); <code>education_level</code> (MAR); <code>satisfaction_score</code> (MNAR); <code>promo_open_rate_90d</code> (MCAR).</li>
  </ul>
</section>

<!-- 4. PROTOCOLO -->
<section>
  <div class="chapter-num">04 · Protocol</div>
  <h1>4. Protocolo Experimental</h1>
  <ol>
    <li><strong>Deduplicación:</strong> Retención del registro más reciente por <code>customer_id</code>. Se removieron 90 filas.</li>
    <li><strong>Target:</strong> Creación de <code>spend_positive = 1 if future_spend_60d > 0 else 0</code>.</li>
    <li><strong>Partición:</strong> 60% Train, 20% Val, 20% Test, estratificado (random_state=2026).</li>
    <li><strong>Aislamiento:</strong> Transformadores entrenados exclusivamente en Train. Test preservado para el reporte final único.</li>
    <li><strong>Fuga de Información:</strong> Exclusión explícita de `post_window_retention_call` y `post_window_discount_amount`.</li>
  </ol>
</section>

<!-- 5. RESULTADOS -->
<section class="chapter">
  <div class="chapter-num">05 · Results</div>
  <h1>5. Resultados Obligatorios</h1>
  <p>La evaluación empírica confirma nuestras hipótesis semánticas.</p>
  
  <figure>
    <img src="salidas_W4/Figura_W4.1_frecuencias_nominales.png" alt="Frecuencias nominales">
    <figcaption>Figura W4.1: Distribución nominal y detección de rareza categórica.</figcaption>
  </figure>

  <figure>
    <img src="salidas_W4/Figura_W4.2_composicion_CLR.png" alt="Composición CLR">
    <figcaption>Figura W4.2: Matriz de correlación espuria (izquierda) vs correlación corregida mediante CLR (derecha).</figcaption>
  </figure>

  <figure>
    <img src="salidas_W4/Figura_W4.3_ciclica_peak_hour.png" alt="Cíclica peak hour">
    <figcaption>Figura W4.3: Codificación de variable cíclica eliminando el salto artificial de 23 a 0.</figcaption>
  </figure>

  <figure>
    <img src="salidas_W4/Figura_W4.4_faltante_avg_order.png" alt="Faltantes Estructurales y Desempeño">
    <figcaption>Figura W4.4: Auditoría de faltantes estructurales y mejora de ROC-AUC del pipeline coherente.</figcaption>
  </figure>
</section>

<!-- 6. RAZONAMIENTOS -->
<section class="chapter">
  <div class="chapter-num">06 · Reasoning</div>
  <h1>6. Razonamientos Obligatorios</h1>
  <div class="takeaway">
    <p><strong>1. ¿Por qué region y plan_type no pueden tratarse como variables ordinales arbitrarias?</strong></p>
    <p>Imponer un Label Encoding (ej. Norte=1, Sur=2) fuerza artificialmente al modelo a creer que "Sur es el doble que Norte", induciendo un sesgo geométrico sin respaldo en el dominio. Una regresión logística trazará una relación monótona falsa entre ellas.</p>
  </div>
  
  <div class="takeaway">
    <p><strong>2. ¿Cuándo conservar una ordinal como entero ordenado?</strong></p>
    <p>Si la distancia conceptual entre los niveles es razonablemente equidistante (ej. Nivel 1 a 2 es similar a 2 a 3), conservarlo como entero preserva la jerarquía paramétrica y ahorra grados de libertad. Si hay no-linealidades graves, es mejor relajar la asunción con splines discretos o one-hot.</p>
  </div>
  
  <div class="takeaway">
    <p><strong>3. Restricción Simplex en shares</strong></p>
    <p>Los canales forman una composición de suma=1. Si el uso Web sube, matemáticamente los otros deben bajar, creando correlaciones negativas espurias (Fig W4.2). Operar directamente con correlación de Pearson aquí viola la independencia. La transformación Centered Log Ratio (CLR) mapea estos datos al espacio real sin restricciones.</p>
  </div>
  
  <div class="takeaway">
    <p><strong>4. Codificación cíclica temporal</strong></p>
    <p>En el dominio entero [0,23], la distancia euclidiana entre 23 y 0 es 23. En el tiempo real es 1 hora. Codificando en proyecciones Seno y Coseno, mapeamos el tiempo a un círculo bidimensional, garantizando distancias temporales coherentes (Fig W4.3).</p>
  </div>
  
  <div class="takeaway">
    <p><strong>5. Faltante estructural de avg_order_value</strong></p>
    <p>La coincidencia exacta entre `is_null(avg_order)` y `transactions_90d == 0` demuestra que este "nulo" es lógico, no estadístico. Imputarlo con la media poblacional inventaría ingresos para clientes inactivos. Rellenar con 0 y agregar un indicador es la forma defensiva y lógicamente sensata de modelarlo.</p>
  </div>
</section>

<!-- 7. DECISIÓN -->
<section class="chapter">
  <div class="chapter-num">07 · Final Decision</div>
  <h1>7. Decisión Final</h1>
  <p>El <strong>Pipeline Coherente</strong> es nuestra decisión final unánime. Al implementar una gestión consciente de faltantes (imputación según MCAR/MAR/MNAR y estructural) y representaciones fidedignas (CLR, Cyclic, OHE), el modelo mejoró su capacidad de generalización superando al Pipeline Ingenuo en la validación, protegiendo al negocio de inferencias espurias.</p>
</section>

<!-- 8. ERRORES EVITADOS -->
<section>
  <div class="chapter-num">08 · Errors Avoided</div>
  <h1>8. Errores Evitados</h1>
  <ul>
    <li><span class="hl">Data Leakage:</span> Excluir `post_window_discount_amount` evitó que el modelo predeciera mágicamente usando el futuro.</li>
    <li><span class="hl">Correlación Espuria Composicional:</span> Se evitaron coeficientes lineales engañosos en la distribución multicanal usando CLR.</li>
    <li><span class="hl">Faltante Estructural como Aleatorio:</span> Evitamos inflar promedios artificialmente al reconocer los ceros estructurales de inactividad.</li>
  </ul>
</section>

<!-- 9. CONCLUSIÓN -->
<section>
  <div class="chapter-num">09 · Didactic Conclusion</div>
  <h1>9. Conclusión Didáctica</h1>
  <p>Para todo científico de datos: Un <code>NaN</code> nunca es solo un `NaN`, y un número no siempre pertenece a $\mathbb{R}$. Interrogar la ontología del dato antes de aplicar transformadores <code>scikit-learn</code> es lo que separa a un operador de código de un modelador estadístico riguroso.</p>
</section>

<!-- 10. REFERENCIAS -->
<section class="chapter">
  <div class="chapter-num">10 · References</div>
  <h1>10. Referencias</h1>
  <ul>
    <li>Documentación de Variables: <code>diccionario_datos_estudiante.csv</code></li>
    <li>Guía de Trabajos EDA: <code>guia_trabajos_eda.pdf</code></li>
    <li>Aitchison, J. (1982). The Statistical Analysis of Compositional Data.</li>
  </ul>
</section>

</body>
</html>
"""
    with open('W4_gonzalez/informe_W4.html', 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html lang=\"es\">\n" + head_content + body_content)


    # 2. HANDOUT W4 (one-pager.html)
    with open('W4_gonzalez/handout_W4_1pagina.html', 'r', encoding='utf-8') as f:
        handout = f.read()

    h_head_match = re.search(r'(<head>.*?</head>)', handout, flags=re.DOTALL)
    if h_head_match:
        h_head = h_head_match.group(1).replace('../fonts/', 'assets/fonts/')
    else:
        h_head = "<head></head>"

    h_body = """
<body>
<div class="container">
  <div class="header">
    <div class="meta">Guía de Referencia Rápida · Ciencia de Datos</div>
    <div class="title">No todo número es una variable continua</div>
  </div>

  <div class="grid">
    <div class="col">
      <h3>1. Nominales (Finitas sin orden)</h3>
      <div class="metric">
        <div class="metric-label">Operaciones Válidas</div>
        <div class="metric-val">Igualdad (==), Moda</div>
      </div>
      <p class="text"><strong>El Riesgo:</strong> Representarlos como enteros (1, 2, 3...) fuerza falsas nociones de distancia matemática.<br><strong>Solución:</strong> One-Hot Encoding.</p>
    </div>

    <div class="col">
      <h3>2. Ordinales (Finitas con orden)</h3>
      <div class="metric">
        <div class="metric-label">Operaciones Válidas</div>
        <div class="metric-val">Desigualdad (>), Mediana</div>
      </div>
      <p class="text"><strong>El Riesgo:</strong> Asumir distancias equidistantes entre rangos (ej. 1 a 2 = 4 a 5).<br><strong>Solución:</strong> Preservar como enteros si la escala subyacente es lineal, o usar codificaciones ordinales monótonas.</p>
    </div>
  </div>
  
  <div class="grid">
    <div class="col">
      <h3>3. Composicionales (Suma = 1)</h3>
      <div class="metric">
        <div class="metric-label">Operaciones Válidas</div>
        <div class="metric-val">Log-Ratios (CLR/ALR)</div>
      </div>
      <p class="text"><strong>El Riesgo:</strong> Correlaciones negativas espurias forzadas por la restricción de suma (un canal sube, el otro debe bajar).<br><strong>Solución:</strong> Transformación Centered Log Ratio.</p>
    </div>

    <div class="col">
      <h3>4. Cíclicas (Reloj, Calendario)</h3>
      <div class="metric">
        <div class="metric-label">Operaciones Válidas</div>
        <div class="metric-val">Trigonometría (Seno/Cos)</div>
      </div>
      <p class="text"><strong>El Riesgo:</strong> El salto artificial entre el fin y el inicio del ciclo (hora 23 y hora 0) crea distancias matemáticas máximas cuando en realidad son adyacentes.<br><strong>Solución:</strong> Codificar en 2 dimensiones usando sen/cos.</p>
    </div>
  </div>
  
  <div style="margin-top:20pt;">
    <h3>Gestión Semántica de Faltantes</h3>
    <ul>
      <li><strong>Estructural:</strong> No falta, "no aplica". Ej: Valor de orden sin compras. Se debe crear indicador binario e imputar con constante cero.</li>
      <li><strong>MNAR:</strong> Falta porque el usuario no quiere revelarlo (ej. muy insatisfecho). Agregar indicador binario es crucial.</li>
      <li><strong>MAR/MCAR:</strong> Falta aleatoriamente o explicable por otras variables. Imputar con modelos (iterative) o tendencia central.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""
    with open('W4_gonzalez/handout_W4_1pagina.html', 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html lang=\"es\">\n" + h_head + h_body)

    # 3. SOCIALIZACION W4 (slides.py)
    slides_code = """
import sys

def render_slides():
    html = '''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Presentación W4</title>
  <style>
    @font-face {
      font-family: "TsangerJinKai02";
      src: url("assets/fonts/TsangerJinKai02-W04.ttf") format("truetype");
    }
    @page { size: 1920px 1080px; margin: 0; }
    body { font-family: "TsangerJinKai02", serif; background: #f5f4ed; color: #141413; margin: 0; padding: 0; }
    .slide { width: 1920px; height: 1080px; padding: 80px 100px; box-sizing: border-box; page-break-after: always; position: relative; }
    h1 { font-size: 80px; color: #1B365D; border-bottom: 4px solid #1B365D; padding-bottom: 20px; }
    h2 { font-size: 50px; color: #3d3d3a; margin-top: 40px; }
    p, li { font-size: 40px; line-height: 1.6; color: #4d4c48; }
    .meta { position: absolute; bottom: 50px; right: 100px; font-size: 30px; color: #87867f; }
    .split { display: flex; gap: 60px; margin-top: 50px; }
    .col { flex: 1; }
    img { max-width: 100%; border-radius: 10px; }
  </style>
</head>
<body>

  <!-- Slide 1 -->
  <div class="slide">
    <h1>Tipo de Dato, Calidad y Representación</h1>
    <h2>Sección 1 (4 min): Teoría y Ontología del Dato</h2>
    <ul>
      <li>No todo número es un escalar continuo $\\mathbb{R}$.</li>
      <li>Restricciones estructurales: Cíclicos (reloj), Composicionales (suma=1).</li>
      <li>Mecanismos de faltantes: MCAR, MAR, MNAR y Estructurales (no aplica).</li>
    </ul>
    <div class="meta">Trabajo 4 · Anderson González</div>
  </div>

  <!-- Slide 2 -->
  <div class="slide">
    <h1>Evidencia Empírica de Representaciones</h1>
    <h2>Sección 2 (7 min): El impacto de la geometría</h2>
    <div class="split">
      <div class="col">
        <p><strong>Composiciones CLR:</strong> Corregimos correlaciones espurias negativas forzadas por la restricción multicanal.</p>
        <p><strong>Codificación Cíclica:</strong> Corregimos el error euclidiano del salto 23h $\\rightarrow$ 0h mapeando al círculo trigonométrico.</p>
      </div>
      <div class="col">
        <img src="salidas_W4/Figura_W4.3_ciclica_peak_hour.png" alt="Ciclica">
      </div>
    </div>
    <div class="meta">Sección 2: Evidencia Empírica</div>
  </div>

  <!-- Slide 3 -->
  <div class="slide">
    <h1>Validación de Pipelines</h1>
    <h2>Sección 3 (2 min): Ingenuo vs Coherente</h2>
    <ul>
      <li><strong>Ingenuo:</strong> Label encoder arbitrario, numéricos crudos, media para todo.</li>
      <li><strong>Coherente:</strong> OHE, Sen/Cos, CLR, Imputación Iterativa e indicadores.</li>
      <li><strong>Resultado:</strong> El pipeline coherente superó en ROC-AUC aislando los efectos semánticos de representación.</li>
    </ul>
    <div class="meta">Sección 3: Pipelines</div>
  </div>

  <!-- Slide 4 -->
  <div class="slide">
    <h1>Tres Errores a Evitar</h1>
    <h2>Sección 4 (2 min): Lecciones para el curso</h2>
    <ol>
      <li><strong>Imputar estructurales con la media:</strong> Destruye señales lógicas (no puedes promediar órdenes sin compras).</li>
      <li><strong>Data Leakage del futuro:</strong> Incluir <code>post_window_discount</code> predice el futuro con el futuro.</li>
      <li><strong>Operar sobre distribuciones Simplex brutas:</strong> Falsa colinealidad en canales compartidos.</li>
    </ol>
    <div class="meta">Sección 4: Lecciones</div>
  </div>

</body>
</html>'''
    with open("socializacion_W4.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Socialización HTML generada.")

if __name__ == '__main__':
    render_slides()
"""
    with open('W4_gonzalez/socializacion_W4.py', 'w', encoding='utf-8') as f:
        f.write(slides_code)

    print("Scripts generadores de documentos listos.")

if __name__ == "__main__":
    main()
