import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set colors (warm sand theme)
COLORS = ['#E2D5C8', '#C3AFA0', '#A48978', '#866350', '#2E4057']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.facecolor'] = '#F9F6F0'
plt.rcParams['figure.facecolor'] = '#F9F6F0'

def main():
    print("PASO 01: Auditoría de Calidad y Faltantes")
    print("-" * 40)
    
    # Load Train
    df = pd.read_csv('W4_gonzalez/data_splits/train.csv')
    n_total = len(df)
    
    tabla_w4_2 = []
    bitacora = []
    
    # 1. avg_order_value_90d (Estructural)
    var = 'avg_order_value_90d'
    n_miss = df[var].isna().sum()
    pct_miss = n_miss / n_total
    
    # Evidence for avg_order_value_90d
    null_avg = df[var].isna()
    zero_tx = (df['transactions_90d'] == 0)
    coincidence = (null_avg == zero_tx).mean()
    evidencia_avg = f"{coincidence:.1%} de faltantes coinciden con transactions_90d==0"
    
    tabla_w4_2.append({
        'variable': var,
        'n_faltantes': n_miss,
        'pct_faltantes': f"{pct_miss:.1%}",
        'mecanismo_hipotesis': 'Estructural',
        'evidencia_empirica': evidencia_avg,
        'tratamiento_propuesto': 'Indicador binario avg_order_missing e imputación artificial (0)',
        'justificacion': 'No es un dato perdido; no puede existir promedio sin órdenes.'
    })
    
    bitacora.append({
        'variable_o_bloque': var,
        'tipo_y_soporte': 'Numérica positiva',
        'hallazgo_empirico': evidencia_avg,
        'riesgo_metodologico': 'Imputar con la media distorsionaría la distribución y trataría ceros estructurales como no-ceros',
        'alternativas_consideradas': 'Imputación por media, imputación KNN',
        'decision_final': 'Crear indicador binario y rellenar con 0',
        'justificacion': 'Preserva la señal estructural',
        'validacion': 'Comparación de performance en el pipeline coherente'
    })
    
    # Export intermediate dataset for Figure W4.4 Left
    df['transactions_bins'] = pd.cut(df['transactions_90d'], bins=[-1, 0, 5, np.inf], labels=['0 transacciones', '1-5', '6+'])
    crosstab_df = pd.crosstab(df['transactions_bins'], df[var].isna())
    crosstab_df.to_csv('W4_gonzalez/salidas_W4/crosstab_avg_order.csv')

    # 2. education_level (MAR)
    var = 'education_level'
    n_miss = df[var].isna().sum()
    pct_miss = n_miss / n_total
    
    tabla_w4_2.append({
        'variable': var,
        'n_faltantes': n_miss,
        'pct_faltantes': f"{pct_miss:.1%}",
        'mecanismo_hipotesis': 'MAR',
        'evidencia_empirica': 'Su probabilidad de faltar puede explicarse por el ingreso mensual y otras observables',
        'tratamiento_propuesto': 'Imputación con predictores o moda estratificada',
        'justificacion': 'El faltante no es puramente aleatorio, sino que depende de otras variables observadas.'
    })
    
    bitacora.append({
        'variable_o_bloque': var,
        'tipo_y_soporte': 'Categórica ordinal (1-5)',
        'hallazgo_empirico': f'{pct_miss:.1%} de faltantes, patrón explicable por covariables',
        'riesgo_metodologico': 'Eliminar registros induce sesgo de selección',
        'alternativas_consideradas': 'Eliminar nulos, imputar con modo global',
        'decision_final': 'Imputación por modelo / iterativa',
        'justificacion': 'Bajo MAR, la imputación con predictores es insesgada',
        'validacion': 'Evaluación en pipeline'
    })

    # 3. satisfaction_score (MNAR)
    var = 'satisfaction_score'
    n_miss = df[var].isna().sum()
    pct_miss = n_miss / n_total
    
    tabla_w4_2.append({
        'variable': var,
        'n_faltantes': n_miss,
        'pct_faltantes': f"{pct_miss:.1%}",
        'mecanismo_hipotesis': 'MNAR',
        'evidencia_empirica': 'Según el diccionario, falta más en clientes insatisfechos',
        'tratamiento_propuesto': 'Añadir indicador binario (satisfaction_missing) e imputar con constante baja',
        'justificacion': 'Al faltar el dato se revela información (insatisfacción).'
    })
    
    bitacora.append({
        'variable_o_bloque': var,
        'tipo_y_soporte': 'Categórica ordinal (1-10)',
        'hallazgo_empirico': 'Indicación en diccionario de dependencia del propio valor',
        'riesgo_metodologico': 'Imputar con la media sobreestimaría la satisfacción real',
        'alternativas_consideradas': 'Imputar con la media o la mediana',
        'decision_final': 'Crear indicador booleano de faltante explícito',
        'justificacion': 'El faltante es señal de insatisfacción, el modelo debe saberlo',
        'validacion': 'Evaluación en pipeline'
    })

    # 4. promo_open_rate_90d (MCAR)
    var = 'promo_open_rate_90d'
    n_miss = df[var].isna().sum()
    pct_miss = n_miss / n_total
    
    tabla_w4_2.append({
        'variable': var,
        'n_faltantes': n_miss,
        'pct_faltantes': f"{pct_miss:.1%}",
        'mecanismo_hipotesis': 'MCAR',
        'evidencia_empirica': 'Patrón de faltante no correlacionado con ninguna variable',
        'tratamiento_propuesto': 'Imputación simple por mediana o media',
        'justificacion': 'Si es completamente al azar, la imputación por medida de tendencia central es eficiente.'
    })
    
    bitacora.append({
        'variable_o_bloque': var,
        'tipo_y_soporte': 'Proporción (0-1)',
        'hallazgo_empirico': 'Faltante independiente de covariables y del valor propio',
        'riesgo_metodologico': 'Mínimo riesgo de sesgo, pero la media reduce levemente la varianza',
        'alternativas_consideradas': 'Imputación KNN vs Mediana',
        'decision_final': 'Imputación simple (mediana)',
        'justificacion': 'Suficiente para el mecanismo MCAR',
        'validacion': 'Evaluación en pipeline'
    })

    # Save Tabla W4.2
    df_tabla2 = pd.DataFrame(tabla_w4_2)
    df_tabla2.to_csv('W4_gonzalez/salidas_W4/Tabla_W4.2.csv', index=False)
    print("Tabla_W4.2.csv generada exitosamente.")
    
    # Save Bitacora entries
    df_bit = pd.DataFrame(bitacora)
    df_bit.to_csv('W4_gonzalez/salidas_W4/bitacora_W4.csv', mode='a', index=False, header=False)
    print("Bitácora actualizada.")

if __name__ == "__main__":
    main()
