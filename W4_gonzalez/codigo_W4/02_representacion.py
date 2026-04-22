import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

COLORS = ['#E2D5C8', '#C3AFA0', '#A48978', '#866350', '#2E4057']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.facecolor'] = '#F9F6F0'
plt.rcParams['figure.facecolor'] = '#F9F6F0'

def main():
    print("PASO 02: Representación y Transformación")
    print("-" * 40)
    
    df = pd.read_csv('W4_gonzalez/data_splits/train.csv')
    
    # 1. Tabla_W4.1.csv
    tabla_w4_1 = [
        {'variable': 'region', 'tipo_semantico': 'Nominal', 'soporte': 'Conjunto finito sin orden', 'riesgo_si_mal_tratado': 'Falsa noción de distancia y orden', 'operacion_valida': 'Igualdad (==), moda', 'operacion_invalida': 'Mayor/menor (>), suma, media'},
        {'variable': 'plan_type', 'tipo_semantico': 'Nominal', 'soporte': 'Conjunto finito sin orden', 'riesgo_si_mal_tratado': 'Falsa noción de distancia y orden', 'operacion_valida': 'Igualdad (==), moda', 'operacion_invalida': 'Mayor/menor (>), suma, media'},
        {'variable': 'education_level', 'tipo_semantico': 'Ordinal', 'soporte': 'Conjunto finito ordenado', 'riesgo_si_mal_tratado': 'Pérdida de jerarquía u asunción de distancias equidistantes', 'operacion_valida': 'Mayor/menor (>), mediana', 'operacion_invalida': 'Suma, media (estrictamente)'},
        {'variable': 'loyalty_tier', 'tipo_semantico': 'Ordinal', 'soporte': 'Conjunto finito ordenado', 'riesgo_si_mal_tratado': 'Pérdida de jerarquía', 'operacion_valida': 'Mayor/menor (>), mediana', 'operacion_invalida': 'Suma, media (estrictamente)'},
        {'variable': 'satisfaction_score', 'tipo_semantico': 'Ordinal', 'soporte': 'Enteros [1,10]', 'riesgo_si_mal_tratado': 'Falsa asunción de intervalo continuo', 'operacion_valida': 'Mayor/menor, mediana', 'operacion_invalida': 'Promedio (estrictamente, asume distancias iguales)'},
        {'variable': 'promo_open_rate_90d', 'tipo_semantico': 'Proporción', 'soporte': 'Continuo [0,1]', 'riesgo_si_mal_tratado': 'Predicciones fuera de rango [0,1]', 'operacion_valida': 'Suma ponderada, media', 'operacion_invalida': 'Sumas no acotadas'},
        {'variable': 'share_web', 'tipo_semantico': 'Composición', 'soporte': 'Simplex (suma=1)', 'riesgo_si_mal_tratado': 'Correlaciones espurias por restricción de suma', 'operacion_valida': 'Centered Log Ratio (CLR)', 'operacion_invalida': 'Correlación de Pearson cruda'},
        {'variable': 'share_mobile', 'tipo_semantico': 'Composición', 'soporte': 'Simplex (suma=1)', 'riesgo_si_mal_tratado': 'Correlaciones espurias por restricción de suma', 'operacion_valida': 'Centered Log Ratio (CLR)', 'operacion_invalida': 'Correlación de Pearson cruda'},
        {'variable': 'share_store', 'tipo_semantico': 'Composición', 'soporte': 'Simplex (suma=1)', 'riesgo_si_mal_tratado': 'Correlaciones espurias por restricción de suma', 'operacion_valida': 'Centered Log Ratio (CLR)', 'operacion_invalida': 'Correlación de Pearson cruda'},
        {'variable': 'share_partner', 'tipo_semantico': 'Composición', 'soporte': 'Simplex (suma=1)', 'riesgo_si_mal_tratado': 'Correlaciones espurias por restricción de suma', 'operacion_valida': 'Centered Log Ratio (CLR)', 'operacion_invalida': 'Correlación de Pearson cruda'},
        {'variable': 'peak_hour_local', 'tipo_semantico': 'Cíclica', 'soporte': 'Enteros mod 24', 'riesgo_si_mal_tratado': 'Salto artificial entre 23 y 0', 'operacion_valida': 'Seno/Coseno', 'operacion_invalida': 'Distancia euclidiana directa'}
    ]
    pd.DataFrame(tabla_w4_1).to_csv('W4_gonzalez/salidas_W4/Tabla_W4.1.csv', index=False)

    bitacora = []

    # 2. Figura W4.1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='region', order=df['region'].value_counts().index, ax=axes[0], palette=COLORS)
    axes[0].set_title('Frecuencias: Region (Nominal)')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.countplot(data=df, x='plan_type', order=df['plan_type'].value_counts().index, ax=axes[1], palette=COLORS)
    axes[1].set_title('Frecuencias: Plan Type (Nominal)')
    
    min_cat = df['region'].value_counts().idxmin()
    min_count = df['region'].value_counts().min()
    axes[0].text(x=len(df['region'].unique())-1, y=min_count+50, s='Rareza categórica', color='#2E4057', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('W4_gonzalez/salidas_W4/Figura_W4.1_frecuencias_nominales.png', dpi=300)
    plt.close()
    
    bitacora.append({'variable_o_bloque': 'Nominales (region, plan_type)', 'tipo_y_soporte': 'Nominal discreta', 'hallazgo_empirico': f'Categoría minoritaria identificada ({min_cat})', 'riesgo_metodologico': 'Label encoding asume orden', 'alternativas_consideradas': 'Label encoding vs One-Hot', 'decision_final': 'One-hot encoding con umbral para raros', 'justificacion': 'Evita jerarquías espurias', 'validacion': 'Validación en pipeline'})

    # 3. Figura W4.2: Compositions and CLR
    comp_cols = ['share_web', 'share_mobile', 'share_store', 'share_partner']
    raw_corr = df[comp_cols].corr()
    
    # Calculate CLR
    eps = 1e-6
    comp_data = df[comp_cols] + eps
    comp_data = comp_data.div(comp_data.sum(axis=1), axis=0)
    geom_mean = np.exp(np.log(comp_data).mean(axis=1))
    clr_data = np.log(comp_data.div(geom_mean, axis=0))
    clr_corr = clr_data.corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(raw_corr, annot=True, cmap='vlag', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Correlación Bruta (Espuria por restricción suma=1)')
    sns.heatmap(clr_corr, annot=True, cmap='vlag', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Correlación CLR (Centered Log Ratio)')
    plt.tight_layout()
    plt.savefig('W4_gonzalez/salidas_W4/Figura_W4.2_composicion_CLR.png', dpi=300)
    plt.close()
    
    bitacora.append({'variable_o_bloque': 'Composición (shares)', 'tipo_y_soporte': 'Simplex suma=1', 'hallazgo_empirico': 'Correlaciones brutas negativas forzadas', 'riesgo_metodologico': 'Colinealidad y efectos espurios en modelos', 'alternativas_consideradas': 'Usar 3 shares vs usar CLR en 4 shares', 'decision_final': 'Aplicar CLR log-ratio', 'justificacion': 'Transforma al espacio real, eliminando restricción de suma', 'validacion': 'Comparación de correlaciones'})

    # 4. Figura W4.3: Cyclic peak_hour_local
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['peak_hour_local'], bins=24, ax=axes[0], color=COLORS[3])
    axes[0].set_title('Histograma Crudo (Salto artificial 23->0)')
    axes[0].set_xlim(-1, 24)
    
    df['sin_hour'] = np.sin(2 * np.pi * df['peak_hour_local'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['peak_hour_local'] / 24)
    scatter = axes[1].scatter(df['sin_hour'], df['cos_hour'], c=df['peak_hour_local'], cmap='twilight', alpha=0.6)
    axes[1].set_title('Codificación Seno/Coseno (Geometría circular correcta)')
    axes[1].set_xlabel('sin_hour')
    axes[1].set_ylabel('cos_hour')
    axes[1].axis('equal')
    fig.colorbar(scatter, ax=axes[1], label='Hora local')
    plt.tight_layout()
    plt.savefig('W4_gonzalez/salidas_W4/Figura_W4.3_ciclica_peak_hour.png', dpi=300)
    plt.close()

    bitacora.append({'variable_o_bloque': 'Cíclica (peak_hour_local)', 'tipo_y_soporte': 'Enteros [0,23]', 'hallazgo_empirico': 'Representación cruda distorsiona distancias temporales', 'riesgo_metodologico': 'El modelo interpretaría la medianoche y 11pm como lejanas', 'alternativas_consideradas': 'Binning vs Seno/Coseno', 'decision_final': 'Codificación Seno y Coseno', 'justificacion': 'Garantiza distancia euclidiana correcta', 'validacion': 'Validación visual circular'})

    df_bit = pd.DataFrame(bitacora)
    df_bit.to_csv('W4_gonzalez/salidas_W4/bitacora_W4.csv', mode='a', index=False, header=False)
    print("Salidas de representación listas.")

if __name__ == "__main__":
    main()
