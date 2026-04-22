import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main():
    os.makedirs('W4_gonzalez/data_splits', exist_ok=True)
    os.makedirs('W4_gonzalez/salidas_W4', exist_ok=True)

    print("PASO 00: Setup y Partición de Datos")
    print("-" * 40)

    # 1. Load data
    file_path = '_recursos/eda_laboratorio_clientes_v1.csv'
    df = pd.read_csv(file_path)
    initial_rows = len(df)
    print(f"Filas iniciales cargadas: {initial_rows}")

    # 2. Deduplicate
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df = df.sort_values(by=['customer_id', 'snapshot_date'], ascending=[True, False])
    df_dedup = df.drop_duplicates(subset='customer_id', keep='first').copy()
    final_rows = len(df_dedup)
    removed_rows = initial_rows - final_rows
    print(f"Filas removidas por duplicidad de customer_id: {removed_rows}")
    print(f"Filas resultantes: {final_rows}")

    # 3. Create target variable
    df_dedup['spend_positive'] = (df_dedup['future_spend_60d'] > 0).astype(int)

    # 4. Split 60/20/20
    X = df_dedup.drop(columns=['spend_positive'])
    y = df_dedup['spend_positive']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=2026, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=2026, stratify=y_temp
    )

    # Rejoin to save
    train_df = X_train.copy()
    train_df['spend_positive'] = y_train

    val_df = X_val.copy()
    val_df['spend_positive'] = y_val

    test_df = X_test.copy()
    test_df['spend_positive'] = y_test

    print(f"Tamaño Train: {len(train_df)} ({len(train_df)/final_rows:.0%})")
    print(f"Tamaño Validation: {len(val_df)} ({len(val_df)/final_rows:.0%})")
    print(f"Tamaño Test: {len(test_df)} ({len(test_df)/final_rows:.0%})")

    # Save
    train_df.to_csv('W4_gonzalez/data_splits/train.csv', index=False)
    val_df.to_csv('W4_gonzalez/data_splits/val.csv', index=False)
    test_df.to_csv('W4_gonzalez/data_splits/test.csv', index=False)

    print("Splits guardados en W4_gonzalez/data_splits/")
    
    # Save a small bitacora entry
    bitacora_path = 'W4_gonzalez/salidas_W4/bitacora_W4.csv'
    if not os.path.exists(bitacora_path):
        with open(bitacora_path, 'w') as f:
            f.write("variable_o_bloque,tipo_y_soporte,hallazgo_empirico,riesgo_metodologico,alternativas_consideradas,decision_final,justificacion,validacion\n")
    
    # Register deduplication entry
    with open(bitacora_path, 'a') as f:
        f.write(f'customer_id,identificador único,"Se encontraron {removed_rows} registros duplicados por el mismo cliente en distintas fechas",Fuga de información cruzada temporalmente o sobre-representación del mismo sujeto,Promediar historial vs quedarse con el más reciente,Quedarse con snapshot_date más reciente,Evita ensuciar la ventana temporal objetivo del último corte,"Verificación 1-a-1 de cardinalidad (ahora {final_rows} clientes únicos)"\n')

if __name__ == "__main__":
    main()
