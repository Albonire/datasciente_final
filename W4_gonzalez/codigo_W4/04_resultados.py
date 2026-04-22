import os
import pandas as pd

def main():
    print("PASO 04: Verificación Final de Resultados")
    print("-" * 40)
    
    expected_files = [
        'W4_gonzalez/salidas_W4/Tabla_W4.1.csv',
        'W4_gonzalez/salidas_W4/Tabla_W4.2.csv',
        'W4_gonzalez/salidas_W4/Tabla_W4.3.csv',
        'W4_gonzalez/salidas_W4/Figura_W4.1_frecuencias_nominales.png',
        'W4_gonzalez/salidas_W4/Figura_W4.2_composicion_CLR.png',
        'W4_gonzalez/salidas_W4/Figura_W4.3_ciclica_peak_hour.png',
        'W4_gonzalez/salidas_W4/Figura_W4.4_faltante_avg_order.png',
        'W4_gonzalez/salidas_W4/bitacora_W4.csv'
    ]
    
    all_exist = True
    for file in expected_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")
            all_exist = False
            
    if all_exist:
        print("\nTodas las salidas han sido generadas correctamente.")
        # Print a short summary of the bitacora
        df = pd.read_csv('W4_gonzalez/salidas_W4/bitacora_W4.csv')
        print(f"Entradas en la bitácora: {len(df)}")
    else:
        print("\nFaltan algunas salidas. Revise la ejecución de los scripts anteriores.")

if __name__ == "__main__":
    main()
