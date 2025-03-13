import os
import pandas as pd

# Directorio base donde están los sitios web (relativo al directorio actual)
data_dir = os.path.join(os.getcwd(), "webs", "data")  # Ruta relativa a la carpeta actual
output_csv = os.path.join(os.getcwd(), "webs", "combined_data.csv")  # Ruta relativa para el archivo de salida


#Lectura masiva de datos en archivos por sitio web.
def load_data(data_dir):
    data = []
    
    for site in os.listdir(data_dir):
        site_path = os.path.join(data_dir, site)
        csv_path = os.path.join(site_path, "data.csv")

        if os.path.isdir(site_path) and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["site"] = site 
            data.append(df)
    
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# Cargar los datos
df = load_data(data_dir)

# Guardar en CSV
if not df.empty:
    df.to_csv(output_csv, index=False)
    print(f"Datos guardados en {output_csv}")
else:
    print("No se encontraron datos para guardar.")