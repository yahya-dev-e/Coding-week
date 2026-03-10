
import pandas as pd
import numpy as np  
def optimize_memory(df):
    """
    Réduit la taille mémoire d'un DataFrame en convertissant les types de données 
    vers des formats plus légers.
    """
    # On calcule la mémoire initiale  
    start_mem = df.memory_usage().sum() / 1024**2
    
    # On boucle sur chaque colonne du tableau
    for col in df.columns:
        col_type = df[col].dtype
        
        # On ne traite que les colonnes numériques (pas les textes/objets)
        if col_type != object:
            c_min = df[col].min() # Valeur minimale dans la colonne
            c_max = df[col].max() # Valeur maximale dans la colonne
            
            # Cas des nombres entiers (Integer)
            if str(col_type)[:3] == 'int':
                # Si les valeurs tiennent dans un petit entier (8 bits), on convertit
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # Sinon, on vérifie pour le format 16 bits
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # On pourrait continuer pour int32, mais le PDF suggère surtout int32 
                else:
                    df[col] = df[col].astype(np.int32)
            
            # Cas des nombres à virgule (Float)
            else:
                # On passe de float64 (précision lourde) à float32 (plus léger) 
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    # Calcul de la mémoire finale après optimisation
    end_mem = df.memory_usage().sum() / 1024**2
    
    # Affichage du gain pour la documentation demandée dans le notebook 
    print(f'Mémoire réduite à {end_mem:.2f} MB (Gain de {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df