
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Remplace les '?' par NaN et convertit en numérique
    df = df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    return df

def preprocess_data(df):
    X = df.drop(columns=['Biopsy'])
    y = df['Biopsy']
    cols = X.columns
    
    # 1. Split (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Imputation (Médiane)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # --- OVERSAMPLING MANUEL ---
    # Séparation des classes
    X_pos = X_train_imp[y_train == 1]
    X_neg = X_train_imp[y_train == 0]
    
    # On duplique les cas positifs (Biopsy=1) pour égaler les négatifs
    np.random.seed(42)
    indices = np.random.choice(len(X_pos), size=len(X_neg), replace=True)
    X_pos_over = X_pos[indices]
    
    # Fusion pour créer le set d'entraînement équilibré
    X_train_final = np.vstack((X_neg, X_pos_over))
    y_train_final = np.hstack((np.zeros(len(X_neg)), np.ones(len(X_neg))))
    # ---------------------------

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, y_train_final, y_test, imputer, scaler, cols

def remove_outliers_iqr(df):
    """
    Supprime les lignes contenant des outliers basés sur la méthode IQR.
    On applique cela uniquement sur les colonnes numériques continues.
    """
    df_final = df.copy()
    
    # On cible les colonnes qui ont des valeurs élevées (comme l'âge ou le tabac)
    # On évite les colonnes binaires (0/1) comme les tests Schiller/Hinselmann
    cols_a_verifier = ['Age', 'Number of sexual partners', 'First sexual intercourse', 
                        'Num of pregnancies', 'Smokes (years)', 'Hormonal Contraceptives (years)']
    
    for col in cols_a_verifier:
        if col in df_final.columns:
            Q1 = df_final[col].quantile(0.25)
            Q3 = df_final[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Définition des bornes (coefficient 1.5 est le standard)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filtrage : on garde ce qui est entre les bornes
            initial_shape = df_final.shape[0]
            df_final = df_final[(df_final[col] >= lower_bound) & (df_final[col] <= upper_bound)]
            
            diff = initial_shape - df_final.shape[0]
            if diff > 0:
                print(f"🧹 {diff} outliers supprimés dans la colonne : {col}")
                
    return df_final




def supprimer_colonnes_zero(df):
    """
    Supprime les colonnes dont toutes les valeurs sont égales à 0.
    """
    # On identifie les colonnes où TOUTES les valeurs valent 0
    colonnes_a_supprimer = [col for col in df.columns if (df[col] == 0).all()]
    
    # On supprime ces colonnes
    df_nettoye = df.drop(columns=colonnes_a_supprimer)
    
    if colonnes_a_supprimer:
        print(f"✅ Colonne(s) supprimée(s) car remplie(s) de 0 : {colonnes_a_supprimer}")
    else:
        print("ℹ️ Aucune colonne ne contient uniquement des 0.")
        
    return df_nettoye





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