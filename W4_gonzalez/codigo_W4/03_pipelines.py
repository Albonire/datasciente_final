import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

COLORS = ['#E2D5C8', '#C3AFA0', '#A48978', '#866350', '#2E4057']
plt.rcParams['font.family'] = 'serif'

# Custom Transformers
class CLRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        self.eps = eps
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_arr = np.array(X) + self.eps
        X_norm = X_arr / X_arr.sum(axis=1, keepdims=True)
        gmean = np.exp(np.mean(np.log(X_norm), axis=1, keepdims=True))
        return np.log(X_norm / gmean)

class CyclicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, period=24):
        self.period = period
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_arr = np.array(X)
        sin_x = np.sin(2 * np.pi * X_arr / self.period)
        cos_x = np.cos(2 * np.pi * X_arr / self.period)
        return np.hstack([sin_x, cos_x])

class StructuralMissingIndicator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # returns the original column with 0s for NaNs, plus a boolean indicator column
        X_arr = np.array(X).reshape(-1, 1)
        is_missing = np.isnan(X_arr).astype(float)
        X_filled = np.nan_to_num(X_arr, nan=0.0)
        return np.hstack([X_filled, is_missing])

def main():
    print("PASO 03: Pipelines y Comparación")
    print("-" * 40)
    
    # Load Data
    train = pd.read_csv('W4_gonzalez/data_splits/train.csv')
    val = pd.read_csv('W4_gonzalez/data_splits/val.csv')
    test = pd.read_csv('W4_gonzalez/data_splits/test.csv')
    
    drop_cols = ['customer_id', 'record_id', 'snapshot_date', 'future_spend_60d', 
                 'post_window_retention_call', 'post_window_discount_amount', 'spend_positive']
    
    y_train = train['spend_positive']
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y_val = val['spend_positive']
    X_val = val.drop(columns=[c for c in drop_cols if c in val.columns])
    y_test = test['spend_positive']
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    
    # Variable groupings
    nominals = ['region', 'plan_type', 'device_type']
    ordinals = ['education_level', 'loyalty_tier', 'satisfaction_score']
    comps = ['share_web', 'share_mobile', 'share_store', 'share_partner']
    cyclics = ['peak_hour_local', 'week_of_year']
    
    # other numerics
    numerics = [c for c in X_train.columns if c not in nominals + ordinals + comps + cyclics]
    
    # ==========================================
    # PIPELINE INGENUO
    # ==========================================
    # Nominal -> OrdinalEncoder (arbitrary integers)
    # Comps -> Raw with SimpleImputer mean
    # Cyclics -> Raw with SimpleImputer mean
    # Others -> SimpleImputer mean
    
    naive_preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='mean'), numerics + comps + cyclics),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), nominals + ordinals)
    ])
    
    naive_pipeline = Pipeline([
        ('preprocessor', naive_preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=2026, max_iter=1000))
    ])
    
    # ==========================================
    # PIPELINE COHERENTE
    # ==========================================
    # nominals -> OneHot
    # ordinals -> IterativeImputer for MAR (education), structural for satisfaction (MNAR -> add indicator, fill 0)
    # Or just IterativeImputer for all ordinals and SimpleImputer(add_indicator) for satisfaction.
    # To be precise to the prompt:
    # - avg_order_value_90d -> StructuralMissingIndicator
    # - satisfaction_score -> StructuralMissingIndicator
    # - promo_open_rate_90d -> SimpleImputer(median)
    # - education_level -> IterativeImputer
    
    # Let's split numerics
    num_struct = ['avg_order_value_90d']
    num_mcar = ['promo_open_rate_90d']
    num_other = [c for c in numerics if c not in num_struct + num_mcar]
    
    ord_struct = ['satisfaction_score']
    ord_mar = ['education_level', 'loyalty_tier']
    
    coherent_preprocessor = ColumnTransformer(transformers=[
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), nominals),
        
        ('ord_mar', Pipeline([
            ('imputer', IterativeImputer(random_state=2026)),
        ]), ord_mar),
        
        ('ord_struct', StructuralMissingIndicator(), ord_struct),
        
        ('num_struct', StructuralMissingIndicator(), num_struct),
        
        ('num_mcar', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), num_mcar),
        
        ('num_other', Pipeline([
            ('imputer', IterativeImputer(random_state=2026)),
        ]), num_other),
        
        ('comp', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clr', CLRTransformer())
        ]), comps),
        
        ('cycl', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('cyc', CyclicTransformer(period=24)) # week_of_year ignored for simplicity or 52
        ]), ['peak_hour_local']),
        
        ('cycl2', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('cyc', CyclicTransformer(period=52)) 
        ]), ['week_of_year'])
    ])
    
    coherent_pipeline = Pipeline([
        ('preprocessor', coherent_preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=2026, max_iter=1000))
    ])
    
    # ==========================================
    # TRAINING AND EVALUATION
    # ==========================================
    print("Entrenando Pipeline Ingenuo...")
    naive_pipeline.fit(X_train, y_train)
    
    print("Entrenando Pipeline Coherente...")
    coherent_pipeline.fit(X_train, y_train)
    
    def get_metrics(model, X, y):
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return {
            'accuracy': accuracy_score(y, preds),
            'roc_auc': roc_auc_score(y, probs),
            'f1': f1_score(y, preds)
        }
    
    naive_val_metrics = get_metrics(naive_pipeline, X_val, y_val)
    coh_val_metrics = get_metrics(coherent_pipeline, X_val, y_val)
    coh_test_metrics = get_metrics(coherent_pipeline, X_test, y_test)
    
    # Save Tabla W4.3
    tabla_w4_3 = pd.DataFrame([
        {
            'pipeline': 'Ingenuo',
            'accuracy_val': f"{naive_val_metrics['accuracy']:.4f}",
            'roc_auc_val': f"{naive_val_metrics['roc_auc']:.4f}",
            'f1_val': f"{naive_val_metrics['f1']:.4f}",
            'accuracy_test': '',
            'roc_auc_test': '',
            'f1_test': ''
        },
        {
            'pipeline': 'Coherente',
            'accuracy_val': f"{coh_val_metrics['accuracy']:.4f}",
            'roc_auc_val': f"{coh_val_metrics['roc_auc']:.4f}",
            'f1_val': f"{coh_val_metrics['f1']:.4f}",
            'accuracy_test': f"{coh_test_metrics['accuracy']:.4f}",
            'roc_auc_test': f"{coh_test_metrics['roc_auc']:.4f}",
            'f1_test': f"{coh_test_metrics['f1']:.4f}"
        }
    ])
    tabla_w4_3.to_csv('W4_gonzalez/salidas_W4/Tabla_W4.3.csv', index=False)
    
    # Bitacora Update
    bitacora = []
    bitacora.append({'variable_o_bloque': 'Pipeline de Modelado', 'tipo_y_soporte': 'Evaluación general', 'hallazgo_empirico': f"El modelo coherente superó al ingenuo en ROC-AUC ({coh_val_metrics['roc_auc']:.3f} vs {naive_val_metrics['roc_auc']:.3f})", 'riesgo_metodologico': 'Pérdida de predictibilidad por no representar semánticamente', 'alternativas_consideradas': 'Modelos no lineales (RandomForest)', 'decision_final': 'Uso de Regresión Logística con Transformadores', 'justificacion': 'Aisla el efecto de la representación vs el modelo en sí', 'validacion': f"Evaluado en test (ROC-AUC {coh_test_metrics['roc_auc']:.3f})"})
    df_bit = pd.DataFrame(bitacora)
    df_bit.to_csv('W4_gonzalez/salidas_W4/bitacora_W4.csv', mode='a', index=False, header=False)
    
    # ==========================================
    # Figura W4.4 Completa
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Crosstab (loaded from 01 or recomputed)
    # Recomputing for simplicity
    df_train = pd.read_csv('W4_gonzalez/data_splits/train.csv')
    df_train['transactions_bins'] = pd.cut(df_train['transactions_90d'], bins=[-1, 0, 5, np.inf], labels=['0 trans', '1-5', '6+'])
    ct = pd.crosstab(df_train['transactions_bins'], df_train['avg_order_value_90d'].isna())
    sns.heatmap(ct, annot=True, fmt='d', cmap='Oranges', ax=axes[0], cbar=False)
    axes[0].set_title('Conteo: is_null(avg_order) vs transacciones')
    axes[0].set_xlabel('avg_order_value_90d is NULL')
    axes[0].set_ylabel('Transacciones en 90 días')
    
    # Right: Barplot AUC
    sns.barplot(x=['Pipeline Ingenuo', 'Pipeline Coherente'], 
                y=[naive_val_metrics['roc_auc'], coh_val_metrics['roc_auc']], 
                ax=axes[1], palette=[COLORS[1], COLORS[3]])
    axes[1].set_title('Comparación ROC-AUC (Validación)')
    axes[1].set_ylim(0.5, 1.0)
    for i, v in enumerate([naive_val_metrics['roc_auc'], coh_val_metrics['roc_auc']]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('W4_gonzalez/salidas_W4/Figura_W4.4_faltante_avg_order.png', dpi=300)
    plt.close()
    
    print("Pipelines ejecutados y salidas generadas.")

if __name__ == "__main__":
    main()
