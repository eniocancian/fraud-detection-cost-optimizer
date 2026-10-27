"""
Pipeline de Detecção de Fraude com XGBoost e Otimização de Custo Financeiro
Base de dados: PaySim Synthetic Financial Datasets for Fraud Detection
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def carregar_e_preparar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega o dataset PaySim, filtra tipos transacionais com risco real de fraude
    (TRANSFER e CASH_OUT) e constrói variáveis contábeis de balanço.
    """
    print(f"-> A carregar base de dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv)

    print("-> A filtrar transações por tipos relevantes (TRANSFER e CASH_OUT)...")
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

    print("-> A criar variáveis de engenharia de atributos e balanço contábil...")
    # Erro contábil na conta de origem: diferença entre o débito esperado e o saldo restante
    df['balance_error_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']

    # Codificação do tipo de operação
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)

    return df


def avaliar_financeiro(
    limiar: float,
    y_real: np.ndarray,
    y_prob: np.ndarray,
    valores_transacoes: np.ndarray,
    custo_fp: float = 50.0
) -> dict:
    """
    Calcula o impacto financeiro consolidado dado um limiar de corte:
    Prejuízo Total = (Qtd FP * custo_fp) + Soma(Valores dos FN)
    """
    y_pred = (y_prob >= limiar).astype(int)

    # Identificação de Falsos Positivos e Falsos Negativos
    fp_mask = (y_real == 0) & (y_pred == 1)
    fn_mask = (y_real == 1) & (y_pred == 0)

    qtd_fp = int(np.sum(fp_mask))
    qtd_fn = int(np.sum(fn_mask))

    custo_operacional_fp = qtd_fp * custo_fp
    prejuizo_fraudes_fn = float(np.sum(valores_transacoes[fn_mask]))
    prejuizo_total = custo_operacional_fp + prejuizo_fraudes_fn

    return {
        'Limiar': round(limiar, 4),
        'Falsos Positivos (FP)': qtd_fp,
        'Custo Operacional FP (R$)': custo_operacional_fp,
        'Fraudes Perdidas (FN)': qtd_fn,
        'Prejuízo com Fraudes (R$)': prejuizo_fraudes_fn,
        'Prejuízo Financeiro Total (R$)': prejuizo_total
    }


def otimizar_limiar_custo(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    valores_transacoes: np.ndarray,
    custo_fp: float = 50.0
) -> pd.DataFrame:
    """
    Executa varredura de limiares entre 0.01 e 0.99 para identificar o ponto ótimo
    de corte que minimiza a perda financeira total.
    """
    limiares = np.linspace(0.01, 0.99, 100)
    resultados = []

    for lim in limiares:
        metr = avaliar_financeiro(lim, y_test, y_prob, valores_transacoes, custo_fp)
        resultados.append(metr)

    return pd.DataFrame(resultados)


def plotar_curva_custo(df_varredura: pd.DataFrame, limiar_otimo: float, menor_prejuizo: float, caminho_saida: str):
    """Gera e salva o gráfico da curva de prejuízo financeiro total versus limiar."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_varredura['Limiar'],
        df_varredura['Prejuízo Financeiro Total (R$)'] / 1e3,
        color='teal',
        lw=2.5,
        label='Prejuízo Total Consolidado (XGBoost)'
    )
    plt.scatter([limiar_otimo], [menor_prejuizo / 1e3], color='red', s=80, zorder=5)
    plt.axvline(
        x=limiar_otimo,
        color='firebrick',
        linestyle='--',
        lw=1.5,
        label=f'Ponto Ótimo: Limiar {limiar_otimo:.2f} (R$ {menor_prejuizo:,.2f})'
    )
    plt.xlabel('Limiar de Probabilidade de Fraude', fontsize=11)
    plt.ylabel('Prejuízo Total Estimado (Milhares de R$)', fontsize=11)
    plt.title('Otimização de Limiar Baseada em Custo Financeiro Real (XGBoost)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()
    print(f"-> Gráfico da curva de custo salvo em: {caminho_saida}")


def plotar_importancia_features(modelo: xgb.XGBClassifier, caminho_saida: str):
    """Gera e salva o gráfico de importância nativa das variáveis por Ganho (Gain)."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    importancias_ganho = modelo.get_booster().get_score(importance_type='gain')
    df_imp = pd.DataFrame({
        'Feature': list(importancias_ganho.keys()),
        'Ganho': list(importancias_ganho.values())
    }).sort_values(by='Ganho', ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(df_imp['Feature'], df_imp['Ganho'], color='teal', edgecolor='black', alpha=0.8)
    plt.xlabel('Importância Média por Ganho (Gain)', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.title('Importância das Variáveis no XGBoost (Métrica de Ganho)', fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()
    print(f"-> Gráfico de feature importance salvo em: {caminho_saida}")


def main():
    parser = argparse.ArgumentParser(description="Esteira Antifraude XGBoost com Otimização de Custo")
    parser.add_argument("--data_path", type=str, default="data/paysim.csv", help="Caminho do arquivo CSV do PaySim")
    parser.add_argument("--custo_fp", type=float, default=50.0, help="Custo operacional em R$ por Falso Positivo")
    args = parser.parse_args()

    # 1. Carga e Preparação
    df = carregar_e_preparar_dados(args.data_path)

    features = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_error_orig', 'type_TRANSFER'
    ]
    X = df[features]
    y = df['isFraud']

    # 2. Divisão Estratificada (80% treino / 20% teste)
    print("-> A particionar treino e teste com estratificação...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Treino do XGBoost com compensação de escala
    peso_escala = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"-> Proporção de desbalanceamento calculada: {peso_escala:.2f}")

    modelo_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=peso_escala,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("-> A treinar modelo XGBoost...")
    modelo_xgb.fit(X_train, y_train)

    # 4. Avaliação com Limiar Padrão (0.50)
    y_prob = modelo_xgb.predict_proba(X_test)[:, 1]
    valores_transacoes = X_test['amount'].values

    print("\n=== Matriz de Confusão (Limiar 0.50) ===")
    pred_050 = (y_prob >= 0.50).astype(int)
    print(confusion_matrix(y_test, pred_050))
    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, pred_050, target_names=['Lícito', 'Fraude'], digits=4))

    res_050 = avaliar_financeiro(0.50, y_test.values, y_prob, valores_transacoes, args.custo_fp)
    print("\n=== Impacto Financeiro (Limiar Padrão 0.50) ===")
    print(pd.DataFrame([res_050]).to_string(index=False))

    # 5. Otimização de Limiar por Custo
    print("\n-> A executar varredura contínua de limiares financeiros...")
    df_varredura = otimizar_limiar_custo(y_test.values, y_prob, valores_transacoes, args.custo_fp)

    idx_otimo = df_varredura['Prejuízo Financeiro Total (R$)'].idxmin()
    limiar_otimo = df_varredura.loc[idx_otimo, 'Limiar']
    menor_prejuizo = df_varredura.loc[idx_otimo, 'Prejuízo Financeiro Total (R$)']

    print(f"\nLimiar financeiramente ótimo: {limiar_otimo:.2f}")
    print(f"Menor prejuízo financeiro total: R$ {menor_prejuizo:,.2f}")
    print("\n=== Resultado Consolidado no Limiar Ótimo ===")
    print(df_varredura.loc[[idx_otimo]].to_string(index=False))

    # 6. Geração dos Gráficos
    plotar_curva_custo(df_varredura, limiar_otimo, menor_prejuizo, "figures/curva_custo_otimizada.png")
    plotar_importancia_features(modelo_xgb, "figures/feature_importance_gain.png")


if __name__ == "__main__":
    main()