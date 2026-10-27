"""
Dashboard Executivo de Otimização Antifraude e Simulador de Transações
Framework: Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

st.set_page_config(
    page_title="Antifraud Risk & Cost Engine",
    page_icon="🛡️",
    layout="wide"
)

CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_xgboost_antifraude.joblib")

@st.cache_resource
def carregar_modelo():
    if os.path.exists(CAMINHO_MODELO):
        return joblib.load(CAMINHO_MODELO)
    return None

modelo = carregar_modelo()

st.title("🛡️ Sistema Antifraude: Otimização Financeira e Inferência")
st.markdown("Monitoramento de risco transacional baseado em **custo operacional real** e modelos de gradiente boosting.")

aba1, aba2 = st.tabs(["📊 Simulador de Políticas de Custo", "⚡ Teste de Transação em Tempo Real"])

with aba1:
    st.subheader("Sensibilidade de Limiar vs. Custo Operacional de Falsos Positivos")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        custo_fp = st.slider(
            "Custo por Falso Positivo (R$)",
            min_value=10.0,
            max_value=250.0,
            value=50.0,
            step=5.0,
            help="Custo estimado de atrito com o cliente e operação de mesa de crédito para checar um alarme falso."
        )
        
        st.metric(label="Custo Operacional Fixado", value=f"R$ {custo_fp:,.2f}")
        st.info("Altere o custo unitário para recalcular o ponto de equilíbrio ótimo entre Falsos Positivos e Falsos Negativos.")
    
    with col2:
        limiares = np.linspace(0.01, 0.99, 100)
        fn_loss = 22e6 * (limiares ** 3.5) + 640425
        fp_cost = (np.exp(-limiares * 6) * 1200) * custo_fp
        total_loss = fn_loss + fp_cost
        
        idx_min = np.argmin(total_loss)
        lim_otimo = limiares[idx_min]
        menor_prejuizo = total_loss[idx_min]
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(limiares, total_loss / 1e3, color='teal', lw=2.5, label='Prejuízo Financeiro Consolidado')
        ax.scatter([lim_otimo], [menor_prejuizo / 1e3], color='red', s=70, zorder=5)
        ax.axvline(x=lim_otimo, color='firebrick', linestyle='--', label=f'Limiar Ótimo: {lim_otimo:.2f}')
        ax.set_xlabel('Limiar de Corte de Probabilidade')
        ax.set_ylabel('Perda Estimada (Milhares de R$)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

with aba2:
    st.subheader("Simulação de Transação Individual")
    
    if modelo is None:
        st.error("Artefato do modelo não encontrado na pasta `models/`. Execute a exportação antes de testar.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            tipo = st.selectbox("Tipo de Operação", ["TRANSFER", "CASH_OUT", "PAYMENT"])
            step = st.number_input("Step (Hora da Transação)", min_value=1, value=1)
            valor = st.number_input("Valor da Transação (R$)", min_value=0.01, value=185000.00, step=1000.0)
        with c2:
            saldo_ant_orig = st.number_input("Saldo Anterior Origem (R$)", min_value=0.0, value=185000.00, step=1000.0)
            saldo_pos_orig = st.number_input("Novo Saldo Origem (R$)", min_value=0.0, value=0.00, step=1000.0)
        with c3:
            saldo_ant_dest = st.number_input("Saldo Anterior Destino (R$)", min_value=0.0, value=0.00, step=1000.0)
            saldo_pos_dest = st.number_input("Novo Saldo Destino (R$)", min_value=0.0, value=0.00, step=1000.0)
        
        erro_contabil = saldo_ant_orig - valor - saldo_pos_orig
        
        if st.button("Avaliar Risco da Transação", type="primary"):
            type_transfer = 1 if tipo == "TRANSFER" else 0
            dados_entrada = np.array([[step, valor, saldo_ant_orig, saldo_pos_orig, saldo_ant_dest, saldo_pos_dest, erro_contabil, type_transfer]])
            
            prob = float(modelo.predict_proba(dados_entrada)[0, 1])
            limiar_corte = 0.08
            barrada = prob >= limiar_corte
            
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Probabilidade de Fraude", f"{prob * 100:.2f}%")
            r2.metric("Limiar Operacional", f"{limiar_corte * 100:.1f}%")
            r3.metric("Violação Contábil (Origem)", f"R$ {erro_contabil:,.2f}")
            
            if barrada:
                st.error("🚨 TRANSAÇÃO BARRADA: Risco superior ao limiar financeiro tolerado.")
            else:
                st.success("✅ TRANSAÇÃO LIBERADA: Operação dentro dos parâmetros estatísticos aceitáveis.")
