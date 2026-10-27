"""
API REST para Inferência Antifraude em Tempo Real
Framework: FastAPI + Pydantic + XGBoost
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Antifraud Risk Engine API",
    description="Serviço de avaliação transacional de risco financeiro com XGBoost calibrado a custo ótimo.",
    version="1.0.0"
)

CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_xgboost_antifraude.joblib")

try:
    modelo = joblib.load(CAMINHO_MODELO)
except Exception:
    modelo = None

LIMIAR_OTIMO = 0.08


class TransacaoInput(BaseModel):
    step: int = Field(..., example=1, description="Passo temporal da simulação (hora)")
    amount: float = Field(..., gt=0, example=150000.00, description="Valor transacionado em R$")
    oldbalanceOrg: float = Field(..., ge=0, example=150000.00, description="Saldo inicial na conta de origem")
    newbalanceOrig: float = Field(..., ge=0, example=0.00, description="Saldo resultante na conta de origem")
    oldbalanceDest: float = Field(..., ge=0, example=0.00, description="Saldo inicial no favorecido")
    newbalanceDest: float = Field(..., ge=0, example=0.00, description="Saldo resultante no favorecido")
    type: str = Field(..., example="TRANSFER", description="Tipo de transação: TRANSFER ou CASH_OUT")


class DecisaoOutput(BaseModel):
    probabilidade_fraude: float
    limiar_utilizado: float
    status_transacao: str
    recomendacao: str
    balance_error_orig: float


@app.get("/")
def health_check():
    return {
        "status": "online",
        "modelo_carregado": modelo is not None,
        "limiar_corte": LIMIAR_OTIMO
    }


@app.post("/predict", response_model=DecisaoOutput)
def prever_fraude(transacao: TransacaoInput):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Artefato do modelo XGBoost não localizado em models/.")
    
    tipo_upper = transacao.type.upper()
    if tipo_upper not in ['TRANSFER', 'CASH_OUT']:
        return DecisaoOutput(
            probabilidade_fraude=0.0,
            limiar_utilizado=LIMIAR_OTIMO,
            status_transacao="APROVADA",
            recomendacao="Operação de baixo risco intrínseco (fora do escopo crítico).",
            balance_error_orig=0.0
        )
    
    balance_error_orig = transacao.oldbalanceOrg - transacao.amount - transacao.newbalanceOrig
    type_transfer = 1 if tipo_upper == 'TRANSFER' else 0
    
    vetor_features = np.array([[
        transacao.step,
        transacao.amount,
        transacao.oldbalanceOrg,
        transacao.newbalanceOrig,
        transacao.oldbalanceDest,
        transacao.newbalanceDest,
        balance_error_orig,
        type_transfer
    ]])
    
    prob = float(modelo.predict_proba(vetor_features)[0, 1])
    eh_fraude = prob >= LIMIAR_OTIMO
    
    return DecisaoOutput(
        probabilidade_fraude=round(prob, 4),
        limiar_utilizado=LIMIAR_OTIMO,
        status_transacao="BARRADA / ALERTA DE RISCO" if eh_fraude else "APROVADA",
        recomendacao="Encaminhar para validação operacional (Mesa de Fraude)" if eh_fraude else "Liberar liquidação imediata.",
        balance_error_orig=round(balance_error_orig, 2)
    )
