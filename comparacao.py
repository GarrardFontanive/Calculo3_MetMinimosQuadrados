
import csv
import io
import numpy as np

from contextlib import redirect_stdout

from otimizacao import (
    carregar_dados,
    montar_sistema,
    eliminacao_gauss
)


# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

n, tempos = carregar_dados()

A, B = montar_sistema(n, tempos)


# ==========================================
# ELIMINACAO DE GAUSS
# ==========================================

with redirect_stdout(io.StringIO()):

    gauss = eliminacao_gauss(A, B)


# ==========================================
# RESULTADOS DE GAUSS-SEIDEL
# ==========================================

with open(
    "convergencia.csv",
    encoding="utf-8"
) as arquivo:

    historico = list(csv.DictReader(arquivo))

ultima = historico[-1]

seidel = np.array([
    float(ultima["a"]),
    float(ultima["b"])
])

iteracoes = int(ultima["iteracao"])


# ==========================================
# SOLUCAO DE REFERENCIA
# ==========================================

referencia = np.linalg.solve(A, B)


# ==========================================
# METRICAS DE COMPARACAO
# ==========================================

def calcular_mse(solucao):

    a, b = solucao

    previstos = a * n + b

    return np.mean((tempos - previstos)**2)


def calcular_residuo(solucao):

    return np.linalg.norm(
        A @ solucao - B,
        ord=np.inf
    )


def calcular_diferenca(solucao):

    return np.linalg.norm(
        solucao - referencia,
        ord=np.inf
    )


# ==========================================
# APRESENTACAO DOS RESULTADOS
# ==========================================

print("=" * 78)
print("COMPARACAO DOS METODOS NUMERICOS")
print("=" * 78)

print(
    f"{'Parametro':<27}"
    f"{'Gauss':>17}"
    f"{'Gauss-Seidel':>17}"
    f"{'NumPy':>17}"
)

print("-" * 78)

for indice, nome in enumerate(["Coeficiente a", "Coeficiente b"]):

    print(
        f"{nome:<27}"
        f"{gauss[indice]:17.10f}"
        f"{seidel[indice]:17.10f}"
        f"{referencia[indice]:17.10f}"
    )

print(
    f"{'Erro quadratico medio':<27}"
    f"{calcular_mse(gauss):17.10f}"
    f"{calcular_mse(seidel):17.10f}"
    f"{calcular_mse(referencia):17.10f}"
)

print(
    f"{'Residuo do sistema':<27}"
    f"{calcular_residuo(gauss):17.4e}"
    f"{calcular_residuo(seidel):17.4e}"
    f"{calcular_residuo(referencia):17.4e}"
)

print(
    f"{'Diferenca para NumPy':<27}"
    f"{calcular_diferenca(gauss):17.4e}"
    f"{calcular_diferenca(seidel):17.4e}"
    f"{0.0:17.4e}"
)

print("-" * 78)

print(f"Iteracoes de Gauss-Seidel: {iteracoes}")

print(
    "Diferenca entre Gauss e Gauss-Seidel: "
    f"{np.linalg.norm(gauss - seidel, ord=np.inf):.4e}"
)

print("=" * 78)

print("CONCLUSAO:")

if np.allclose(
    gauss,
    seidel,
    atol=1e-10,
    rtol=0
):

    print("Os metodos apresentam resultados equivalentes")
    print("dentro da tolerancia numerica estabelecida.")

else:

    print("Os resultados ultrapassam a tolerancia.")

print("=" * 78)
