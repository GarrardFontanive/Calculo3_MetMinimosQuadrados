
import csv
import sys
import numpy as np


TOLERANCIA = 1e-10
MAX_ITERACOES = 1000

np.set_printoptions(precision=8, suppress=True)


# ==========================================
# LEITURA DOS DADOS EXPERIMENTAIS
# ==========================================

def carregar_dados():

    with open("dados.csv", encoding="utf-8") as arquivo:

        dados = list(csv.DictReader(arquivo))

    n = np.array([
        float(item["n"]) for item in dados
    ])

    tempos = np.array([
        float(item["tempo_ms"]) for item in dados
    ])

    return n, tempos


# ==========================================
# SISTEMA NORMAL DOS MINIMOS QUADRADOS
# ==========================================

def montar_sistema(n, tempos):

    A = np.array([
        [np.sum(n**2), np.sum(n)],
        [np.sum(n), len(n)]
    ], dtype=float)

    B = np.array([
        np.sum(n * tempos),
        np.sum(tempos)
    ], dtype=float)

    return A, B


# ==========================================
# ELIMINACAO DE GAUSS
# ==========================================

def eliminacao_gauss(A, B):

    M = np.column_stack((A, B)).copy()

    ordem = len(B)

    print("\nMatriz aumentada inicial:")
    print(M)

    for k in range(ordem - 1):

        # Pivotamento parcial
        pivo = k + np.argmax(abs(M[k:, k]))

        M[[k, pivo]] = M[[pivo, k]]

        if abs(M[k, k]) < 1e-15:
            raise ValueError("Pivo nulo.")

        for i in range(k + 1, ordem):

            multiplicador = M[i, k] / M[k, k]

            M[i] -= multiplicador * M[k]

            print(f"\nMultiplicador: {multiplicador:.8f}")

            print(
                f"L{i+1} = L{i+1} "
                f"- ({multiplicador:.8f}) L{k+1}"
            )

            print(M)

    # Substituicao retroativa

    solucao = np.zeros(ordem)

    for i in range(ordem - 1, -1, -1):

        if abs(M[i, i]) < 1e-15:
            raise ValueError("Sistema singular.")

        soma = np.dot(
            M[i, i+1:ordem],
            solucao[i+1:]
        )

        solucao[i] = (
            M[i, -1] - soma
        ) / M[i, i]

    return solucao


# ==========================================
# METODO DE GAUSS-SEIDEL
# ==========================================

def gauss_seidel(A, B):

    ordem = len(B)

    solucao = np.zeros(ordem)

    historico = []

    print("\nEstimativa inicial: (0, 0)")
    print(f"Tolerancia: {TOLERANCIA:.1e}\n")

    print(
        f"{'Iteracao':>9} "
        f"{'a':>14} "
        f"{'b':>14} "
        f"{'Residuo':>14}"
    )

    for k in range(1, MAX_ITERACOES + 1):

        anterior = solucao.copy()

        for i in range(ordem):

            soma_anterior = np.dot(
                A[i, :i],
                solucao[:i]
            )

            soma_posterior = np.dot(
                A[i, i+1:],
                anterior[i+1:]
            )

            solucao[i] = (
                B[i]
                - soma_anterior
                - soma_posterior
            ) / A[i, i]

        erro = np.linalg.norm(
            solucao - anterior,
            ord=np.inf
        )

        residuo = np.linalg.norm(
            A @ solucao - B,
            ord=np.inf
        )

        historico.append([
            k,
            solucao[0],
            solucao[1],
            erro,
            residuo
        ])

        if k <= 10 or k % 20 == 0:

            print(
                f"{k:9d} "
                f"{solucao[0]:14.8f} "
                f"{solucao[1]:14.8f} "
                f"{residuo:14.4e}"
            )

        if erro < TOLERANCIA and residuo < TOLERANCIA:

            print(f"\nConvergencia em {k} iteracoes.")

            break

    else:

        raise RuntimeError(
            "Limite de iteracoes atingido."
        )

    with open(
        "convergencia.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow([
            "iteracao",
            "a",
            "b",
            "erro_iterativo",
            "residuo"
        ])

        escritor.writerows(historico)

    return solucao


# ==========================================
# ANALISE DOS RESULTADOS
# ==========================================

def apresentar_resultados(solucao, n, tempos):

    a, b = solucao

    previstos = a * n + b

    residuos = tempos - previstos

    mse = np.mean(residuos**2)

    rmse = np.sqrt(mse)

    r2 = 1 - (
        np.sum(residuos**2)
        / np.sum((tempos - np.mean(tempos))**2)
    )

    print("\n" + "=" * 68)
    print("RESULTADO DA OTIMIZACAO")
    print("=" * 68)

    print(f"a = {a:.10f}")
    print(f"b = {b:.10f}")

    print(f"\nT(n) = {a:.8f}n {b:+.8f}")

    print(f"\nErro quadratico medio: {mse:.10f} ms^2")

    print(f"RMSE: {rmse:.10f} ms")

    print(f"R^2: {r2:.10f}")

    print("=" * 68)


# ==========================================
# MATRIZ HESSIANA
# ==========================================

def analisar_hessiana(A, n):

    H = (2 / len(n)) * A

    determinante = np.linalg.det(H)

    print("\nMatriz Hessiana:")
    print(H)

    print(f"\nDeterminante: {determinante:.10f}")

    print(f"E_aa: {H[0, 0]:.10f}")

    if determinante > 0 and H[0, 0] > 0:

        print("Classificacao: minimo estrito.")

        print("Funcao quadratica estritamente convexa.")

        print("Portanto, o minimo tambem e global.")


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

if __name__ == "__main__":

    n, tempos = carregar_dados()

    A, B = montar_sistema(n, tempos)

    metodo = sys.argv[1] if len(sys.argv) > 1 else "gauss"

    print("=" * 68)
    print("OTIMIZACAO POR MINIMOS QUADRADOS")
    print("=" * 68)

    print("\nMatriz dos coeficientes:")
    print(A)

    print("\nVetor dos termos independentes:")
    print(B)

    if metodo == "gauss":

        print("\nMETODO DE ELIMINACAO DE GAUSS")

        solucao = eliminacao_gauss(A, B)

    elif metodo == "seidel":

        print("\nMETODO DE GAUSS-SEIDEL")

        solucao = gauss_seidel(A, B)

    else:

        raise ValueError(
            "Utilize 'gauss' ou 'seidel'."
        )

    apresentar_resultados(solucao, n, tempos)

    analisar_hessiana(A, n)

    # Validacao independente pelo NumPy

    referencia = np.linalg.solve(A, B)

    diferenca = np.linalg.norm(
        solucao - referencia,
        ord=np.inf
    )

    print("\nValidacao pelo NumPy:")

    print(referencia)

    print(f"Diferenca numerica: {diferenca:.4e}")
