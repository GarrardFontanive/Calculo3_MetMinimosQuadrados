
import csv
import time
import random
import platform
import statistics


# ==========================================
# CONFIGURACOES DO EXPERIMENTO
# ==========================================

TAMANHOS = [
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    600_000
]

REPETICOES = 15

ARQUIVO_SAIDA = "dados.csv"


# ==========================================
# ALGORITMO DE BUSCA SEQUENCIAL
# ==========================================

def busca_sequencial(vetor, alvo):

    for indice, elemento in enumerate(vetor):

        if elemento == alvo:
            return indice

    return -1


# ==========================================
# MEDICAO DO TEMPO DE EXECUCAO
# ==========================================

def medir_tempo(vetor, alvo):

    inicio = time.perf_counter_ns()

    resultado = busca_sequencial(vetor, alvo)

    fim = time.perf_counter_ns()

    if resultado != -1:
        raise RuntimeError("O alvo nao deveria existir.")

    tempo_ms = (fim - inicio) / 1_000_000

    return tempo_ms


# ==========================================
# EXPERIMENTO COMPUTACIONAL
# ==========================================

def executar_experimento():

    resultados = []

    print("=" * 72)
    print("EXPERIMENTO COMPUTACIONAL")
    print("Busca sequencial e metodo dos minimos quadrados")
    print("=" * 72)

    print(f"Sistema: {platform.system()}")
    print(f"Python: {platform.python_version()}")
    print(f"Repeticoes: {REPETICOES}")

    print("=" * 72)

    # Criacao dos vetores fora da medicao.
    vetores = {
        tamanho: list(range(tamanho))
        for tamanho in TAMANHOS
    }

    # O alvo nao existe em nenhum vetor.
    # A busca sempre percorre todos os elementos.
    alvo = -1

    tempos = {
        tamanho: []
        for tamanho in TAMANHOS
    }

    # Aquecimento inicial.
    for vetor in vetores.values():
        busca_sequencial(vetor, alvo)

    # Alterna a ordem dos testes para reduzir
    # efeitos da ordem de execucao.
    gerador = random.Random(42)

    for repeticao in range(REPETICOES):

        ordem = TAMANHOS.copy()

        gerador.shuffle(ordem)

        for tamanho in ordem:

            tempo = medir_tempo(
                vetores[tamanho],
                alvo
            )

            tempos[tamanho].append(tempo)

    # Tratamento dos resultados.
    for tamanho in TAMANHOS:

        mediana = statistics.median(tempos[tamanho])

        media = statistics.mean(tempos[tamanho])

        desvio = statistics.stdev(tempos[tamanho])

        resultados.append({
            "elementos": tamanho,
            "n": tamanho / 100_000,
            "tempo_ms": mediana,
            "media_ms": media,
            "desvio_padrao_ms": desvio
        })

        print(
            f"N = {tamanho:>7,d} | "
            f"Mediana = {mediana:>8.4f} ms | "
            f"Desvio = {desvio:>7.4f} ms"
        )

    return resultados


# ==========================================
# EXPORTACAO DOS DADOS
# ==========================================

def salvar_csv(resultados):

    with open(
        ARQUIVO_SAIDA,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=[
                "elementos",
                "n",
                "tempo_ms",
                "media_ms",
                "desvio_padrao_ms"
            ]
        )

        escritor.writeheader()

        escritor.writerows(resultados)

    print("=" * 72)

    print(f"Dados exportados para: {ARQUIVO_SAIDA}")

    print("=" * 72)


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

if __name__ == "__main__":

    resultados = executar_experimento()

    salvar_csv(resultados)
