
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from otimizacao import carregar_dados, montar_sistema


# ==========================================
# CONFIGURACOES GERAIS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
PASTA = ROOT / "figuras" / "graficos"
PASTA_DADOS = ROOT / "dados"
PASTA.mkdir(parents=True, exist_ok=True)

AZUL = "#24578A"
VERMELHO = "#C0392B"
VERDE = "#238636"
CINZA = "#64748B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.alpha": 0.25,
    "savefig.dpi": 300
})


# ==========================================
# DADOS EXPERIMENTAIS
# ==========================================

n, tempos = carregar_dados()

A, B = montar_sistema(n, tempos)

a_otimo, b_otimo = np.linalg.solve(A, B)

previstos = a_otimo * n + b_otimo

residuos = tempos - previstos

erro_minimo = np.mean(residuos**2)

r2 = 1 - (
    np.sum(residuos**2)
    / np.sum((tempos - np.mean(tempos))**2)
)


# ==========================================
# FUNCAO OBJETIVO
# ==========================================

def erro(a, b):

    return np.mean(
        (a[..., None] * n + b[..., None] - tempos)**2,
        axis=-1
    )


def salvar(figura, nome):

    for extensao in ["png", "pdf"]:

        destino = PASTA / f"{nome}.{extensao}"

        figura.savefig(
            destino,
            bbox_inches="tight",
            dpi=300
        )

        print(f"Gerado: {destino}")

    plt.close(figura)


# ==========================================
# 01 - REGRESSAO LINEAR
# ==========================================

def grafico_regressao():

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_continuo = np.linspace(0, 6.3, 300)

    t_continuo = (
        a_otimo * n_continuo + b_otimo
    )

    ax.plot(
        n_continuo,
        t_continuo,
        color=AZUL,
        linewidth=2,
        label="Modelo ajustado por mínimos quadrados"
    )

    ax.scatter(
        n,
        tempos,
        color=VERMELHO,
        edgecolors="white",
        linewidths=0.8,
        s=70,
        zorder=5,
        label="Tempos medidos"
    )

    for ni, ti in zip(n, tempos):

        ax.annotate(
            f"{ti:.3f}",
            (ni, ti),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    texto = (
        f"T(n) = {a_otimo:.6f}n "
        f"+ {b_otimo:.6f}\n"
        f"R² = {r2:.8f}"
    )

    ax.text(
        0.04,
        0.95,
        texto,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#CBD5E1"
        )
    )

    ax.set_title(
        "Ajuste linear do tempo de execução"
    )

    ax.set_xlabel(
        "Quantidade de elementos (× 100 mil)"
    )

    ax.set_ylabel("Tempo de execução (ms)")

    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, max(tempos) * 1.15)

    ax.set_xticks(np.arange(0, 7))

    ax.grid(True)

    ax.legend(loc="lower right")

    fig.tight_layout()

    salvar(fig, "01_regressao_linear")


# ==========================================
# MALHA DA FUNCAO OBJETIVO
# ==========================================

valores_a = np.linspace(
    a_otimo - 0.15,
    a_otimo + 0.15,
    160
)

valores_b = np.linspace(
    b_otimo - 0.6,
    b_otimo + 0.6,
    160
)

AA, BB = np.meshgrid(valores_a, valores_b)

EE = erro(AA, BB)


# ==========================================
# 02 - SUPERFICIE TRIDIMENSIONAL
# ==========================================

def grafico_superficie():

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    superficie = ax.plot_surface(
        AA,
        BB,
        EE,
        cmap="viridis",
        alpha=0.88,
        edgecolor="none",
        antialiased=True
    )

    ax.scatter(
        a_otimo,
        b_otimo,
        erro_minimo,
        color=VERMELHO,
        edgecolor="white",
        s=90,
        depthshade=False,
        label="Mínimo global"
    )

    ax.plot(
        [a_otimo, a_otimo],
        [b_otimo, b_otimo],
        [erro_minimo, np.max(EE)],
        color=VERMELHO,
        linestyle="--",
        linewidth=1.2
    )

    ax.set_title(
        "Superfície do erro quadrático médio",
        pad=20
    )

    ax.set_xlabel("Coeficiente a")
    ax.set_ylabel("Coeficiente b")
    ax.set_zlabel("E(a, b) (ms²)")

    ax.view_init(
        elev=29,
        azim=-130
    )

    ax.set_box_aspect((1, 1, 0.75))

    fig.colorbar(
        superficie,
        ax=ax,
        shrink=0.65,
        pad=0.12,
        label="Erro quadrático médio (ms²)"
    )

    ax.legend(loc="upper left")

    fig.tight_layout()

    salvar(fig, "02_superficie_erro")


# ==========================================
# 03 - CURVAS DE NIVEL
# ==========================================

def grafico_contornos():

    fig, ax = plt.subplots(figsize=(8, 6))

    niveis = erro_minimo + np.geomspace(
        0.0001,
        np.max(EE) - erro_minimo,
        16
    )

    contornos = ax.contour(
        AA,
        BB,
        EE,
        levels=niveis,
        cmap="viridis",
        linewidths=1.5
    )

    ax.clabel(
        contornos,
        inline=True,
        fontsize=8,
        fmt="%.4f"
    )

    ax.scatter(
        a_otimo,
        b_otimo,
        color=VERMELHO,
        edgecolor="white",
        s=90,
        zorder=5,
        label="Mínimo global"
    )

    ax.axvline(
        a_otimo,
        color=CINZA,
        linestyle=":",
        alpha=0.6
    )

    ax.axhline(
        b_otimo,
        color=CINZA,
        linestyle=":",
        alpha=0.6
    )

    ax.annotate(
        f"({a_otimo:.5f}; {b_otimo:.5f})",
        xy=(a_otimo, b_otimo),
        xytext=(25, 25),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color=VERMELHO
        ),
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        )
    )

    ax.set_title(
        "Curvas de nível da função objetivo"
    )

    ax.set_xlabel("Coeficiente a")
    ax.set_ylabel("Coeficiente b")

    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()

    salvar(fig, "03_curvas_de_nivel")


# ==========================================
# 04 - CONVERGENCIA DE GAUSS-SEIDEL
# ==========================================

def grafico_convergencia():

    with open(
        PASTA_DADOS / "convergencia.csv",
        encoding="utf-8"
    ) as arquivo:

        historico = list(csv.DictReader(arquivo))

    iteracoes = np.array([
        int(linha["iteracao"])
        for linha in historico
    ])

    residuos_sistema = np.array([
        float(linha["residuo"])
        for linha in historico
    ])

    erros_iterativos = np.array([
        float(linha["erro_iterativo"])
        for linha in historico
    ])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.semilogy(
        iteracoes,
        residuos_sistema,
        color=AZUL,
        linewidth=2,
        label="Resíduo do sistema linear"
    )

    ax.semilogy(
        iteracoes,
        erros_iterativos,
        color=CINZA,
        linestyle="--",
        linewidth=1.5,
        label="Variação entre iterações"
    )

    ax.axhline(
        1e-10,
        color=VERDE,
        linestyle="--",
        linewidth=1.5,
        label="Tolerância = 10⁻¹⁰"
    )

    ax.scatter(
        iteracoes[-1],
        residuos_sistema[-1],
        color=VERMELHO,
        s=55,
        zorder=5
    )

    ax.annotate(
        f"Convergência: {iteracoes[-1]} iterações",
        xy=(
            iteracoes[-1],
            residuos_sistema[-1]
        ),
        xytext=(-150, 35),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color=VERMELHO
        ),
        fontsize=10
    )

    ax.set_title(
        "Convergência do método de Gauss-Seidel"
    )

    ax.set_xlabel("Número de iterações")

    ax.set_ylabel(
        "Norma do resíduo / erro iterativo"
    )

    ax.grid(True, which="both")

    ax.legend(loc="upper right")

    fig.tight_layout()

    salvar(fig, "04_convergencia_gauss_seidel")


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

if __name__ == "__main__":

    print("=" * 70)
    print("GERACAO DOS GRAFICOS")
    print("=" * 70)

    grafico_regressao()

    grafico_superficie()

    grafico_contornos()

    grafico_convergencia()

    print("=" * 70)

    print(f"a = {a_otimo:.10f}")
    print(f"b = {b_otimo:.10f}")
    print(f"Erro minimo = {erro_minimo:.10f}")
    print(f"R² = {r2:.10f}")

    print("=" * 70)
    print("Todos os graficos foram gerados.")
    print("=" * 70)

