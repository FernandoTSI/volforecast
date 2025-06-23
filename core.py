import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VolForecastStrategy:
    def __init__(self,
                 ticker="^BVSP",
                 window_vol=20,
                 desvio_std=1.7,
                 max_drawdown=0.25,
                 start="2020-01-01",
                 end=None,
                 freq="daily"):
        self.ticker = ticker
        self.window_vol = window_vol
        self.desvio_std = desvio_std
        self.max_drawdown = -abs(max_drawdown)  # Negativo
        self.start = start
        self.end = end
        self.freq = freq

        self.capital_inicial = 1.0
        self.dados = None
        self.precos = None
        self.sinais = None
        self.resultados = None
        self.operacoes = []

    def baixar_dados(self):
        self.dados = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True)
        self.precos = self.dados['Close']

    def calcular_sinais(self):
        media = self.precos.rolling(self.window_vol).mean()
        vol = self.precos.rolling(self.window_vol).std()
        res = media + self.desvio_std * vol
        sup = media - self.desvio_std * vol

        sinais = pd.DataFrame(index=self.precos.index)
        sinais['preco'] = self.precos
        sinais['res'] = res
        sinais['sup'] = sup
        sinais['compra'] = (self.precos < sup) & (self.precos.shift(1) >= sup.shift(1))
        sinais['venda'] = (self.precos > res) & (self.precos.shift(1) <= res.shift(1))

        self.sinais = sinais

    def executar_backtest(self):
        capital = self.capital_inicial
        max_capital = capital
        capital_hist = [capital]
        drawdowns = [0]
        datas = [self.precos.index[0]]
        self.operacoes = []
        stop = False

        for i in range(1, len(self.sinais)-1):
            if stop:
                capital_hist.append(capital)
                datas.append(self.precos.index[i])
                drawdowns.append(drawdowns[-1])
                continue

            entrada = False
            lucro = 0
            tipo_op = "hold"

            if self.sinais['compra'].iloc[i]:
                preco_entrada = self.precos.iloc[i]
                preco_saida = self.precos.iloc[i+1]
                lucro = (preco_saida - preco_entrada) / preco_entrada
                tipo_op = "compra"
                entrada = True

            elif self.sinais['venda'].iloc[i]:
                preco_entrada = self.precos.iloc[i]
                preco_saida = self.precos.iloc[i+1]
                lucro = (preco_entrada - preco_saida) / preco_entrada
                tipo_op = "venda"
                entrada = True

            if entrada:
                capital *= (1 + lucro)
                self.operacoes.append({
                    "data_entrada": self.precos.index[i],
                    "data_saida": self.precos.index[i+1],
                    "tipo": tipo_op,
                    "lucro": lucro,
                    "capital_antes": capital_hist[-1],
                    "capital_depois": capital
                })

            max_capital = max(max_capital, capital)
            drawdown = (capital - max_capital) / max_capital

            capital_hist.append(capital)
            datas.append(self.precos.index[i+1])
            drawdowns.append(drawdown)

            if drawdown <= self.max_drawdown:
                print(f"⚠️ Stop acionado em {self.precos.index[i+1]} | Drawdown: {drawdown:.2%}")
                stop = True

        self.resultados = pd.DataFrame({
            "Capital": capital_hist,
            "Drawdown": drawdowns
        }, index=datas)

    def run(self):
        self.baixar_dados()
        self.calcular_sinais()
        self.executar_backtest()
        return self.resultados, pd.DataFrame(self.operacoes)

    def summary(self):
        if self.resultados is None:
            print("🚫 Execute `.run()` primeiro.")
            return

        final = self.resultados['Capital'].iloc[-1]
        retorno_total = (final - 1) * 100
        max_dd = self.resultados['Drawdown'].min() * 100
        num_ops = len(self.operacoes)
        acertos = sum(op['lucro'] > 0 for op in self.operacoes)
        taxa_acerto = (acertos / num_ops * 100) if num_ops else 0

        print("="*50)
        print("Resumo do Backtest")
        print("="*50)
        print(f"Capital Final: {final:.2f} ({retorno_total:.2f}%)")
        print(f"Drawdown Máximo: {max_dd:.2f}%")
        print(f"Número de Operações: {num_ops}")
        print(f"Taxa de Acerto: {taxa_acerto:.2f}%")
        print("="*50)

    def plot(self):
        if self.resultados is None:
            print("🚫 Execute `.run()` primeiro.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(self.precos, label="Preço", color="black")
        ax1.plot(self.sinais['res'], label="Resistência", color="red", linestyle='--')
        ax1.plot(self.sinais['sup'], label="Suporte", color="green", linestyle='--')
        ax1.set_title("Sinais de Entrada")
        ax1.legend()
        ax1.grid(True)

        df_ops = pd.DataFrame(self.operacoes)
        if not df_ops.empty:
            compras = df_ops[df_ops['tipo'] == 'compra']
            vendas = df_ops[df_ops['tipo'] == 'venda']
            ax1.scatter(compras['data_entrada'], self.precos.loc[compras['data_entrada']],
                        marker='^', color='green', label='Compra', s=100)
            ax1.scatter(vendas['data_entrada'], self.precos.loc[vendas['data_entrada']],
                        marker='v', color='red', label='Venda', s=100)

        ax2.plot(self.resultados['Capital'], label="Capital", color="blue")
        ax2.axhline(y=1, color="black", linestyle="--")
        ax3 = ax2.twinx()
        ax3.fill_between(self.resultados.index, self.resultados['Drawdown'], 0,
                         where=self.resultados['Drawdown'] < 0,
                         color="red", alpha=0.3, label="Drawdown")
        ax3.axhline(y=self.max_drawdown, color="darkred", linestyle="--", label="Limite DD")
        ax2.set_title("Capital e Drawdown")
        ax2.legend(loc="upper left")
        ax3.legend(loc="lower left")
        plt.tight_layout()
        plt.show()