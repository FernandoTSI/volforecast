import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import time
import logging
from pathlib import Path
from datetime import datetime

class VolForecastStrategy:
    def __init__(
        self,
        ticker="^BVSP",
        window_vol=20,
        threshold=None,
        max_drawdown=0.25,
        start="2020-01-01",
        end=None,
        freq="daily",
        capital_inicial=10000,
        tentativas=3,
        intervalo_tentativas=5,
        output_dir="results"
    ):
        self.ticker = ticker
        self.window_vol = window_vol
        self.desvio_std = threshold if threshold is not None else 1.7
        self.threshold = threshold
        self.max_drawdown = max_drawdown
        self.start = start
        self.end = end if end else datetime.today().strftime('%Y-%m-%d')
        self.freq = freq
        self.capital_inicial = capital_inicial
        self.tentativas = tentativas
        self.intervalo_tentativas = intervalo_tentativas
        self.output_dir = output_dir
        self.precos = None
        self.sinais = None
        self.resultados = None
        self.operacoes = []
        self.metrics = {}

        # Configurar logging e diretório de saída
        self._setup_logging_and_output()

    def _setup_logging_and_output(self):
        """Configura logging e cria diretório de saída"""
        # Criar diretório de saída se não existir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configurar logging para arquivo e console
        self.logger = logging.getLogger('VolForecast')
        self.logger.setLevel(logging.INFO)
        
        # Formatar mensagens
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Handler para arquivo
        file_handler = logging.FileHandler(f'{self.output_dir}/volforecast.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Iniciando estratégia para {self.ticker}")

    def baixar_dados(self):
        """Baixa dados do Yahoo Finance com sistema de cache e múltiplas tentativas"""
        # Criar nome de arquivo de cache único
        cache_file = f"{self.output_dir}/{self.ticker.replace('^', '').replace('=', '')}_{self.start}_{self.end}_cache.csv"
        
        if os.path.exists(cache_file):
            self.logger.info(f"Carregando dados de cache: {cache_file}")
            try:
                self.precos = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                self.precos.columns = ['close']
                self.logger.info("Dados carregados com sucesso do cache")
                return
            except Exception as e:
                self.logger.warning(f"Erro ao carregar cache: {str(e)}. Baixando novos dados.")

        for i in range(self.tentativas):
            try:
                self.logger.info(f"Tentativa {i+1}/{self.tentativas} de baixar dados para {self.ticker}")
                dados = yf.download(
                    self.ticker,
                    start=self.start,
                    end=self.end,
                    progress=False
                )

                if not dados.empty:
                    self.precos = dados[['Close']].copy()
                    self.precos.columns = ['close']
                    self.logger.info(f"Dados baixados com sucesso para {self.ticker}")
                    
                    # Salvar em cache para uso futuro
                    self.precos.to_csv(cache_file)
                    self.logger.info(f"Dados salvos em cache: {cache_file}")
                    return
                else:
                    self.logger.warning("Dados retornaram vazios do Yahoo Finance")

            except Exception as e:
                self.logger.error(f"Erro na tentativa {i+1}: {str(e)}")

            if i < self.tentativas - 1:
                wait_time = self.intervalo_tentativas * (i + 1)  # Backoff exponencial
                self.logger.info(f"Aguardando {wait_time} segundos...")
                time.sleep(wait_time)

        raise ValueError(f"Falha ao baixar dados após {self.tentativas} tentativas")

    def calcular_sinais(self):
        """Calcula sinais de compra/venda com base nas bandas de volatilidade"""
        if self.precos is None or self.precos.empty:
            raise ValueError("Dados de preços não disponíveis")

        # Definir fator de anualização baseado na frequência
        if self.freq == "daily":
            fator_anual = 252
        elif self.freq == "weekly":
            fator_anual = 52
        elif self.freq == "monthly":
            fator_anual = 12
        else:
            fator_anual = 252  # Padrão para diário
            self.logger.warning(f"Frequência {self.freq} não reconhecida. Usando padrão diário.")

        # Calcular retornos e volatilidade
        self.precos['retorno'] = np.log(self.precos['close'] / self.precos['close'].shift(1))
        self.precos['volatilidade'] = self.precos['retorno'].rolling(self.window_vol).std() * np.sqrt(fator_anual)

        # Calcular bandas de volatilidade
        self.precos['banda_superior'] = self.precos['close'] * (1 + self.desvio_std * self.precos['volatilidade'] / np.sqrt(fator_anual))
        self.precos['banda_inferior'] = self.precos['close'] * (1 - self.desvio_std * self.precos['volatilidade'] / np.sqrt(fator_anual))

        # Gerar sinais com limpeza para evitar sinais consecutivos iguais
        self.precos['sinal'] = 0
        self.precos.loc[self.precos['close'] > self.precos['banda_superior'], 'sinal'] = -1  # Vender
        self.precos.loc[self.precos['close'] < self.precos['banda_inferior'], 'sinal'] = 1    # Comprar
        
        # Remover sinais consecutivos iguais (só gera sinal quando muda de direção)
        self.precos['sinal'] = self.precos['sinal'].replace(0, method='ffill')
        changes = self.precos['sinal'].diff().fillna(0) != 0
        self.precos['sinal'] = self.precos['sinal'].where(changes, 0)

        self.sinais = self.precos[['sinal']].copy()

    def executar_backtest(self):
        """Executa o backtest da estratégia"""
        if self.sinais is None:
            raise ValueError("Sinais não calculados")

        if self.precos.empty:
            raise ValueError("Dados de preços vazios")

        capital = self.capital_inicial
        posicao = 0
        preco_entrada = 0
        max_capital = capital
        capital_hist = [capital]
        drawdowns = [0]
        datas = [self.precos.index[0]]
        self.operacoes = []
        stop = False

        for i in range(1, len(self.precos)):
            if stop:
                break

            data = self.precos.index[i]
            preco_atual = self.precos.iloc[i]['close']
            sinal = self.sinais.iloc[i]['sinal']

            # Atualizar máximo de capital para cálculo de drawdown
            if capital > max_capital:
                max_capital = capital
                
            # Calcular drawdown atual
            drawdown = (max_capital - capital) / max_capital
            drawdowns.append(drawdown)

            # Verificar stop-loss por drawdown
            if drawdown >= self.max_drawdown:
                self.logger.warning(f"Stop-loss acionado por drawdown de {drawdown*100:.2f}% em {data}")
                stop = True
                # Fechar posição se estiver aberta
                if posicao != 0:
                    retorno = posicao * (preco_atual / preco_entrada - 1)
                    capital *= (1 + retorno)
                    self.operacoes.append({
                        'data': data,
                        'tipo': 'Stop-loss',
                        'preco': preco_atual,
                        'retorno': retorno,
                        'capital': capital,
                        'posicao': 'Long' if posicao > 0 else 'Short'
                    })
                    posicao = 0
                    preco_entrada = 0
                break

            # Fechar posição se sinal contrário ou mudança de direção
            if posicao != 0 and (sinal != np.sign(posicao) or sinal == 0):
                retorno = posicao * (preco_atual / preco_entrada - 1)
                capital *= (1 + retorno)
                self.operacoes.append({
                    'data': data,
                    'tipo': 'Fechamento',
                    'preco': preco_atual,
                    'retorno': retorno,
                    'capital': capital,
                    'posicao': 'Long' if posicao > 0 else 'Short'
                })
                posicao = 0
                preco_entrada = 0

            # Abrir nova posição se houver sinal e sem posição atual
            if posicao == 0 and sinal != 0:
                # Calcular tamanho da posição com base no capital atual
                posicao = sinal * (capital / preco_atual)
                preco_entrada = preco_atual
                self.operacoes.append({
                    'data': data,
                    'tipo': 'Compra' if sinal > 0 else 'Venda',
                    'preco': preco_atual,
                    'retorno': 0,
                    'capital': capital,
                    'posicao': 'Long' if sinal > 0 else 'Short'
                })

            # Atualizar histórico a cada dia
            capital_hist.append(capital)
            datas.append(data)

        # Fechar posição aberta no final do período
        if posicao != 0 and not stop:
            retorno = posicao * (preco_atual / preco_entrada - 1)
            capital *= (1 + retorno)
            self.operacoes.append({
                'data': self.precos.index[-1],
                'tipo': 'Fechamento Final',
                'preco': preco_atual,
                'retorno': retorno,
                'capital': capital,
                'posicao': 'Long' if posicao > 0 else 'Short'
            })
            capital_hist[-1] = capital  # Atualizar último valor

        # Criar DataFrame de resultados
        self.resultados = pd.DataFrame({
            'data': datas,
            'capital': capital_hist,
            'drawdown': drawdowns,
            'preco': self.precos['close'].reindex(datas).values
        }).set_index('data')
        
        # Calcular métricas de performance
        self._calcular_metricas()

    def _calcular_metricas(self):
        """Calcula várias métricas de performance da estratégia"""
        if self.resultados is None or self.resultados.empty:
            return
            
        # Retorno total
        retorno_total = (self.resultados['capital'].iloc[-1] / self.capital_inicial - 1)
        self.metrics['retorno_total_pct'] = retorno_total * 100
        
        # Retorno anualizado
        dias = (self.resultados.index[-1] - self.resultados.index[0]).days
        anos = dias / 365.25
        retorno_anualizado = (1 + retorno_total) ** (1/anos) - 1 if anos > 0 else 0
        self.metrics['retorno_anualizado_pct'] = retorno_anualizado * 100
        
        # Máximo drawdown
        max_drawdown = self.resultados['drawdown'].max()
        self.metrics['max_drawdown_pct'] = max_drawdown * 100
        
        # Sharpe Ratio
        retornos_diarios = self.resultados['capital'].pct_change().dropna()
        risk_free_rate = 0.0  # Taxa livre de risco
        sharpe = (retornos_diarios.mean() - risk_free_rate/252) * np.sqrt(252) / retornos_diarios.std()
        self.metrics['sharpe_ratio'] = sharpe
        
        # Sortino Ratio
        downside_returns = retornos_diarios[retornos_diarios < 0]
        sortino = (retornos_diarios.mean() - risk_free_rate/252) * np.sqrt(252) / downside_returns.std() if not downside_returns.empty else np.nan
        self.metrics['sortino_ratio'] = sortino
        
        # Win rate
        operacoes_df = pd.DataFrame(self.operacoes)
        if not operacoes_df.empty:
            operacoes_lucrativas = operacoes_df[operacoes_df['retorno'] > 0]
            win_rate = len(operacoes_lucrativas) / len(operacoes_df) if len(operacoes_df) > 0 else 0
            self.metrics['win_rate_pct'] = win_rate * 100
            self.metrics['num_operacoes'] = len(operacoes_df)
            self.metrics['retorno_medio_por_operacao_pct'] = operacoes_df['retorno'].mean() * 100
        else:
            self.metrics['win_rate_pct'] = 0
            self.metrics['num_operacoes'] = 0
            self.metrics['retorno_medio_por_operacao_pct'] = 0

    def run(self):
        """Executa todo o fluxo da estratégia"""
        self.logger.info("Iniciando execução da estratégia")
        self.baixar_dados()
        self.calcular_sinais()
        self.executar_backtest()
        self.logger.info("Estratégia executada com sucesso")
        return self.resultados, pd.DataFrame(self.operacoes)

    def summary(self):
        """Exibe resumo estatístico da estratégia"""
        if self.resultados is None:
            raise ValueError("Backtest não executado")

        # Criar relatório detalhado
        report = f"""
        {'='*60}
        {'RESUMO DA ESTRATÉGIA':^60}
        {'='*60}
        {'Ativo:':<25}{self.ticker:>35}
        {'Período:':<25}{self.resultados.index[0].date()} a {self.resultados.index[-1].date():>35}
        {'Frequência:':<25}{self.freq:>35}
        {'Capital Inicial:':<25}R${self.capital_inicial:>34,.2f}
        {'Capital Final:':<25}R${self.resultados['capital'].iloc[-1]:>34,.2f}
        {'Retorno Total:':<25}{self.metrics['retorno_total_pct']:>34.2f}%
        {'Retorno Anualizado:':<25}{self.metrics['retorno_anualizado_pct']:>34.2f}%
        {'Máximo Drawdown:':<25}{self.metrics['max_drawdown_pct']:>34.2f}%
        {'Número de Operações:':<25}{self.metrics['num_operacoes']:>34}
        {'Win Rate:':<25}{self.metrics['win_rate_pct']:>34.2f}%
        {'Retorno Médio por Operação:':<25}{self.metrics['retorno_medio_por_operacao_pct']:>34.2f}%
        {'Sharpe Ratio:':<25}{self.metrics['sharpe_ratio']:>34.4f}
        {'Sortino Ratio:':<25}{self.metrics['sortino_ratio']:>34.4f}
        {'='*60}
        """
        print(report)
        return report

    def plot(self, save_fig=True):
        """Gera gráficos de performance e salva se necessário"""
        if self.resultados is None or self.precos is None:
            raise ValueError("Dados não disponíveis para plotagem")

        # Criar figura com subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'Análise da Estratégia VolForecast - {self.ticker}', fontsize=16)

        # Gráfico 1: Preços e bandas
        ax1.plot(self.precos['close'], label='Preço', color='black', linewidth=1.5)
        ax1.plot(self.precos['banda_superior'], 'r--', alpha=0.7, label='Banda Superior')
        ax1.plot(self.precos['banda_inferior'], 'g--', alpha=0.7, label='Banda Inferior')
        
        # Adicionar operações
        operacoes_df = pd.DataFrame(self.operacoes)
        if not operacoes_df.empty:
            compras = operacoes_df[operacoes_df['tipo'] == 'Compra']
            vendas = operacoes_df[operacoes_df['tipo'] == 'Venda']
            fechamentos = operacoes_df[operacoes_df['tipo'].isin(['Fechamento', 'Fechamento Final', 'Stop-loss'])]
            
            ax1.scatter(compras['data'], compras['preco'], color='green', marker='^', s=100, label='Compras', zorder=5)
            ax1.scatter(vendas['data'], vendas['preco'], color='red', marker='v', s=100, label='Vendas', zorder=5)
            ax1.scatter(fechamentos['data'], fechamentos['preco'], color='blue', marker='o', s=80, label='Fechamentos', zorder=5)
        
        ax1.set_title('Preço e Sinais')
        ax1.set_ylabel('Preço')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.fill_between(self.precos.index, 
                         self.precos['banda_inferior'], 
                         self.precos['banda_superior'], 
                         color='gray', alpha=0.2)

        # Gráfico 2: Evolução do capital
        ax2.plot(self.resultados['capital'], 'b-', label='Capital', linewidth=2)
        ax2.axhline(y=self.capital_inicial, color='black', linestyle='--', label='Capital Inicial')
        ax2.set_title('Evolução do Capital')
        ax2.set_ylabel('Capital (R$)')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.fill_between(self.resultados.index, 
                         self.resultados['capital'], 
                         self.capital_inicial, 
                         where=(self.resultados['capital'] > self.capital_inicial), 
                         facecolor='green', alpha=0.3)
        ax2.fill_between(self.resultados.index, 
                         self.resultados['capital'], 
                         self.capital_inicial, 
                         where=(self.resultados['capital'] < self.capital_inicial), 
                         facecolor='red', alpha=0.3)

        # Gráfico 3: Drawdown
        ax3.plot(self.resultados['drawdown'] * 100, 'r-', label='Drawdown')
        ax3.axhline(y=self.max_drawdown * 100, color='black', linestyle='--', label='Limite Drawdown')
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Data')
        ax3.legend(loc='best')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.fill_between(self.resultados.index, 
                         self.resultados['drawdown'] * 100, 
                         color='red', alpha=0.3)

        # Ajustar layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Salvar figura se necessário
        if save_fig:
            fig_path = f"{self.output_dir}/volforecast_performance_{self.ticker}.png"
            plt.savefig(fig_path, dpi=300)
            self.logger.info(f"Gráfico salvo em: {fig_path}")
        
        plt.show()

    def save_results(self):
        """Salva resultados e operações em arquivos CSV"""
        if self.resultados is not None:
            result_path = f"{self.output_dir}/volforecast_results_{self.ticker}.csv"
            self.resultados.to_csv(result_path)
            self.logger.info(f"Resultados salvos em: {result_path}")
        
        if self.operacoes:
            oper_path = f"{self.output_dir}/volforecast_operacoes_{self.ticker}.csv"
            pd.DataFrame(self.operacoes).to_csv(oper_path, index=False)
            self.logger.info(f"Operações salvas em: {oper_path}")
        
        if self.metrics:
            metrics_path = f"{self.output_dir}/volforecast_metrics_{self.ticker}.txt"
            with open(metrics_path, 'w') as f:
                f.write(self.summary())
            self.logger.info(f"Métricas salvas em: {metrics_path}")

    def optimize_threshold(self, min_val=1.0, max_val=3.0, step=0.1):
        """Otimiza o parâmetro de threshold através de busca em grid"""
        self.logger.info("Iniciando otimização de threshold")
        
        best_threshold = None
        best_sharpe = -np.inf
        results = []
        
        # Testar diferentes valores de threshold
        for threshold in np.arange(min_val, max_val + step, step):
            try:
                self.logger.info(f"Testando threshold = {threshold:.2f}")
                self.threshold = threshold
                self.desvio_std = threshold
                
                # Recalcular sinais e backtest
                self.calcular_sinais()
                self.executar_backtest()
                
                # Registrar resultados
                results.append({
                    'threshold': threshold,
                    'retorno_total': self.metrics['retorno_total_pct'],
                    'max_drawdown': self.metrics['max_drawdown_pct'],
                    'sharpe': self.metrics['sharpe_ratio'],
                    'sortino': self.metrics['sortino_ratio'],
                    'num_operacoes': self.metrics['num_operacoes']
                })
                
                # Atualizar melhor Sharpe
                if self.metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = self.metrics['sharpe_ratio']
                    best_threshold = threshold
                    
            except Exception as e:
                self.logger.error(f"Erro ao testar threshold {threshold}: {str(e)}")
        
        # Salvar resultados da otimização
        opt_df = pd.DataFrame(results)
        opt_path = f"{self.output_dir}/volforecast_optimization_{self.ticker}.csv"
        opt_df.to_csv(opt_path, index=False)
        self.logger.info(f"Resultados da otimização salvos em: {opt_path}")
        
        self.logger.info(f"Melhor threshold encontrado: {best_threshold:.2f} com Sharpe: {best_sharpe:.4f}")
        
        # Restaurar melhor parâmetro
        self.threshold = best_threshold
        self.desvio_std = best_threshold
        self.calcular_sinais()
        self.executar_backtest()
        
        return opt_df