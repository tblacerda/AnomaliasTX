# Análise Estatística de Anomalias em PLOSS, Jitter e Latência

Este projeto realiza a análise estatística de anomalias em dados de desempenho de rede (PLOSS, Jitter, Latência) utilizando métodos como Isolation Forest, DBSCAN, e SHAP para explicabilidade. O objetivo é identificar, classificar e explicar anomalias em sites de telecomunicações, facilitando a tomada de decisão e o monitoramento da qualidade de serviço.

## Principais Funcionalidades
- **Detecção de Anomalias**: Utiliza Isolation Forest para identificar comportamentos anômalos em séries temporais de indicadores de rede.
- **Explicabilidade**: Integra SHAP para explicar as decisões do modelo de detecção de anomalias.
- **Clusterização**: Agrupa sites com comportamento semelhante usando DBSCAN.
- **Cálculo de Medianas**: Compara períodos recentes e históricos para avaliar variações.
- **Geração de Relatórios**: Exporta resultados em Excel e gera gráficos para visualização.
- **Automação de ETL**: Leitura, limpeza e atualização automática dos dados a partir de planilhas e bancos de dados.

## Estrutura dos Principais Arquivos
- `MULTIVARIADO.py`: Pipeline principal de detecção e análise de anomalias.
- `MULTIVARIADO_SHAP.py`: Versão com explicabilidade SHAP integrada.
- `ler_bd.py`, `criar_bd.py`: Scripts utilitários para manipulação do banco de dados.
- `bd/`: Scripts SQL, banco de dados SQLite e arquivos auxiliares.
- `entrada/`: Planilhas de entrada.
- `SAIDA/`: Resultados e relatórios gerados.

## Como Executar
1. **Pré-requisitos**:
   - Python 3.8+
   - Instale as dependências:
     ```bash
     pip install -r requirements.txt
     ```
   - (Opcional) Instale o R e configure o caminho se for usar scripts R.
2. **Configuração**:
   - Edite o arquivo `settings.ini` com as credenciais e URLs necessárias.
3. **Execução**:
   - Execute o pipeline principal:
     ```bash
     python MULTIVARIADO.py
     ```
   - Para análise com SHAP:
     ```bash
     python MULTIVARIADO_SHAP.py
     ```

## Principais Parâmetros
- `__DIAS_AMOSTRA__`: Número de dias recentes para análise.
- `__AVAIL__`: Threshold de disponibilidade mínima.
- `_THRESHOLD_ploss_`, `_THRESHOLD_jitter`, `_THRESHOLD_latency`: Thresholds para variação dos indicadores.
- `_ANOMALY_FILTER_`: Filtro para considerar anomalias relevantes.

## Resultados
- Relatórios em Excel com ranking de sites, clusters, e explicações das anomalias.
- Gráficos de barras para visualização de aumentos/reduções por ANF.
- Logs detalhados em `last_execution.log`.

## Observações
- O projeto utiliza paralelização para acelerar a análise em grandes volumes de dados.
- O arquivo `.gitignore` já está configurado para ignorar arquivos temporários, de dados e de saída.

## Licença
Projeto interno para análise de desempenho de rede. Uso restrito.
