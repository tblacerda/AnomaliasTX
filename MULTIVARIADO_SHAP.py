"""
This module performs anomaly detection on network performance data using Isolation Forest and other statistical methods.
Classes:
    AnomalyDetection: A class for detecting anomalies in network performance data.
Functions:
    AppendBb(_LOCAL_=True): Reads new data from a spreadsheet and appends it to the database.
    LerBd(__AVAIL__): Reads data from the database and filters it based on availability.
    IncluirMunUfAnf(df): Reads basic data to complement the output.
    Clusterizacao(df): Performs clustering using DBSCAN.
    CalculoMedianas(df, N=__DIAS_AMOSTRA__): Calculates medians for recent and historical data.
    avaliar_variacao(df, threshold_ploss, threshold_jitter, threshold_latency, anomaly_filter): Evaluates variations in network performance metrics.
    plot_aumento_e_reducao_por_anf(df): Plots the increase and decrease in performance metrics by ANF.
    main(): Main function to run the daily analysis and generate the output.
Constants:
    __DB_PLOSS__: Path to the database file.
    __TODAY__: Current date.
    __DIR_SAIDA__: Output directory.
    __ARQUIVO_SAIDA__: Output file name.
    __DIAS_MAX__: Maximum number of days considered.
    __DIAS_MIN__: Minimum number of days considered.
    __DIAS_AMOSTRA__: Number of recent days to be evaluated.
    __AVAIL__: Availability threshold.
    D_BD: Date from which to read data from the database.
    _LOCAL_: Flag to indicate if the data is local.
    _URL_: URL to download the data.
    _ANOMALY_FILTER_: Anomaly filter threshold.
    _THRESHOLD_ploss_: Packet loss threshold.
    _THRESHOLD_jitter: Jitter threshold.
    _THRESHOLD_latency: Latency threshold.
"""
import time
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import seaborn as sns
from datetime import timedelta, date
from tqdm import tqdm
import warnings
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import requests
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
import logging
import shap
# Configure logging to overwrite the log file each time
start_time = time.time()
logging.basicConfig(filename='last_execution.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                     filemode='w')

warnings.filterwarnings("ignore")
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
# Total de dias para ler do BD
#__TODAY__ = pd.to_datetime('2024-09-21') 
__DB_PLOSS__ = 'bd/PLOSS.banco'
__TODAY__ = date.today()
__DIR_SAIDA__ = 'SAIDA/'
__ARQUIVO_SAIDA__ = 'analise_estatistica.xlsx'   ######
__DIAS_MAX__ = 45   # Populacao maxima considerada
__DIAS_MIN__ = 30   # Populacao minima considerada
__DIAS_AMOSTRA__ = 5 # ao menos __DIAS_AMOSTRA__ / 2. Amostra de dias recentes a ser avaliada 
__AVAIL__ = 0.7  # Considera apenas sites com 100% de amostras
# Data a partir da qual é para se ler do Banco de dados
D_BD = pd.to_datetime(__TODAY__ - timedelta(days=__DIAS_MAX__ + __DIAS_AMOSTRA__)).strftime("%Y-%m-%d")
_LOCAL_ = False
_ANOMALY_FILTER_ = -0.05
_THRESHOLD_ploss_ = 0.01
_THRESHOLD_jitter = 2 # ms
_THRESHOLD_latency = 20 #ms
# elif CURR_DIR.split("\\")[-1] == 'Scripts':
#     os.chdir("../")
# set execution directory the same as the script directory
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv('settings.ini'))
_URL_ = config('URL')
user_oracle = config('user_oracle')
pass_oracle = config('pass_oracle')


class AnomalyDetection:
    # https://medium.com/@limyenwee_19946/unsupervised-outlier-detection-with-isolation-forest-eab398c593b2
    def __init__(self, N=__DIAS_AMOSTRA__):
        """
        Initializes the MULTIVARIADO class.

        Parameters:
        N (int): Number of recent days for comparison. Default is __DIAS_AMOSTRA__.

        Attributes:
        N (int): Stores the number of recent days for comparison.
        model (IsolationForest): Model to detect anomalies.
        transformer (PowerTransformer): Transformer to apply Yeo-Johnson transformation and standardization.
        """
        self.N = N  # Number of recent days for comparison
        self.model = IsolationForest(contamination='auto', random_state= 1, n_estimators= 250)  # Model to detect anomalies
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    
    def apply_transformations(self, df):
        """
        Apply Box-Cox or Yeo-Johnson transformation to the 'jitter', 'ploss', and 'latency' columns for each unique site in the DataFrame.
        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be transformed. 
                       It must have columns 'Site', 'jitter', 'ploss', and 'latency'.
        Returns:
        pandas.DataFrame: The DataFrame with transformed 'jitter', 'ploss', and 'latency' columns for each site.
        Notes:
        - The transformation is applied separately for each unique site in the 'Site' column.
        - A small value (0.001) is added to the data to ensure all values are positive before applying the transformation.
        - If the Box-Cox transformation fails for a site, a warning is issued, and the original data for that site is retained.
        """
        # Apply Box-Cox OR Yeo-johnson transformation for each site
        sites = df['Site'].unique()
        for site in tqdm(sites, desc="Applying transformations"):
            site_data = df[df['Site'] == site][['jitter', 'ploss', 'latency']]
            
            # Add a small value to ensure data is positive
            site_data_adj = site_data + 0.001

            try:
                # Apply Box-Cox transformation
                df.loc[df['Site'] == site, ['jitter', 'ploss', 'latency']] = self.transformer.fit_transform(site_data_adj)
            except Exception as e:
                warnings.warn(f"Box-Cox transformation failed for site {site}: {e}")
                # Skip the transformation for this site and keep original data
                df.loc[df['Site'] == site, ['jitter', 'ploss', 'latency']] = site_data
        return df
    
    def detect_anomalies_for_site(self, site_data):
        """
        Detects anomalies for a given site based on recent and historical data.
        Parameters:
        site_data (DataFrame): A pandas DataFrame containing the site's data with columns 'jitter', 'ploss', 'latency', and 'Site'.
        Returns:
        float: The average anomaly score for the recent data. Returns 0 if there is insufficient data.
        Raises:
        UserWarning: If there is insufficient data for the site or insufficient training data.
        """
        # Check if the data is sufficient
        if len(site_data) <= self.N:
            warnings.warn(f"Insufficient data for site: {site_data['Site'].iloc[0]}")
            return 0
        
        recent_days = site_data.tail(self.N)
        historical_data = site_data.iloc[:-self.N]

        # Train model on historical data
        X_train = historical_data[['jitter', 'ploss', 'latency']].values
        if len(X_train) < 1:
            warnings.warn(f"Insufficient training data for site: {site_data['Site'].iloc[0]}")
            return 0

        # Fit model and calculate anomaly scores for the recent data
        self.model.fit(X_train)
        X_recent = recent_days[['jitter', 'ploss', 'latency']].values
        time_weights = np.linspace(1.0, 2.0, len(X_recent)) 
        anomaly_scores = self.model.decision_function(X_recent)  # Get anomaly scores for recent data
        anomaly_scores *= time_weights

        # SHAP explanation
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_recent)

        # Visualize SHAP values for the first recent data point
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0], X_recent[0], feature_names=['jitter', 'ploss', 'latency'])
        anomaly_scores = self.model.decision_function(X_recent)  # Get anomaly scores for recent data # tiago
        anomaly_scores *= time_weights

        # Create a result array for the entire data, with zeros for historical data
        #result = np.zeros(len(site_data))
        #result[-self.N:] = anomaly_scores  # Assign scores to the last N entries (recent data)
        
        # Return average anomaly score for the recent data
        avg_score = anomaly_scores.mean()
        return avg_score

    def detect_anomalies(self, df):
        """
        Detect anomalies in the given DataFrame.
        This method processes each unique site in the DataFrame to detect anomalies
        and assigns an anomaly score to each site. It also ranks the sites based on
        their average anomaly scores.
        Args:
            df (pd.DataFrame): The input DataFrame containing site data. It must have
                               a column named 'Site' which contains site identifiers.
        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The DataFrame with an additional column 'Anomaly' 
                                indicating the anomaly score for each site.
                - list: A list of tuples where each tuple contains a site identifier 
                        and its average anomaly score, sorted by the anomaly score 
                        in ascending order (lower scores are worse).
        """
        # Process each site to detect anomalies
        site_ranking = []
        sites = df['Site'].unique()
        for site in tqdm(sites, desc="Detecting anomalies"):
            site_data = df[df['Site'] == site]
            avg_score = self.detect_anomalies_for_site(site_data)
            df.loc[df['Site'] == site, 'Anomaly'] = avg_score
            # Append site and its average score for ranking
            site_ranking.append((site, avg_score))
        
        # Sort the ranking by the average anomaly score (lower scores are worse)
        site_ranking = sorted(site_ranking, key=lambda x: x[1])
        return df, site_ranking
    
    def run_daily_analysis(self, df):
        """
        Perform daily analysis on the provided DataFrame.
        This method sorts the DataFrame by 'Dia', applies necessary transformations,
        detects anomalies, and ranks sites based on their anomaly scores.
        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be analyzed.
        Returns:
        pandas.DataFrame: A DataFrame containing the median anomaly scores for each site,
                          with the most recent 'Dia' value.
        """
        # Sort the DataFrame by 'Dia'
        df = df.sort_values(by='Dia')

        # Apply transformations
        df_transformed = self.apply_transformations(df)
        # Detect anomalies and get site rankings
    #    anomalies, site_ranking = self.detect_anomalies(df_transformed)
        anomalies, _ = self.detect_anomalies(df_transformed)
       
        # Filter anomalies to include only those <= -0.05
        #anomalies = anomalies[anomalies['Anomaly'] <= _ANOMALY_FILTER_]

        # Rank sites based on their anomaly scores (lowest score gets rank 1)
        anomalies = anomalies.sort_values(by='Anomaly')
        #anomalies['Rank'] = anomalies.groupby('Dia').cumcount() + 1

        # Add the rank column to the anomalies DataFrame

       # anomalies = anomalies[['Dia', 'Site', 'Anomaly', 'Rank']]
        anomalies = anomalies[['Dia', 'Site', 'Anomaly']]

        anomalies = anomalies.groupby('Site').tail(__DIAS_AMOSTRA__)
        anomalies = anomalies.groupby('Site').median().reset_index()
        anomalies['Dia'] = df_transformed['Dia'].max()
        return anomalies

def AppendBb(_LOCAL_ = True):
    '''
    Ler dados novos de uma planilha e anexar no BD
    '''
    logging.info('Starting AppendBb function')
    def baixar_csv_para_dataframe(url = _URL_):
        try:
            print('Conectando ao MicroStrategy...')
            # Envia uma requisição HTTP GET para o URL
            url = _URL_
            resposta = requests.get(url)
            
            # Verifica se a requisição foi bem-sucedida (código 200)
            if resposta.status_code == 200:
                print("Response Code: ", resposta.status_code)
                # Converte o conteúdo da resposta em string e usa StringIO para ler como CSV
                csv_string = StringIO(resposta.content.decode('utf-16le'))
                
                # Carrega o CSV no pandas dataframe com o delimitador correto
                df = pd.read_csv(csv_string, delimiter=';', skip_blank_lines=True, encoding='UTF-16 LE' )
                df.dropna(how='all', inplace=True)
                logging.info('Finished AppendBb function')
                return df
            else:
                print(f"Falha ao baixar o arquivo. Código de status: {resposta.status_code}")
                logging.info('Finished AppendBb function with error')
                return None
        
        except Exception as e:
            print(f"Ocorreu um erro ao baixar e ler o arquivo: {e}")
            logging.info('Finished AppendBb function with error')
            return print("Leitura ok")


    def UltimaDataBanco(NomeDoBanco=__DB_PLOSS__):
        '''Resgata a última data no banco de dados'''

        NomeDoBanco = __DB_PLOSS__
        conn = None
        try:
            conn = sqlite3.connect(NomeDoBanco)
            cur = conn.cursor()
            cur.execute("SELECT MAX(Dia) FROM PLOSS")
            UltimaData = cur.fetchall()
            UltimaData = UltimaData[0][0]
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

        return UltimaData

    if _LOCAL_:
        PATH = r".\entrada"
        def p2f(x):
            x = x.replace(',', '.')
            if x.find('(') > -1:
                x = '0'
            return float(x.strip('%'))/100

        all_files = glob.glob(PATH + "\*.xls*")
        li = []
        all_files

        for filename in all_files:
            print(filename)
            temp = pd.read_excel(filename, skiprows=2)
            li.append(temp)        
        df = pd.concat(li, axis=0, ignore_index=True)

    else:       
        df = baixar_csv_para_dataframe()
        #df = pd.read_csv(r'apipbi.csv', delimiter=';', skip_blank_lines=True, encoding='UTF-16 LE')
        df['Availability Avg'] = pd.to_numeric(df['Availability Avg'].str.replace('%', '').str.replace(',', '.'), errors='coerce')
        df['Packet Loss Avg'] = pd.to_numeric(df['Packet Loss Avg'].str.replace('%', '').str.replace(',', '.'), errors='coerce')
        df['Packet Loss P95 Max'] = pd.to_numeric(df['Packet Loss P95 Max'].str.replace('%', '').str.replace(',', '.'), errors='coerce')
        df['Availability Avg'] = df['Availability Avg'] / 100
        df['Packet Loss Avg'] = df['Packet Loss Avg'] / 100
        df['Packet Loss P95 Max'] = df['Packet Loss P95 Max'] / 100

    df.rename(columns = {'BTS/NodeB/EnodeB':'BTS/Nodeb/Enodeb',
                        'Regional NodeB':'Regional Nodeb',
                        'Estado NodeB': 'Estado Nodeb',
                        'ANF NodeB': 'ANF Nodeb',
                        'Cidade NodeB':'Cidade Nodeb',
                        }, inplace = True)
    df.drop_duplicates(inplace=True, subset=['Dia', 'BTS/Nodeb/Enodeb'])
    
    df = df.loc[df['Availability Avg'] >= __AVAIL__]
    #df = df.loc[(df['Regional Nodeb'] == "TNE")]
    df['Dia'] = pd.to_datetime(df['Dia'], format = "%d/%m/%Y")
    UltimoDia = UltimaDataBanco(__DB_PLOSS__)
    UltimoDia = pd.to_datetime(pd.to_datetime(UltimoDia) + timedelta(days=1))
    df = df.loc[df['Dia'] >= UltimoDia]
    # Dias = df['Dia'].unique()
    conn = sqlite3.connect(__DB_PLOSS__)
    df.to_sql(name='PLOSS', con=conn, if_exists='append', index=False)
    conn.close()

    return np.nan

def LerBd(__AVAIL__):
    logging.info('Starting LerBd function')
    conn = sqlite3.connect(__DB_PLOSS__)
    queryString = """SELECT Dia,
                "BTS/Nodeb/Enodeb", "Packet Loss Avg", "Availability Avg", "Jitter Avg", "Latency Avg"
                FROM PLOSS WHERE "Regional Nodeb" IN ("TNE")  AND Dia >= ? order by Dia"""
    # queryString = """SELECT * FROM PLOSS"""

    dfPloss = pd.read_sql(queryString, con=conn, params=[D_BD])
    conn.close()
    dfPloss = dfPloss.loc[dfPloss['Availability Avg'] >= __AVAIL__]
    #dfPloss.drop("Availability Avg", axis=1, inplace=True)
    dfPloss.rename({'Station ID' : 'END_ID',
                    'BTS/Nodeb/Enodeb' : 'Site',
                    'Availability Avg': 'avail',
                    'Packet Loss Avg' : 'ploss',
                    'Jitter Avg': 'jitter',
                    'Latency Avg': 'latency'},
                    axis=1, inplace=True)

    dfPloss.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfPloss.drop_duplicates(inplace=True, subset=['Dia', 'Site'])
    dfPloss['Dia'] = pd.to_datetime(dfPloss['Dia'])
    dfPloss['avail'] = pd.to_numeric(dfPloss['avail'], errors='coerce')
    dfPloss['ploss'] = pd.to_numeric(dfPloss['ploss'], errors='coerce')
    dfPloss['jitter'] = pd.to_numeric(dfPloss['jitter'], errors='coerce')
    dfPloss['latency'] = pd.to_numeric(dfPloss['latency'], errors='coerce')
    dfPloss.dropna(inplace=True)
    dfPloss = dfPloss.sort_values(by=['Site', 'Dia'])
    dfPloss = dfPloss.reset_index(drop=True)
    dfPloss.drop_duplicates(inplace=True)
    logging.info('Finished LerBd function')

    return dfPloss

def IncluirMunUfAnf(df):
    '''
        Ler os dados basicos para poder complementar na saida
    '''

    conn = sqlite3.connect(__DB_PLOSS__)
    queryString = """SELECT "BTS/Nodeb/Enodeb", "Estado Nodeb",
                    "ANF Nodeb", "Cidade Nodeb"
                FROM PLOSS WHERE "Regional Nodeb" IN ("TNE")"""

    dfDados = pd.read_sql(queryString, con=conn)
    dfDados.rename({'BTS/Nodeb/Enodeb' : 'Site',
                    'Estado Nodeb' : 'UF',
                    'ANF Nodeb' : 'ANF',
                    'Cidade Nodeb' : 'Cidade'
                    },
                    axis=1, inplace=True)
    conn.close()
    dfDados.drop_duplicates(inplace=True)

    dfMerged = pd.merge(df, dfDados, on='Site', how='left')

    last_3_columns = dfMerged.iloc[:, -3:]
    dfMerged = dfMerged.iloc[:, :-3]
    dfMerged = pd.concat([dfMerged.iloc[:, :2],
                        last_3_columns,
                        dfMerged.iloc[:, 2:]], axis=1)

    return dfMerged

def Clusterizacao(df):
    '''Clusterizacao utilizando dbScan
    '''
    from sklearn.cluster import DBSCAN
    
    df_temp = df.copy(deep = True)
    dfEndId = pd.read_excel('bd\ENDID_SITEID.xlsx') # manter atualizado
    df = pd.merge(df, dfEndId, how='inner', on='Site')

    df=df.loc[df['ploss']=='aumentou']

    variables = ['Anomaly','ploss_recente', 'avail_recente',
       'jitter_recente', 'latency_recente', 'ploss_historico',
       'avail_historico', 'jitter_historico', 'latency_historico',
        'lat', 'lon' ]

    X = df[variables]
    X.fillna(inplace=True, method='ffill') #melhorar isso aqui
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=3, min_samples=2)
    clusters = dbscan.fit_predict(X)

    # Add cluster labels as a new column in the original DataFrame
    df['Cluster'] = clusters
    df['Cluster'] = df['Cluster'].apply(lambda x: f"C{x}")
    df['Cluster'] = df['Cluster'].replace('C-1', 'Sem cluster')
    df = df[['Site', 'Cluster']]
    df_temp = pd.merge(df_temp, df, how='left', on='Site')
    return df_temp

def CalculoMedianas(df, N=  __DIAS_AMOSTRA__):
    # Exemplo de DataFrame
    # df = seu DataFrame

    # Converter a coluna 'Dia' para o tipo datetime se necessário
    df['Dia'] = pd.to_datetime(df['Dia'])

    # Obter o último dia disponível no DataFrame
    last_day = df['Dia'].max()

    # Separar o DataFrame em dois: um com os últimos "N" dias e outro com os demais
    df_recent = df[df['Dia'] >= last_day - pd.Timedelta(days=N)]
    df_historic = df[df['Dia'] < last_day - pd.Timedelta(days=N)]
    df_recent = df_recent.drop(['Dia'], axis = 1)
    df_historic = df_historic.drop(['Dia'], axis = 1)

    # Agrupar os dois DataFrames por 'Site' e calcular a mediana
    df_recent_grouped = df_recent.groupby('Site').median().add_suffix('_recente').reset_index()
    df_historic_grouped = df_historic.groupby('Site').median().add_suffix('_historico').reset_index()

    # Juntar os dois DataFrames em um único, usando 'Site' como chave
    df_combined = pd.merge(df_recent_grouped, df_historic_grouped, on='Site', how='outer')
    return df_combined

def avaliar_variacao(df, threshold_ploss, threshold_jitter, threshold_latency, anomaly_filter):
    def avaliar(row):
        # Avaliar ploss
        if row['ploss_recente'] > row['ploss_historico'] + threshold_ploss:
            ploss = 'aumentou'
        elif row['ploss_recente'] < row['ploss_historico'] - threshold_ploss:
            ploss = 'diminuiu'
        else:
            ploss = 'constante'
        
        # Avaliar jitter
        if row['jitter_recente'] > row['jitter_historico'] + threshold_jitter:
            jitter = 'aumentou'
        elif row['jitter_recente'] < row['jitter_historico'] - threshold_jitter:
            jitter = 'diminuiu'
        else:
            jitter = 'constante'
        
        # Avaliar latencia
        if row['latency_recente'] > row['latency_historico'] + threshold_latency:
            latencia = 'aumentou'
        elif row['latency_recente'] < row['latency_historico'] - threshold_latency:
            latencia = 'diminuiu'
        else:
            latencia = 'constante'
        #if latencia == 'constante' and jitter == 'constante' and ploss =='constante':
        
        return pd.Series([ploss, jitter, latencia])
    
    # Aplicar a função de avaliação para cada linha do DataFrame
    df[['ploss', 'jitter', 'latencia']] = df.apply(avaliar, axis=1)

    df = df.sort_values(by='Anomaly')
    df['Rank'] = range(1, len(df) + 1)
    #df.loc[df['Anomaly'] > anomaly_filter, 'Rank'] = None

    df = df[['Site', 'Dia', 'UF', 'ANF', 'Cidade', 'Anomaly', 'Rank',
    'ploss', 'jitter', 'latencia','ploss_recente', 'avail_recente',
    'jitter_recente', 'latency_recente', 'ploss_historico',
    'avail_historico', 'jitter_historico', 'latency_historico']]
    
    columns_to_round = [
    'ploss', 'jitter', 'latencia', 'ploss_recente', 'avail_recente',
    'jitter_recente', 'latency_recente', 'ploss_historico',
    'avail_historico', 'jitter_historico', 'latency_historico']

    df.loc[(df['ploss'] == 'constante') &
        (df['jitter'] == 'constante') &
        (df['latencia'] == 'constante'), 'Anomaly'] = 0
    df[columns_to_round] = df[columns_to_round].round(2)
    df = df.sort_values(by='Anomaly')
    df['Rank'] = range(1, len(df) + 1)
    df.loc[df['Anomaly'] > anomaly_filter, 'Rank'] = None
    df.loc[df['Rank'].isna(), 'ploss'] = 'constante'
    df.loc[df['Rank'].isna(), 'latencia'] = 'constante'
    df.loc[df['Rank'].isna(), 'jitter'] = 'constante'
    return df

def plot_aumento_e_reducao_por_anf(df):
    # Filtrar linhas com aumento de PLOSS, Jitter e Latencia
    aumento_ploss = df[df['ploss'] == 'aumentou']
    aumento_jitter = df[df['jitter'] == 'aumentou']
    aumento_latencia = df[df['latencia'] == 'aumentou']
    
    # Filtrar linhas com redução de PLOSS, Jitter e Latencia
    reducao_ploss = df[df['ploss'] == 'diminuiu']
    reducao_jitter = df[df['jitter'] == 'diminuiu']
    reducao_latencia = df[df['latencia'] == 'diminuiu']
    
    # Contar a quantidade de linhas por ANF para aumento
    contagem_aumento_ploss = aumento_ploss['ANF'].value_counts()
    contagem_aumento_jitter = aumento_jitter['ANF'].value_counts()
    contagem_aumento_latencia = aumento_latencia['ANF'].value_counts()
    
    # Contar a quantidade de linhas por ANF para redução
    contagem_reducao_ploss = reducao_ploss['ANF'].value_counts()
    contagem_reducao_jitter = reducao_jitter['ANF'].value_counts()
    contagem_reducao_latencia = reducao_latencia['ANF'].value_counts()
    
    # Criar DataFrame com as contagens para aumento
    contagens_aumento = pd.DataFrame({
        'PLOSS': contagem_aumento_ploss,
        'JITTER': contagem_aumento_jitter,
        'LATENCIA': contagem_aumento_latencia
    }).fillna(0)
    
    # Criar DataFrame com as contagens para redução
    contagens_reducao = pd.DataFrame({
        'PLOSS': contagem_reducao_ploss,
        'JITTER': contagem_reducao_jitter,
        'LATENCIA': contagem_reducao_latencia
    }).fillna(0)
    
    # Plotar gráfico de barras para aumento
    contagens_aumento.plot(kind='bar', figsize=(10, 6))
    plt.title('Quantidade de Sites com Aumento de PLOSS, JITTER e LATENCIA por ANF')
    plt.xlabel('ANF')
    plt.ylabel('Quantidade')
    plt.show()
    
    # Plotar gráfico de barras para redução
    contagens_reducao.plot(kind='bar', figsize=(10, 6))
    plt.title('Quantidade de Sites com Redução de PLOSS, JITTER e LATENCIA por ANF')
    plt.xlabel('ANF')
    plt.ylabel('Quantidade')
    plt.show()

def process_chunk(chunk):
    logging.info('Processing a chunk')
    anomaly_detector = AnomalyDetection(N=__DIAS_AMOSTRA__)
    anomalies = anomaly_detector.run_daily_analysis(chunk)
    # dfMedianas = CalculoMedianas(chunk)
    # df_combined = pd.merge(anomalies, dfMedianas, on='Site', how='left')
    # df_combined = IncluirMunUfAnf(df_combined)
    # df_saida = avaliar_variacao(df_combined, _THRESHOLD_ploss_, _THRESHOLD_jitter, _THRESHOLD_latency, _ANOMALY_FILTER_)
    logging.info('Finished processing a chunk')
    return anomalies

def split_data_into_chunks(dfConsolidado, num_chunks):
    """
    Split the DataFrame into chunks based on unique 'Site' values.

    Parameters:
    dfConsolidado (pd.DataFrame): The input DataFrame to be split.
    num_chunks (int): The number of chunks to split the DataFrame into.

    Returns:
    list: A list of DataFrame chunks.
    """
    # Step 1: Get unique 'Site' values
    unique_sites = dfConsolidado['Site'].unique()
    
    # Step 2: Split unique 'Site' values into chunks
    site_chunks = np.array_split(unique_sites, num_chunks)
    
    # Step 3: Create DataFrame chunks based on 'Site' chunks
    chunks = [dfConsolidado[dfConsolidado['Site'].isin(sites)] for sites in site_chunks]
    
    return chunks

def main():
    __RODAR_PARALELO__ = True
    logging.info('Starting main function')
    AppendBb(_LOCAL_)
    start_time = time.time()
    dfConsolidado = LerBd(__AVAIL__)
    logging.info('Data loaded and filtered')
    if __RODAR_PARALELO__:
        logging.info('Running in Parallel mode')
            # Split the DataFrame into chunks
        num_chunks = 4  # Adjust based on your CPU cores
        chunks = split_data_into_chunks(dfConsolidado, num_chunks)
        logging.info(f'Split data into {num_chunks} chunks')
        with ProcessPoolExecutor() as executor:
            print("Executing in parallel...")
            logging.info('Processing all chunks...')
            results = list(executor.map(process_chunk, chunks))
            logging.info('Finished processing all chunks')
            #Combine results from all chunks
            anomalies = pd.concat(results, ignore_index=True)
            logging.info('Combined results from all chunks')
            # Use tqdm to monitor the execution
    else:
        logging.info('Running in single mode')
        # Running the analysis
        anomaly_detector = AnomalyDetection(N=__DIAS_AMOSTRA__)
        anomalies = anomaly_detector.run_daily_analysis(dfConsolidado)
       
    dfMedianas = CalculoMedianas(dfConsolidado)
    df_combined = pd.merge(anomalies, dfMedianas, on='Site', how='left')
    df_combined = IncluirMunUfAnf(df_combined)
    df_saida = avaliar_variacao(df_combined,_THRESHOLD_ploss_ , _THRESHOLD_jitter, _THRESHOLD_latency, _ANOMALY_FILTER_)
    df_cluster = Clusterizacao(df_saida)
    
    
    #plot_aumento_e_reducao_por_anf(df_cluster)
#   df_cluster.to_excel(__DIR_SAIDA__ + '/PLOSS_multivariado.xlsx', index=False)
    df_cluster.to_excel(__DIR_SAIDA__ + '/PLOSS_multivariado_sequencial.xlsx', index=False)
    logging.info('Saved results to Excel file')

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f'Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s')
    print(f'Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s')

if __name__ == '__main__':
    main()

