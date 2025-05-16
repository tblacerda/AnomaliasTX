import pandas as pd
import datetime
import numpy as np
import sqlite3
from sqlite3 import Error
import scipy.stats as stats
from openpyxl.writer.excel import ExcelWriter
from openpyxl.workbook import Workbook
import seaborn as sns
from datetime import datetime, timedelta, date
from statistics import stdev
from tqdm import tqdm
import sys
import warnings
from pyfiglet import Figlet
import os.path
import os
import glob
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
# Total de dias para ler do BD
__TODAY__ = pd.to_datetime('2024-09-21') 
__DB_PLOSS__ = 'bd/PLOSS.banco'
#__TODAY__ = date.today()
__MULTIPLICADOR__ = 1 # quantas semanas para analisar, variando a data D-1
__DIR_SAIDA__ = 'SAIDA/'
__ARQUIVO_SAIDA__ = 'analise_estatistica.xlsx'   ######
__ARQUIVO_SITES_SEM_DADOS__ = 'Sites_sem_dados_PLOSS.xlsx'
__TARGET__ = 0.001   # 0.1%
__D_X__ = 1 # 1 é igual a D-1, 2 é igual a D-2
__DIAS_MAX__ = 45   # Populacao maxima considerada
__DIAS_MIN__ = 21   # Populacao minima considerada
__DIAS_AMOSTRA__ = 3 # ao menos __DIAS_AMOSTRA__ / 2. Amostra de dias recentes a ser avaliada 
__CONFIANCA__ = 0.98 # intervalo de confianca 0.95
__MEDIA_MOVEL__ = 1  # Dias da média movel para a janela longa.
__FILTRO__ = 3 # Quantos STD_DEV filtrar de outliers
__AVAIL__ = 0.80  # Considera apenas sites com 100% de amostras
__DIV__ = 3 # divide dias amostras para decidir se tem dados suficientes ou nao
#  se for 1, desabilita a média movel. Padrão 3.
# Data a partir da qual é para se ler do Banco de dados
D_BD = pd.to_datetime(__TODAY__ - timedelta(days=__DIAS_MAX__ + __DIAS_AMOSTRA__)).strftime("%Y-%m-%d")
# D_CORTE = pd.to_datetime(__TODAY__ - )
# Ajuste do diretorio Base para "Plano_de_Qualidade_2020"
#CURR_DIR = os.getcwd()
# if CURR_DIR.split("\\")[-1] == 'Plano_de_Qualidade_2020':
#     pass

# elif CURR_DIR.split("\\")[-1] == 'Scripts':
#     os.chdir("../")

CURR_DIR = os.getcwd()
print(CURR_DIR)

def BoasVindas():
    f = Figlet(font='5lineoblique')
    f = Figlet()
    print(f.renderText('Analise Estatistica'))
    

def JanelaCurta(dias):
    ''' Retorna os dias da janela curta em uma lista
        D_BD é a data para pegar no Banco de Dados. A partir desta data
    '''
    today = __TODAY__ # - timedelta(days=27)
    Janela_curta = []
    D_1 = pd.to_datetime(today - timedelta(days=__D_X__))
    # D_1 = D_1.strftime("%Y-%m-%d")
    for i, dia in  enumerate(range(1, dias + 1 ,1)):
        # print(i, dia)
        D = pd.to_datetime(today - timedelta(days=dia))
        # D = D.strftime("%Y-%m-%d")
        Janela_curta.append(D)

    return D_1, set(Janela_curta)


def testTstudent(mediaPopulacao, StdDevPop, df, confianca):
    '''
    Utiliza igual a função anterior, porém utiliza o pacote de estatistica
    do python
    '''
    intervalo = stats.t.interval(
        confidence=confianca,
        df=df,
        loc=mediaPopulacao,
        scale=StdDevPop)

    return intervalo


def remover_outliers(dataframe, coluna, num_desvios):
    '''
    Remove os outliers calculando o Z-score e removendo todos os
    casos com valor acima de num_desvios
    '''
    try:
        z_scores = stats.zscore(dataframe[coluna])
        abs_z_score = np.abs(z_scores)
        filtrado = (abs_z_score < num_desvios)
        new_df = dataframe[filtrado]
        return new_df
    except:
        return dataframe


def AnaliseEstatistica(df, janela_curta):
    '''
    Analise estatistica.
    Entrada: Tabela com Dia x Metrica
    Saida: string com 'constante', 'aumento', 'diminuição' ou 'sem dados'
    '''
    try:
        df.sort_values(by=['Dia'], ascending=True, inplace=True)
        df.dropna(inplace=True, axis=1)
        df = df.iloc[- __DIAS_MAX__:]
        #df = remover_outliers(df, 'value', __FILTRO__)
        df_amostra = df.copy(deep=True)
        df_amostra = df_amostra.query("Dia in @janela_curta")
        UltimaAmostra = df_amostra['value'].iloc[-1]
        df = df.iloc[:df.shape[0]-__DIAS_AMOSTRA__, :]
        df['MediaMovel'] = df['value'].rolling(__MEDIA_MOVEL__).mean()
        df = remover_outliers(df, 'value', __FILTRO__)
        df.dropna(how='any', axis=0, inplace=True)
        df_amostra = remover_outliers(df_amostra, 'value', __FILTRO__) # 30-04-2024
        mediaAmostra = df_amostra['value'].mean()
        desvioPadraoPop = stdev(df['MediaMovel'])
        N = df['MediaMovel'].count()
        mediaPop = df['MediaMovel'].mean()

        if len(janela_curta & set(df_amostra['Dia'])) >= ((__DIAS_AMOSTRA__) / __DIV__):
            intervalo = testTstudent(mediaPop, desvioPadraoPop, N-1, __CONFIANCA__)
            intervalo = list(intervalo)

            if df.shape[0] >= __DIAS_MIN__:
                resultado = 'Constante'

                if mediaAmostra > intervalo[1] and UltimaAmostra < intervalo[1]:
                     #if (intervalo[1] - UltimaAmostra) >= __TARGET__:
                    resultado = 'Normalizou em D-1'

                if mediaAmostra < intervalo[0] and UltimaAmostra < intervalo[0]:
                    if (intervalo[0] - mediaAmostra) >= __TARGET__:
                        resultado = 'Diminuição'

                elif mediaAmostra > intervalo[1] and UltimaAmostra > intervalo[1]:
                    if mediaAmostra > __TARGET__ and UltimaAmostra > __TARGET__:
                        resultado = 'Aumento'

                    # Corrige os intervalos negativos
                if intervalo[0] < 0:
                    intervalo[0] = 0
                if intervalo[1] > 100:
                    intervalo[1] = 100

                new_row = [mediaPop, intervalo[0], intervalo[1], mediaAmostra, desvioPadraoPop, resultado]
                return new_row
            else:
                resultado = 'Sem dados suficientes'
                print("dias na Janela longa: ", df.shape[0])
                print("dias amostra        : ", df_amostra.shape[0])
                new_row = [ mediaPop, -1, -1, mediaAmostra, desvioPadraoPop, resultado]
                return new_row
    except:
        resultado = 'Erro!'
        new_row = [mediaPop, -1, -1, mediaAmostra, desvioPadraoPop, resultado]
        return new_row


def AppendBb():
    '''
    Ler dados novos de uma planilha e anexar no BD
    '''
    PATH = r".\entrada"

    def p2f(x):
        x = x.replace(',', '.')
        if x.find('(') > -1:
            x = '0'
        return float(x.strip('%'))/100

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

    all_files = glob.glob(PATH + "\*.xls*")
    li = []
    all_files

    for filename in all_files:
        print(filename)
        temp = pd.read_excel(filename, skiprows=2)
        li.append(temp)


    
    df = pd.concat(li, axis=0, ignore_index=True)
    df.rename(columns = {'BTS/NodeB/EnodeB':'BTS/Nodeb/Enodeb',
                         'Regional NodeB':'Regional Nodeb',
                         'Estado NodeB': 'Estado Nodeb',
                         'ANF NodeB': 'ANF Nodeb',
                         'Cidade NodeB':'Cidade Nodeb',
                         }, inplace = True)
    df.drop_duplicates(inplace=True, subset=['Dia', 'BTS/Nodeb/Enodeb'])
#    df = df.loc[df['Availability Avg'] >= __AVAIL__]
    df = df.loc[(df['Regional Nodeb'] == "TNE")]
    df['Dia'] = pd.to_datetime(df['Dia'])
    UltimoDia = UltimaDataBanco(__DB_PLOSS__)
    UltimoDia = pd.to_datetime(pd.to_datetime(UltimoDia) + timedelta(days=1))
    df = df.loc[df['Dia'] >= UltimoDia]
    # Dias = df['Dia'].unique()

    conn = sqlite3.connect(__DB_PLOSS__)
    df.to_sql(name='PLOSS', con=conn, if_exists='append', index=False)
    conn.close()
    return np.nan


def LerBd(__AVAIL__):
# PLOSS
    conn = sqlite3.connect(__DB_PLOSS__)
    queryString = """SELECT Dia,
                "BTS/Nodeb/Enodeb", "Packet Loss Avg", "Availability Avg"
                FROM PLOSS WHERE "Regional Nodeb" IN ("TNE")  AND Dia >= ? order by Dia"""

    dfPloss = pd.read_sql(queryString, con=conn, params=[D_BD])
    conn.close()
    dfPloss = dfPloss.loc[dfPloss['Availability Avg'] >= __AVAIL__]
    dfPloss.drop("Availability Avg", axis=1, inplace=True)
    dfPloss.rename({'Station ID' : 'END_ID',
                    'BTS/Nodeb/Enodeb' : 'Site',
                    'Packet Loss Avg' : 'value'},
                    axis=1, inplace=True)

    dfPloss.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfPloss.dropna(how='any', subset=['value'], inplace=True)
    dfPloss.drop_duplicates(inplace=True, subset=['Dia', 'Site'])
    dfPloss['value'] = dfPloss['value'].round(3)
    dfPloss['Dia'] = pd.to_datetime(dfPloss['Dia'])

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


###

def AgrupaPorEndId(df):
    ''' O objetivo é completar os dias faltando de um SITE ID 
        com os dados de outro SITE ID do mesmo END_ID
    '''

    dfEndId = pd.read_excel('bd\ENDID_SITEID.xlsx') # manter atualizado
    df = pd.merge(df, dfEndId, how='inner', on='Site')
    df.drop('Site', axis = 1, inplace=True)
    df1 = df.groupby(['Endid', 'Dia'])['value'].mean().reset_index()
    df = pd.merge(df1, dfEndId, how='inner', on='Endid')
    df.drop('Endid', axis = 1, inplace=True)

    return df
###

def main():
    # DataFrame de Saida
    BoasVindas()
    AppendBb()
    dfConsolidado = LerBd(__AVAIL__)
    dfConsolidado= AgrupaPorEndId(dfConsolidado) #opcional abr-2024


    Colunas = ['Corte',
               'Site',
               'Tech',
               'Indicador',
               'Media populacional',
               'Limite Inferior',
               'Limite Superior',
               'Media amostral',
               'desvioPop',
               'Tendencia']

    list_saida = []

    SITES = dfConsolidado['Site'].unique() 
    # SITES = SITES[:2]
    Tech = '4G'
    Indicador = 'PLOSS'
    D_1 , janela_curta = JanelaCurta(__DIAS_AMOSTRA__)
    D_1
    janela_curta
    D_1 = D_1.strftime("%d-%m-%Y")
    for site in tqdm(SITES):
        sys.stdout.flush()
        df_site = dfConsolidado.query("Site in @site")
        #df_site = df_site.fillna(method='ffill', limit=1).fillna(method='bfill', limit=1)
        if df_site.shape[0] >=  __DIAS_MIN__:
            resultado = 'Sem dados suficientes'
            try:
                NEWROW = AnaliseEstatistica(df_site, janela_curta) # 6
                NEWROW.insert(0, Indicador)
                NEWROW.insert(0, Tech)
                NEWROW.insert(0, site)
                NEWROW.insert(0, D_1)
            except Exception:
                resultado = "Sem dados suficientes"
                NEWROW = [D_1,site, Tech, Indicador, -1, -1, -1, -1, -1, resultado]
        else:
            resultado = "Sem dados suficientes"
            NEWROW = [D_1,site, Tech, Indicador, -1, -1, -1, -1, -1, resultado]
        list_saida = list_saida + [NEWROW]

    df_saida = pd.DataFrame(data=list(filter(None, list_saida)),
                            columns=Colunas)
    df_saida = IncluirMunUfAnf(df_saida)
    df_saida.to_excel(__DIR_SAIDA__+__ARQUIVO_SAIDA__, index=False)


    # Filtrar o data frame para selecionar apenas as linhas onde a coluna "tendência" é igual a "aumento"
    df_aumento = df_saida.loc[df_saida["Tendencia"] == "Aumento"]

    df_acima2pc = df_saida.loc[df_saida["Media amostral"] >= 0.02]

    # Agrupar o data frame filtrado pela coluna ANF e contar quantas vezes cada valor aparece
    df_agrupado = df_aumento.groupby("ANF").size()

    # Criar o gráfico de barras usando o matplotlib
    df_agrupado.plot.bar()
    plt.xlabel("ANF")
    plt.ylabel("Quantidade")
    plt.title("Quantidade de sites com tendência de aumento por UF")
    plt.show()  

    # Agrupar o data frame filtrado pela coluna ANF e contar quantas vezes cada valor aparece
    df_agrupado = df_acima2pc.groupby("ANF").size()

    # Criar o gráfico de barras usando o matplotlib
    df_agrupado.plot.bar()
    plt.xlabel("ANF")
    plt.ylabel("Quantidade")
    plt.title("Quantidade de sites com PLOSS acima de 2% por UF")
    plt.show()  

    df_saida['Tendencia'].value_counts()


if __name__ == '__main__':
    main()

