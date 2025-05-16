import pandas as pd
import numpy as np
from numpy import sqrt as sqrt
import scipy.stats as stats
from scipy.stats import t as student_t
from statistics import stdev
from tqdm import tqdm
from datetime import datetime, date
  
_DATA_HORA_COLUMN_ = 'Data'
_DATA_COLUMN_ = 'data'
_HORA_COLUMN_ = 'hora'
_INDICE_ = 'KPI'
_DROP_ = ['Region', 'UF', 'ANF']

# no primeiro dia:
# com amostras de 0 a 13h (14 amostras) so pode usar Cohen'd de 1.3
# as 20h (21 amostras) pode-se utilizar 1.2
# com 24 amostras, cohen's d = 1
# com 48 amostras, cohen's d = 0.7 

class TOST_service:
    '''
    Classe para o servico de analise TOST
    Two One-Sided T-test
    ''' 
    def TOSTtwo(self, m1, m2,
                sd1, sd2,
                n1, n2,
                low_eqbound_d,
                high_eqbound_d,
                alpha = 0.05,
                var_equal = False,
                plot = False,
                verbose = False):
        '''
        TOST function for an independent t-test (Cohen's d)
        @param m1 mean of group 1
        @param m2 mean of group 2
        @param sd1 standard deviation of group 1
        @param sd2 standard deviation of group 2
        @param n1 sample size in group 1
        @param n2 sample size in group 2
        @param low_eqbound_d lower equivalence bounds (e.g., -0.5) expressed in standardized mean 
        difference (Cohen's d)
        @param high_eqbound_d upper equivalence bounds (e.g., 0.5) expressed in standardized mean 
        difference (Cohen's d)
        @param alpha alpha level (default = 0.05)
        @param var.equal logical variable indicating whether equal variances assumption is assumed to 
        be TRUE or FALSE.  Defaults to FALSE.
        @param plot set whether results should be plotted (plot = TRUE) or not (plot = FALSE) - defaults
        to TRUE
        @param verbose logical variable indicating whether text output should be generated (verbose 
        = TRUE) or not (verbose = FALSE) - default to TRUE
        @return Returns TOST t-value 1, TOST p-value 1, TOST t-value 2, TOST p-value 2, degrees of 
        freedom, low equivalence bound, high equivalence bound, low equivalence bound in Cohen's d,
        high equivalence bound in Cohen's d, Lower limit confidence interval TOST, Upper limit 
        confidence interval TOST
        @importFrom stats pnorm pt qnorm qt
        @importFrom graphics abline plot points segments title
        @examples
        ## Eskine (2013) showed that participants who had been exposed to organic
        ## food were substantially harsher in their moral judgments relative to
        ## those exposed to control (d = 0.81, 95% CI: [0.19, 1.45]). A
        ## replication by Moery & Calin-Jageman (2016, Study 2) did not observe
        ## a significant effect (Control: n = 95, M = 5.25, SD = 0.95, Organic
        ## Food: n = 89, M = 5.22, SD = 0.83). Following Simonsohn's (2015)
        ## recommendation the equivalence bound was set to the effect size the
        ## original study had 33% power to detect (with n = 21 in each condition,
        ## this means the equivalence bound is d = 0.48, which equals a
        ## difference of 0.384 on a 7-point scale given the sample sizes and a
        ## pooled standard deviation of 0.894). Using a TOST equivalence test
        ## with default alpha = 0.05, not assuming equal variances, and equivalence
        ## bounds of d = -0.43 and d = 0.43 is significant, t(182) = -2.69,
        ## p = 0.004. We can reject effects larger than d = 0.43.
        TOSTtwo(m1=5.25,m2=5.22,sd1=0.95,sd2=0.83,n1=95,n2=89,low_eqbound_d=-0.43,high_eqbound_d=0.43)
        @section References:
        Berger, R. L., & Hsu, J. C. (1996). Bioequivalence Trials, Intersection-Union Tests
        and Equivalence Confidence Sets. Statistical Science, 11(4), 283-302.
        Gruman, J. A., Cribbie, R. A., & Arpin-Cribbie, C. A. (2007).
        The effects of heteroscedasticity on tests of equivalence. 
        Journal of Modern Applied Statistical Methods, 6(1), 133-140, 
        formula for Welch's t-test on page 135
        
        THIS FUNCTION WAS REPLICATED FOR EDUCATIONAL PURPOSES AS IT WAS REPLACED BY
        tsum_TOST wich is better designed and has a broader usage.
        Return list
        '''

        if  (n1 < 2) or (n2 < 2):
            return "The sample size should be larger than 1."

        if (1<=alpha or alpha < 0):
            return "The alpha level should be a positive value between 0 and 1."
        
        if (sd1 <= 0 or sd2 <=0):
            return "The standard deviation should be a positive value."
        
        ## Fim dos checks
        # Calculate TOST, t-test, 90% CIs and 95% CIs
        
        if var_equal == True:
            sdpooled = sqrt((((n1 - 1)*(sd1**2))+(n2 - 1)*(sd2**2))/((n1+n2)-2))
            low_eqbound = low_eqbound_d*sdpooled
            high_eqbound = high_eqbound_d*sdpooled
            degree_f = n1+n2-2
        
            dist = student_t(df=degree_f,loc=0,scale=1 )


            t1 = ((m1-m2)-low_eqbound)/(sdpooled*sqrt(1/n1 + 1/n2))  #students t-test lower bound
            lower_tail_false = 1- dist.cdf(t1)  
            p1 = lower_tail_false 
            t2 = ((m1-m2)-high_eqbound)/(sdpooled*sqrt(1/n1 + 1/n2)) #students t-test upper bound
            lower_tail_true = dist.cdf(t2)
            p2 = lower_tail_true
            
            t = (m1-m2)/(sdpooled*sqrt(1/n1 + 1/n2))
            
            lower_tail_true2 = dist.cdf(-abs(t))
            pttest = 2*lower_tail_true2
            
            LL90 = (m1-m2)-student_t.ppf(1-alpha, n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
            UL90 = (m1-m2)+student_t.ppf(1-alpha, n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
            LL95 = (m1-m2)-student_t.ppf(1-(alpha/2), n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
            UL95 = (m1-m2)+student_t.ppf(1-(alpha/2), n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
        else:
            sdpooled = sqrt((sd1**2 + sd2**2)/2) #calculate sd root mean squared for Welch's t-test
            low_eqbound = low_eqbound_d*sdpooled
            high_eqbound = high_eqbound_d*sdpooled
            degree_f = (sd1**2/n1+sd2**2/n2)**2/(((sd1**2/n1)**2/(n1-1))+((sd2**2/n2)**2/(n2-1))) #degrees of freedom for Welch's t-test        
            dist = student_t(df=degree_f,loc=0,scale=1 )
            t1 = ((m1-m2)-low_eqbound)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test upper bound
            lower_tail_false = 1- dist.cdf(t1)  
            p1 = lower_tail_false 
            t2 = ((m1-m2)-high_eqbound)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test lower bound
            lower_tail_true = dist.cdf(t2)
            p2 = lower_tail_true
            t = (m1-m2)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test NHST    
            lower_tail_true2 = dist.cdf(-abs(t))
            pttest = 2*lower_tail_true2
        
            LL90 = (m1-m2)-student_t.ppf(1-alpha, degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Lower limit for CI Welch's t-test
            UL90 = (m1-m2)+student_t.ppf(1-alpha, degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Upper limit for CI Welch's t-test
            LL95 = (m1-m2)-student_t.ppf(1-(alpha/2), degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Lower limit for CI Welch's t-test
            UL95 = (m1-m2)+student_t.ppf(1-(alpha/2), degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Upper limit for CI Welch's t-test
    
        ptost = max(p1,p2) #Get highest p-value for summary TOST result
        ttost = t2
        if (abs(t1) < abs(t2)):
            ttost = t1
    
        dif = (m1-m2)
        testoutcome = "non-significant"
        
        if pttest < alpha:
            testoutcome = "significant"
        
        TOSToutcome = "non-significant"
        if ptost<alpha:
            TOSToutcome = "significant"
        
        if verbose == True:

            print("TOST Results:")
            print(80*"=")
            print("t-value lower bound: %0.4f ; tp-value lower bound: %0.4f"%(t1, p1))
            print("t-value upper bound: %0.4f ; tp-value upper bound: %0.4f"%(t2, p2))
            print("Degrees of freedom: %0.2f"%(round(degree_f, 2)))
            print("Equivalence bounds (Cohen's d): low eqbound: %0.4f ; high eqbound: %0.4f"%(low_eqbound_d, high_eqbound_d))
            print("TOST confidence interval: lower bound %0.4f CI: %0.4f; upper bound %0.4f CI: %0.4f"%((100*1-alpha*2),round(LL90,3),(100*1-alpha*2),round(UL90,3)))
            print("NHST confidence interval: lower bound %0.4f CI: %0.4f; upper bound %0.4f CI: %0.4f"%((100*1-alpha),round(LL95,3),(100*1-alpha),round(UL95,3)))
            print("\nEquivalence Test Result:")
            print(80*"=")
            print("The equivalence test was %s, t(%0.2f) = %0.4f, p = %0.4f, given equivalence bounds of %0.4f and %0.4f (on a raw scale) and an alpha of %0.3f"%(TOSToutcome, degree_f,ttost, ptost, low_eqbound, high_eqbound, alpha))
            
            print("\nNull Hypothesis Test Result:")
            print(80*"=")
            print("The null hypothesis test was %s, t(%0.4f) = %0.4f, p = %0.4f, given an alpha of %0.3f"%(testoutcome, degree_f, t, pttest, alpha))

            if (pttest <= alpha and ptost <= alpha):
                combined_outcome = "NHST: reject null significance hypothesis that the effect is equal to 0. \n TOST: reject null equivalence hypothesis."
            
            if (pttest < alpha and ptost > alpha):
                combined_outcome = "NHST: reject null significance hypothesis that the effect is equal to 0. \n TOST: Don't reject null equivalence hypothesis."

            if (pttest > alpha and ptost <= alpha):
                combined_outcome = "NHST: Don't reject null significance hypothesis that the effect is equal to 0. \n TOST: reject null equivalence hypothesis."
            
            if (pttest > alpha and ptost > alpha):
                combined_outcome = "NHST: Don't reject null significance hypothesis that the effect is equal to 0. \n TOST: Don't reject null equivalence hypothesis."
            print("\nOutcome:\n %s"%(combined_outcome))
            print(80*"=")
            saida = [dif, t1, p1, t2, p2, degree_f, low_eqbound, high_eqbound, low_eqbound_d, high_eqbound_d,
                LL90, UL90, LL95, UL95, t, pttest]
        
        
        if verbose == False:
            saida = 'Inconclusivo'
            # Equivalente, Não-equivalente ou inconclusivo
            if pttest > alpha and ptost <= alpha:
                saida = "Equivalente" #A
                
            if pttest <= alpha and ptost > alpha:
                saida = "Não equivalente" # B
                
            if pttest <= alpha and ptost<= alpha:
                saida = "Equivalente" #C
            
            if pttest > alpha and ptost > alpha:
                saida = "Inconclusivo"
                
        return saida


    def Pre_ProcessMS(self, df_concat,data_column, indice,
                    remover):
        '''
        Preprocessa o arquivo excel de entrada
        file_path: arquivo para leitura
        sheet: aba
        data_column: nome da columna com a data
        indice: nome da cidade
        remover: colunas para remover
        dia_virada: data da mudança
        
        saida 
        '''
        
        df = df_concat
        _DATA_COLUMN_ = "Dia"
        # df[_HORA_COLUMN_] = pd.to_datetime(df[data_column]).dt.hour
        # df_temp = df.pop(_HORA_COLUMN_)
        # df.insert(0,_HORA_COLUMN_,df_temp)
        # df[_DATA_COLUMN_] = pd.to_datetime(df[data_column]).dt.date
        # df[_DATA_COLUMN_] = pd.to_datetime(df[_DATA_COLUMN_], format='%Y-%m-%d')
        
        # df_temp = df.pop(_DATA_COLUMN_)
        # df.insert(0,_DATA_COLUMN_,df_temp)
        # df.drop(columns=_DATA_HORA_COLUMN_, inplace=True)
        # df.drop(columns=remover , inplace = True)
        
        df = df.melt(id_vars=[_DATA_COLUMN_, indice], var_name='KPI')

        return df

    def Pre_Process(self, file_path, sheet,data_column, indice,
                    remover):
        '''
        Preprocessa o arquivo excel de entrada
        file_path: arquivo para leitura
        sheet: aba
        data_column: nome da columna com a data
        indice: nome da cidade
        remover: colunas para remover
        dia_virada: data da mudança
        
        saida 
        '''
        
        df = pd.read_excel(file_path, sheet)
        df[_HORA_COLUMN_] = pd.to_datetime(df[data_column]).dt.hour
        df_temp = df.pop(_HORA_COLUMN_)
        df.insert(0,_HORA_COLUMN_,df_temp)
        df[_DATA_COLUMN_] = pd.to_datetime(df[data_column]).dt.date
        df[_DATA_COLUMN_] = pd.to_datetime(df[_DATA_COLUMN_], format='%Y-%m-%d')
        
        df_temp = df.pop(_DATA_COLUMN_)
        df.insert(0,_DATA_COLUMN_,df_temp)
        df.drop(columns=_DATA_HORA_COLUMN_, inplace=True)
        df.drop(columns=remover , inplace = True)
        
        df = df.melt(id_vars=[_DATA_COLUMN_, _HORA_COLUMN_, indice], var_name='KPI')

        return df
    
    def Analisar_equivalencia(self, df, low_eq_bound, high_eq_bound,
                              data_col, Col_indice, data_corte):
        '''
        Utilizando a função TOSTtwo e os valores de equivalencia,
        analisa todos os indices quanto a todas as colunas
        data_corte: data que separa antes e depois da condicao analisada
        df : data frame
        low_eq_bound: cohen's d low
        high_eq_bound: cohen's d high
        data_col: nome da coluna com a data
        indice: coluna com o indice utilizado. pode ser municipio, site ...
        outras colunas: sao todas numericas para serem analisadas.
        IMPORTANTE: assume-se que todas as colunas depois do indice são de dados
        a serem analisados
        padrao: |DATA|HORA|INDICE|..... dados .....|
        '''
        # Para cada indice....
        #   Para cada kpi....
        #   divide em duas tabelas de acordo com a data_corte: 
        #   df_referencia e df_evento. depois da data_corte
        #   Verifica a hora_max df_evento e, faz com que df_referencia só tenha 
        #   até esse horario também
        #   Faz os calculos necessários para o TOST
        #   Chama o TOST
        #   Guarda o resultado em outro DataFrame, df_resultado
        #   devolve o df_resultado.

        Colunas = [Col_indice,
                  'KPI',
                  'Ref_N',
                  'Ref_media',
                  'Ref_std',
                  'Evento_N',
                  'Evento_media',
                  'Evento_std',
                  'Resultado']
        list_saida=[]
        Resultado = 'Não executado'


        data_cut = datetime.strptime(data_corte,"%Y-%m-%d")
        df.rename({Col_indice:'indice'}, axis=1, inplace=True)
        indices = df['indice'].unique()
        for indice in tqdm(indices):
            df_indices = df.query("indice in @indice")
            indicadores = df_indices['KPI'].unique()
            df_ref = df_indices.loc[df_indices[data_col] < data_cut]
            df_evento = df_indices.loc[df_indices[data_col] >= data_cut]
            
            
            
            for indicador in indicadores:
                Resultado = 'Não executado'
                df_temp_ref = df_ref.query("KPI in @indicador")
                average_ind_ref = df_temp_ref['value'].mean()
                std_ind_ref = df_temp_ref['value'].std()
                N_ind_ref = df_temp_ref['value'].count()
                
                df_temp_evento = df_evento.query("KPI in @indicador")
                average_ind_evento = df_temp_evento['value'].mean()
                std_ind_evento = df_temp_evento['value'].std()
                N_ind_evento = df_temp_evento['value'].count()
               
                Resultado = self.TOSTtwo(m1 = average_ind_ref,
                                    m2 = average_ind_evento,
                                    sd1 = std_ind_ref,
                                    sd2 = std_ind_evento,
                                    n1 = N_ind_ref,
                                    n2 = N_ind_evento,
                                    low_eqbound_d = low_eq_bound,
                                    high_eqbound_d = high_eq_bound)
                if Resultado == 'Não equivalente':
                    if average_ind_evento > average_ind_ref:
                        Resultado = "Aumentou"
                    if average_ind_evento < average_ind_ref:
                        Resultado = "Reduziu"
                        
                NEWROW = [indice,
                          indicador,
                          N_ind_ref,
                          round(average_ind_ref, 2),
                          round(std_ind_ref, 2),
                          N_ind_evento,
                          round(average_ind_evento, 2),
                          round(std_ind_evento, 2),
                          Resultado]
                list_saida = list_saida + [NEWROW]

            #KPIs = df_indice

        df_resultado = pd.DataFrame(data=list(filter(None, list_saida)) , columns=Colunas)

        return df_resultado

if __name__ == "__main__":
    TOASTER_4G = TOST_service()
#    TOASTER_3G = TOST_service()
#    TOASTER_2G = TOST_service()
    
    #concatenar bases
    df1 = pd.read_excel(r'dados\D-1_LTE_MAIO.xlsx')
    df2 = pd.read_excel(r'dados\D-1_LTE_JUNHO.xlsx')
    df_concat = pd.concat([df1,df2])
        
    df_concat = df_concat[df_concat['DISP_COUNTER_TOTAL 4G (com filtro OPER)']>=0.95]
    df_concat.drop([ 'Município','Unnamed: 2', 'Faixa Populacional', 'Regional',
                                'ANF','Estado', 'Operadora', 'ANF'], axis=1, inplace=True)
    #Microstrategy preprossess


    df_concat.drop(['Station ID','Metrics', 'Hora'], axis=1, inplace=True)
    df_4g= TOASTER_4G.Pre_ProcessMS(df_concat, 'Dia','RAN Node',
                               ['Hora'])

    analise_4G = TOASTER_4G.Analisar_equivalencia(df_4g,-0.75,0.75,'Dia',
                                                  'RAN Node','2023-06-01')
    
    analise_4G.to_excel('Analise_TOST_4G.xlsx', index=False)