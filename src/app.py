import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import requests  # Import the requests library

from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from urllib.request import Request, urlopen

import gradio as gr
import tiktoken
import time
import openai
from openai import OpenAI

client = OpenAI(api_key="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")


urls = ['https://portal.ifba.edu.br/','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-integrados---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-integrados-proeja---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-subsequentes---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-concomitantes---','https://portal.ifba.edu.br/campi/escolhacampus']

domain = "portal.ifba.edu.br"

url = "https://portal.ifba.edu.br/"


import hashlib

def get_text_from_url(url):
    '''
        Essa função recebe uma url e faz o scraping do texto da página
        Retorna o texto da página e salva o texto em um arquivo <url>.txt
    '''
    
    # Analisa a URL e pega o domínio
    local_domain = urlparse(url).netloc
    print(local_domain)

    # Fila para armazenar as urls para fazer o scraping
    fila = deque(urls)
    print(fila)

    # Criar um diretório para armazenar os arquivos de texto
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # Enquanto a fila não estiver vazia, continue fazendo o scraping
    while fila:
        # Pega a próxima URL da fila
        url = fila.pop()
        print("Próxima url",url) # Checa próxima url

        # Salva o texto da url em um arquivo <url>.txt
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:
            driver = Chrome()
            driver.get(url)
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            text = page_soup.get_text()
            f.write(text)

        driver.close()
#get_text_from_url(url)

def remove_newlines(serie):
    '''
        Essa função recebe uma série e remove as quebras de linha
    '''
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Criar uma lista para armazenar os arquivos de texto
texts=[]
# Obter todos os arquivos de texto no diretório de texto
for file in os.listdir("text/" + domain + "/"):
    # Abra o arquivo e leia o texto
    with open("text/" + domain + "/" + file, "r") as f:
        text = f.read()
        # Omita as primeiras 20 linhas e as últimas 4 linhas e, em seguida, substitua  -, _, e #update com espaços.
        texts.append((file[20:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

#Criar um Dataframe a partir da lista de textos
df = pd.DataFrame(texts, columns = ['page_name', 'text'])

# Defina a coluna de texto para ser o texto bruto com as novas linhas removidas
df['text'] = df.page_name + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

# Carregar o tokenizador cl100k_base que foi projetado para funcionar com o modelo ada-002
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize o texto e salve o número de tokens em uma nova coluna
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize a distribuição do número de tokens por linha usando um histograma
df.hist(column='n_tokens')


max_tokens = 500

# Função para dividir o texto em partes de um número máximo de tokens
def split_into_many(text, max_tokens = max_tokens):

    # Dividir o texto em frases
    sentences = text.split('. ')

    # Obter o número de tokens para cada sentença
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Percorrer as sentenças e os tokens unidos em uma tupla
    for sentence, token in zip(sentences, n_tokens):

        # Se o número de tokens até o momento mais o número de tokens na frase atual for maior 
        # do que o número máximo de tokens, adicione o bloco à lista de blocos e redefina
        # o bloco e os tokens até o momento
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # Se o número de tokens na frase atual for maior que o número máximo de 
        # tokens, vá para a próxima sentença
        if token > max_tokens:
            continue

        # Caso contrário, adicione a frase ao bloco e adicione o número de tokens ao total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

# Percorrer o dataframe
for row in df.iterrows():

    # Se o texto for None, vá para a próxima linha
    if row[1]['text'] is None:
        continue

    # Se o número de tokens for maior que o número máximo de tokens, divida o texto em partes
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Caso contrário, adicione o texto à lista de textos abreviados
    else:
        shortened.append( row[1]['text'] )


df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.hist(column='n_tokens')

num_tot_tokens = df['n_tokens'].sum()
print("Número total de tokens",num_tot_tokens)

df.head()

i = 0
for text in df['text']:
    i+=1
    
print("Número de trechos de texto com no máximo",max_tokens,"tokens :",i)

print("Custo total de treinamento do embedding: $",num_tot_tokens /1000 * 0.0001)

#def read_openai_api_key():
#    with open('openai_secret.txt', 'r') as file:
#        api_key = file.read().strip()
#    return api_key

#my_api_key = read_openai_api_key()
my_api_key = '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'

#openai.api_key = read_openai_api_key()
openai.api_key = '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'

# Verificar se o arquivo 'embeddings.csv' existe
embeddings_file_path = 'processed/embeddings.csv'
if os.path.exists(embeddings_file_path):
    # Se existir, carregar os embeddings do arquivo
    df = pd.read_csv(embeddings_file_path, index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    print("Embeddings carregados do arquivo 'embeddings.csv'")
else:
    # Se não existir, criar os embeddings
    i = 0
    embeddings = []
    for text in df['text']:
        time.sleep(0)
        print(i)
        try:
            embedding = client.embeddings.create(input=text, model='text-embedding-3-small').data[0].embedding
            print("Fazendo embedding do texto")
            embeddings.append(embedding)
        except openai.APIConnectionError as e:
            if e:
                print("O servidor não pôde ser alcançado")
                print(e.__cause__)  # uma Exception subjacente, provavelmente levantada dentro do httpx.
                break
        except openai.RateLimitError as e:
            if e:
                print("Foi recebido um código de status 429; devemos recuar um pouco.")
                break
        except openai.APIStatusError as e:
            if e:
                print("Outro código de status fora da faixa 200 foi recebido")
                print(e.status_code)
                print(e.response)
                break
        i += 1

    df['embeddings'] = embeddings
    df.to_csv(embeddings_file_path)
    print("Embeddings salvos no arquivo 'embeddings.csv'")
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

import numpy as np
from scipy.spatial.distance import cdist

def distances_from_embeddings(q_embeddings, embeddings, distance_metric='cosine'):
    """
    Calcula as distâncias entre o embedding de consulta e um conjunto de embeddings usando uma métrica de distância específica.
    
    Parâmetros:
    - q_embeddings: O embedding da consulta (pode ser uma lista ou array).
    - embeddings: Um array ou DataFrame de embeddings.
    - distance_metric: A métrica de distância a ser utilizada (por exemplo, 'cosine' para a distância cosseno).

    Retorna:
    - Um array de distâncias entre o embedding de consulta e os embeddings fornecidos.
    """
    try:
        # Converte q_embeddings para um array NumPy, se não for do tipo array
        q_embeddings = np.array(q_embeddings)

        # Converte embeddings para um array NumPy, se não for do tipo array
        embeddings = np.array(embeddings)

        # Imprime as formas dos embeddings
        print("Shape of q_embedding:", q_embeddings.shape)
        print("Shape of embeddings:", embeddings.shape)



        # Verifica a métrica de distância e calcula as distâncias
        if distance_metric == 'cosine':
            distances = cdist(q_embeddings, embeddings, metric='cosine')[0]
        elif distance_metric == 'euclidean':
            distances = cdist(q_embeddings, embeddings, metric='euclidean')[0]
        else:
            raise ValueError("Métrica de distância inválida.")

        # Adiciona as distâncias ao DataFrame
        embeddings['distances'] = distances
    except openai.APIStatusError as e:
        print(e.status_code)
    except ValueError as e:
        print("Métrica de distância inválida.", e)
    except KeyError as e:
        print("Chave inexistente no dataframe:", e)
    except IndexError as e:
        print("Índice do embedding da consulta inválido.", e)
    except Exception as e:
        print("Erro inesperado:", e)
    return embeddings




def create_context(question, df, max_len=1800, size="ada"):

    try:
        # Obter a embeddings para a pergunta que foi feita
        q_embeddings = client.embeddings.create(input=question,model='text-embedding-ada-002').data[0].embedding

        # Obter as distâncias a partir dos embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


        returns = []
        cur_len = 0

        # Classifique por distância e adicione o texto ao contexto
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            
            # Adicionar o comprimento do texto ao comprimento atual
            cur_len += row['n_tokens'] + 4
            
            # Se o contexto for muito longo, quebre
            if cur_len > max_len:
                break
            
            # Caso contrário, adicione-o ao texto que está sendo retornado
            returns.append(row["text"])

        # Retornar o contexto
        return "\n\n###\n\n".join(returns)
    except Exception as e:
        print('Erro na hora de criar contexto', e)

def answer_question(
                    df=df,
                    model="gpt-3.5-turbo-instruct",
                    question="O que é o IFBA?",
                    max_len=1800,
                    size="ada",
                    debug=False,
                    max_tokens=150,
                    stop_sequence=None):
    """
    Responder a uma pergunta com base no contexto mais semelhante dos textos do dataframe
    """
    context = create_context(
        question,
        df=df,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = client.completions.create(
            prompt=f"Responda as perguntas com base no contexto abaixo, e se a pergunta não puder ser respondida diga \"Eu não sei responder so\"\nContexto: {context}\n\n---\n\nPergunta: {question}\nResposta:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
    #     retornar ""
      
answer_question(question="O que é o IFBA?", debug=False)
answer_question(df, question="Quais cursos técnicos tem no IFBA campus camaçari?")
answer_question(df, question="Qual o contato do IFBA?")

def chatgpt_clone(input, history):
     history= history or []
     s = list(sum(history, ()))
     s.append(input)
     inp = ' '.join(s)
     output=answer_question(question = inp)
     history.append((input, output))
     return history, history

css = """
.gradio-container {background-color: black}

"""

with gr.Blocks(theme=gr.themes.Soft(),css=css) as block:
     gr.Markdown("""<h1><center> Assistente do IFBA</center></h1>""")
     chatbot=gr.Chatbot(label="Conversa")
     message=gr.Textbox(label="Faça sua pergunta",placeholder="O que você gostaria de saber sobre o IFBA?")
     state = gr.State()
     submit = gr.Button("Perguntar")
     submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])
block.launch(debug=True, share=True)
