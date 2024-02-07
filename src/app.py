import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import requests  # Import the requests library

from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urlparse
from selenium.webdriver import Chrome

import gradio as gr
import tiktoken
import time
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('openaisecret'))

urls = ['https://portal.ifba.edu.br/','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-integrados---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-integrados-proeja---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-subsequentes---','https://portal.ifba.edu.br/ensino/nossos-cursos/curso-tecnico/tecnicos-concomitantes---','https://portal.ifba.edu.br/campi/escolhacampus']

domain = "portal.ifba.edu.br"

url = "https://portal.ifba.edu.br/"

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

# Criar uma lista para armazenar os arquivos d texto
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
from scipy.spatial.distance import cosine

def create_context(question, df, max_len=1800):
    try:
        # Obter os embeddings para a pergunta
        q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding
        
        # Calcular as distâncias a partir dos embeddings da pergunta
        df['distances'] = df['embeddings'].apply(lambda x: cosine(q_embeddings, np.array(x)))

        returns = []
        cur_len = 0

        # Classificar por distância e adicionar o texto ao contexto
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            # Adicionar o comprimento do texto ao comprimento atual
            cur_len += row['n_tokens'] + 4
            
            # Se o contexto for muito longo, interromper
            if cur_len > max_len:
                break
            
            # Caso contrário, adicionar ao texto retornado
            returns.append(row["text"])

        # Retornar o contexto
        return "\n\n###\n\n".join(returns)
    except Exception as e:
        print('Erro ao criar contexto:', e)

# Testar a função
contexto = create_context("Quais cursos técnicos tem no IFBA campus camaçari?", df)
print('contexto: ',contexto)

print(df['distances'])

def answer_question(
                    df=df,
                    model="gpt-3.5-turbo-instruct",
                    question="",
                    max_len=1800,
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
    )

    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = client.completions.create(
            prompt=f"Converse e responda as perguntas com base no contexto abaixo e se a pergunta não puder ser respondida diga \nContexto: {context}\n\n---\n\nPergunta: {question}\nResposta:",
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model= model,
        )
        print('=-'*50)
        print(response)
        print('=-'*50)
        # Retorna o texto da primeira escolha (choice) da resposta
        response =  response.choices[0].text.strip()
        return response

    except Exception as e:
        print('Erro no repondendo questão: ',e)

pergunta = ''
while pergunta != 'sair':
    pergunta = input('Digite sua pergunta: ')
    answer_question(question=pergunta, debug=False)
